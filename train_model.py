from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from feature_engineering import (
    BASE_CATEGORICAL_FIELDS,
    CROSS_FIELD_SPECS,
    NUMERIC_FEATURE_NAMES,
    AggregateStats,
    build_aggregate_stats,
    build_context,
    build_numeric_features,
    build_sparse_indices,
    dump_json,
    load_episode_features,
    load_user_features,
    log,
    maybe_log_progress,
    positive_int,
    should_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final DeepFM model for repeat-play probability prediction."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/splits/stratified_train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/splits/stratified_val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--user-feature-csv", type=Path, default=Path("data/user_feature.csv"))
    parser.add_argument("--episode-feature-csv", type=Path, default=Path("data/episode_feature.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_deepfm"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--deep-dims", type=int, nargs="+", default=(128, 64))
    parser.add_argument("--hash-bucket-size", type=int, default=800_000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200_000)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class CTRDataset(IterableDataset):
    def __init__(
        self,
        csv_path: Path,
        stats: AggregateStats,
        user_features: dict[str, dict],
        episode_features: dict[str, dict],
        alpha: float,
        bucket_size: int,
        max_rows: int | None,
        log_every: int,
        has_label: bool,
        training_mode: bool,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.stats = stats
        self.user_features = user_features
        self.episode_features = episode_features
        self.alpha = alpha
        self.bucket_size = bucket_size
        self.max_rows = max_rows
        self.log_every = log_every
        self.has_label = has_label
        self.training_mode = training_mode

    def __iter__(self):
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                if should_stop(row_idx, self.max_rows):
                    break
                context = build_context(row, self.user_features, self.episode_features)
                label = positive_int(row["label"]) if self.has_label else 0
                dense = build_numeric_features(
                    context=context,
                    stats=self.stats,
                    alpha=self.alpha,
                    label=label if self.training_mode and self.has_label else None,
                    is_training_row=self.training_mode and self.has_label,
                )
                sparse = build_sparse_indices(context, self.bucket_size)
                sample = {
                    "dense": dense,
                    "sparse": sparse,
                    "id": row.get("id", str(row_idx)),
                }
                if self.has_label:
                    sample["label"] = label
                maybe_log_progress(row_idx + 1, self.log_every, self.csv_path.name)
                yield sample


def collate_batch(batch: list[dict]) -> dict:
    dense = torch.tensor([item["dense"] for item in batch], dtype=torch.float32)
    sparse = torch.tensor([item["sparse"] for item in batch], dtype=torch.long)
    payload = {"dense": dense, "sparse": sparse, "id": [item["id"] for item in batch]}
    if "label" in batch[0]:
        payload["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return payload


def make_loader(
    csv_path: Path,
    stats: AggregateStats,
    user_features: dict[str, dict],
    episode_features: dict[str, dict],
    alpha: float,
    bucket_size: int,
    max_rows: int | None,
    log_every: int,
    batch_size: int,
    has_label: bool,
    training_mode: bool,
    num_workers: int,
) -> DataLoader:
    dataset = CTRDataset(
        csv_path=csv_path,
        stats=stats,
        user_features=user_features,
        episode_features=episode_features,
        alpha=alpha,
        bucket_size=bucket_size,
        max_rows=max_rows,
        log_every=log_every,
        has_label=has_label,
        training_mode=training_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class DeepFMModel(nn.Module):
    def __init__(
        self,
        dense_dim: int,
        sparse_fields: int,
        bucket_size: int,
        embedding_dim: int,
        deep_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.sparse_embedding = nn.Embedding(bucket_size, embedding_dim)
        self.linear_embedding = nn.Embedding(bucket_size, 1)
        self.dense_linear = nn.Linear(dense_dim, 1)

        deep_input_dim = dense_dim + sparse_fields * embedding_dim
        self.deep_tower = build_mlp(deep_input_dim, deep_dims, dropout)
        self.deep_output = nn.Linear(deep_dims[-1], 1)

        nn.init.xavier_uniform_(self.sparse_embedding.weight)
        nn.init.zeros_(self.linear_embedding.weight)
        nn.init.xavier_uniform_(self.dense_linear.weight)
        nn.init.zeros_(self.dense_linear.bias)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        sparse_emb = self.sparse_embedding(sparse)
        linear_part = self.linear_embedding(sparse).sum(dim=1).squeeze(1)
        linear_part = linear_part + self.dense_linear(dense).squeeze(1)

        summed = sparse_emb.sum(dim=1)
        sum_square = summed * summed
        square_sum = (sparse_emb * sparse_emb).sum(dim=1)
        fm_part = 0.5 * (sum_square - square_sum).sum(dim=1)

        deep_input = torch.cat([dense, sparse_emb.reshape(sparse_emb.size(0), -1)], dim=1)
        deep_hidden = self.deep_tower(deep_input)
        deep_part = self.deep_output(deep_hidden).squeeze(1)

        return linear_part + fm_part + deep_part


class ApproxAUC:
    def __init__(self, bins: int = 4096) -> None:
        self.bins = bins
        self.positive_bins = [0 for _ in range(bins)]
        self.negative_bins = [0 for _ in range(bins)]

    def add(self, probability: float, label: int) -> None:
        idx = min(self.bins - 1, max(0, int(probability * (self.bins - 1))))
        if label == 1:
            self.positive_bins[idx] += 1
        else:
            self.negative_bins[idx] += 1

    def score(self) -> float:
        total_positive = sum(self.positive_bins)
        total_negative = sum(self.negative_bins)
        if total_positive == 0 or total_negative == 0:
            return 0.5
        negative_seen = 0
        auc_numerator = 0.0
        for idx in range(self.bins):
            positives = self.positive_bins[idx]
            negatives = self.negative_bins[idx]
            auc_numerator += positives * negative_seen
            auc_numerator += 0.5 * positives * negatives
            negative_seen += negatives
        return auc_numerator / (total_positive * total_negative)


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_rows = 0
    total_loss = 0.0
    auc = ApproxAUC()
    for batch in loader:
        dense = batch["dense"].to(device)
        sparse = batch["sparse"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(dense, sparse)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        label_values = labels.detach().cpu().numpy()
        batch_size = len(probabilities)
        total_rows += batch_size
        total_loss += loss.item() * batch_size
        for probability, label in zip(probabilities, label_values):
            auc.add(float(probability), int(label))
    return {
        "rows": total_rows,
        "logloss": total_loss / total_rows if total_rows else None,
        "approx_auc": auc.score(),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_rows = 0
    total_loss = 0.0
    auc = ApproxAUC()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    for batch in loader:
        dense = batch["dense"].to(device)
        sparse = batch["sparse"].to(device)
        labels = batch["label"].to(device)
        logits = model(dense, sparse)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        label_values = labels.cpu().numpy()
        batch_size = len(probabilities)
        total_rows += batch_size
        total_loss += float(loss.item())
        for probability, label in zip(probabilities, label_values):
            auc.add(float(probability), int(label))
    return {
        "rows": total_rows,
        "logloss": total_loss / total_rows if total_rows else None,
        "approx_auc": auc.score(),
    }


@torch.no_grad()
def write_predictions(model, loader, device, output_csv: Path) -> dict:
    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for batch in loader:
            dense = batch["dense"].to(device)
            sparse = batch["sparse"].to(device)
            logits = model(dense, sparse)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for row_id, probability in zip(batch["id"], probabilities):
                writer.writerow([row_id, f"{float(probability):.8f}"])
                total_rows += 1
    return {"rows": total_rows, "output_csv": str(output_csv)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    log("Training configuration:")
    log(
        json.dumps(
            {
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "output_dir": str(args.output_dir),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "alpha": args.alpha,
                "embedding_dim": args.embedding_dim,
                "deep_dims": list(args.deep_dims),
                "hash_bucket_size": args.hash_bucket_size,
                "dropout": args.dropout,
                "device": str(device),
                "predict_test": args.predict_test,
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    user_features = load_user_features(args.user_feature_csv, args.log_every)
    episode_features = load_episode_features(args.episode_feature_csv, args.log_every)
    stats = build_aggregate_stats(
        train_csv=args.train_csv,
        user_features=user_features,
        episode_features=episode_features,
        max_rows=args.max_train_rows,
        log_every=args.log_every,
    )

    train_loader = make_loader(
        csv_path=args.train_csv,
        stats=stats,
        user_features=user_features,
        episode_features=episode_features,
        alpha=args.alpha,
        bucket_size=args.hash_bucket_size,
        max_rows=args.max_train_rows,
        log_every=args.log_every,
        batch_size=args.batch_size,
        has_label=True,
        training_mode=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        csv_path=args.val_csv,
        stats=stats,
        user_features=user_features,
        episode_features=episode_features,
        alpha=args.alpha,
        bucket_size=args.hash_bucket_size,
        max_rows=args.max_val_rows,
        log_every=args.log_every,
        batch_size=args.batch_size,
        has_label=True,
        training_mode=False,
        num_workers=args.num_workers,
    )

    sparse_field_count = len(BASE_CATEGORICAL_FIELDS) + len(CROSS_FIELD_SPECS)
    model = DeepFMModel(
        dense_dim=len(NUMERIC_FEATURE_NAMES),
        sparse_fields=sparse_field_count,
        bucket_size=args.hash_bucket_size,
        embedding_dim=args.embedding_dim,
        deep_dims=tuple(args.deep_dims),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        log(f"Starting epoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        log(
            f"Epoch {epoch} finished: "
            f"train_logloss={train_metrics['logloss']:.6f}, "
            f"train_auc={train_metrics['approx_auc']:.6f}, "
            f"val_logloss={val_metrics['logloss']:.6f}, "
            f"val_auc={val_metrics['approx_auc']:.6f}"
        )
        if val_metrics["approx_auc"] > best_val_auc:
            best_val_auc = val_metrics["approx_auc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = args.output_dir / "model.pt"
    metrics_path = args.output_dir / "metrics.json"
    torch.save(model.state_dict(), model_path)
    dump_json(
        metrics_path,
        {
            "config": {
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "alpha": args.alpha,
                "embedding_dim": args.embedding_dim,
                "deep_dims": list(args.deep_dims),
                "hash_bucket_size": args.hash_bucket_size,
                "dropout": args.dropout,
                "device": str(device),
            },
            "aggregate_stats": {
                "global_rows": stats.global_rows,
                "global_positive": stats.global_positive,
                "global_rate": stats.global_rate,
            },
            "history": history,
            "feature_names": list(NUMERIC_FEATURE_NAMES),
            "base_categorical_fields": list(BASE_CATEGORICAL_FIELDS),
            "cross_field_specs": [list(spec) for spec in CROSS_FIELD_SPECS],
        },
    )

    if args.predict_test:
        test_loader = make_loader(
            csv_path=args.test_csv,
            stats=stats,
            user_features=user_features,
            episode_features=episode_features,
            alpha=args.alpha,
            bucket_size=args.hash_bucket_size,
            max_rows=args.max_test_rows,
            log_every=args.log_every,
            batch_size=args.batch_size,
            has_label=False,
            training_mode=False,
            num_workers=args.num_workers,
        )
        write_predictions(model, test_loader, device, args.output_dir / "result.csv")

    log(f"Artifacts written to {args.output_dir}")
    log(f"  - model: {model_path}")
    log(f"  - metrics: {metrics_path}")


if __name__ == "__main__":
    main()

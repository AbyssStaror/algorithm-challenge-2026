# To generate predictions for the test set, 
# run this script with a trained DeepFM checkpoint and the appropriate csv files.
# The output will be a result.csv file with id and label columns.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from train_dcn import (
    BASE_CATEGORICAL_FIELDS,
    CROSS_FIELD_SPECS,
    NUMERIC_FEATURE_NAMES,
    build_aggregate_stats,
    load_episode_features,
    load_user_features,
    log,
    make_loader,
    resolve_device,
)
from train_deepfm import DeepFMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate result.csv predictions for the test set using a trained DeepFM checkpoint."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/deepfm_full_bs2048/model.pt"),
        help="Path to the trained DeepFM checkpoint.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/splits/stratified_train.csv"),
        help="Training split used to build aggregate stats.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/test.csv"),
        help="Test csv to score.",
    )
    parser.add_argument(
        "--user-feature-csv",
        type=Path,
        default=Path("data/user_feature.csv"),
        help="User feature table.",
    )
    parser.add_argument(
        "--episode-feature-csv",
        type=Path,
        default=Path("data/episode_feature.csv"),
        help="Episode feature table.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("result.csv"),
        help="Output csv path.",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="Prediction batch size.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
        help="Smoothing strength used by rate features.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for sparse fields.",
    )
    parser.add_argument(
        "--deep-dims",
        type=int,
        nargs="+",
        default=(128, 64),
        help="Hidden layer sizes for the deep tower.",
    )
    parser.add_argument(
        "--hash-bucket-size",
        type=int,
        default=800_000,
        help="Shared hash bucket size for sparse fields and crosses.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout value used at train time.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:0 style string.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Keep 0 on Windows unless needed.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200_000,
        help="Progress logging interval while scanning csv files.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap when building aggregate stats.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


@torch.no_grad()
def write_result_csv(
    model: DeepFMModel,
    loader,
    device: torch.device,
    output_csv: Path,
) -> None:
    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    log("Prediction configuration:")
    log(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "train_csv": str(args.train_csv),
                "test_csv": str(args.test_csv),
                "output_csv": str(args.output_csv),
                "batch_size": args.batch_size,
                "device": str(device),
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

    sparse_field_count = len(BASE_CATEGORICAL_FIELDS) + len(CROSS_FIELD_SPECS)
    model = DeepFMModel(
        dense_dim=len(NUMERIC_FEATURE_NAMES),
        sparse_fields=sparse_field_count,
        bucket_size=args.hash_bucket_size,
        embedding_dim=args.embedding_dim,
        deep_dims=tuple(args.deep_dims),
        dropout=args.dropout,
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    write_result_csv(
        model=model,
        loader=test_loader,
        device=device,
        output_csv=args.output_csv,
    )
    log(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

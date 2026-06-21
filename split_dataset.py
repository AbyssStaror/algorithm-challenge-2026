from __future__ import annotations

import argparse
import csv
import json
import random
from array import array
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create local validation splits for the algorithm-challenge-2026 train set. "
            "The script supports both a row-level stratified split and a stricter "
            "group-by-user split."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to the training csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory used to store split manifests and summaries.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("stratified", "group_by_uid"),
        default=("stratified", "group_by_uid"),
        help="Split methods to run.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio. Usually 0.1 or 0.2 works well for this task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used by the split builders.",
    )
    parser.add_argument(
        "--uid-col",
        default="uid",
        help="User id column name in the training file.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Binary label column name in the training file.",
    )
    parser.add_argument(
        "--write-subsets",
        action="store_true",
        help="Also materialize train/val csv files for each split method.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick smoke tests.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1_000_000,
        help="Progress logging interval while scanning the csv.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    return args


def log(message: str) -> None:
    print(message, flush=True)


def should_stop(row_idx: int, max_rows: int | None) -> bool:
    return max_rows is not None and row_idx >= max_rows


def maybe_log_progress(row_idx: int, every: int, stage: str) -> None:
    if row_idx > 0 and row_idx % every == 0:
        log(f"[{stage}] processed {row_idx:,} rows")


def shuffle_array(values: array, rng: random.Random) -> None:
    for idx in range(len(values) - 1, 0, -1):
        swap_idx = rng.randint(0, idx)
        values[idx], values[swap_idx] = values[swap_idx], values[idx]


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def positive_int(label_value: str) -> int:
    return 1 if str(label_value).strip() == "1" else 0


def dump_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)


def manifest_paths(output_dir: Path, method: str, write_subsets: bool) -> dict[str, Path]:
    paths = {
        "manifest": output_dir / f"{method}_manifest.csv",
        "summary": output_dir / f"{method}_summary.json",
    }
    if write_subsets:
        paths["train_subset"] = output_dir / f"{method}_train.csv"
        paths["val_subset"] = output_dir / f"{method}_val.csv"
    return paths


def write_row_assignments(
    train_path: Path,
    output_dir: Path,
    method: str,
    max_rows: int | None,
    log_every: int,
    split_selector,
    write_subsets: bool,
    extra_summary: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = manifest_paths(output_dir, method, write_subsets)

    split_counts = defaultdict(int)
    split_positive_counts = defaultdict(int)

    with train_path.open("r", newline="", encoding="utf-8") as source_handle, paths["manifest"].open(
        "w", newline="", encoding="utf-8"
    ) as manifest_handle:
        reader = csv.DictReader(source_handle)
        manifest_writer = csv.writer(manifest_handle)
        manifest_writer.writerow(["row_idx", "split"])

        train_subset_handle = None
        val_subset_handle = None
        train_subset_writer = None
        val_subset_writer = None
        try:
            if write_subsets:
                train_subset_handle = paths["train_subset"].open("w", newline="", encoding="utf-8")
                val_subset_handle = paths["val_subset"].open("w", newline="", encoding="utf-8")
                train_subset_writer = csv.DictWriter(train_subset_handle, fieldnames=reader.fieldnames)
                val_subset_writer = csv.DictWriter(val_subset_handle, fieldnames=reader.fieldnames)
                train_subset_writer.writeheader()
                val_subset_writer.writeheader()

            for row_idx, row in enumerate(reader):
                if should_stop(row_idx, max_rows):
                    break

                split_name = split_selector(row_idx, row)
                manifest_writer.writerow([row_idx, split_name])
                split_counts[split_name] += 1
                split_positive_counts[split_name] += positive_int(row["label"])

                if write_subsets:
                    if split_name == "train":
                        train_subset_writer.writerow(row)
                    elif split_name == "val":
                        val_subset_writer.writerow(row)
                    else:
                        raise ValueError(f"Unexpected split name: {split_name}")

                maybe_log_progress(row_idx + 1, log_every, f"{method}:write")
        finally:
            if train_subset_handle is not None:
                train_subset_handle.close()
            if val_subset_handle is not None:
                val_subset_handle.close()

    total_rows = sum(split_counts.values())
    total_positive = sum(split_positive_counts.values())
    summary = {
        "method": method,
        "train_path": str(train_path),
        "manifest_path": str(paths["manifest"]),
        "seed_note": "See command line seed for deterministic reconstruction.",
        "max_rows": max_rows,
        "counts": {
            "total_rows": total_rows,
            "train_rows": split_counts["train"],
            "val_rows": split_counts["val"],
            "total_positive": total_positive,
            "train_positive": split_positive_counts["train"],
            "val_positive": split_positive_counts["val"],
            "train_positive_rate": safe_rate(split_positive_counts["train"], split_counts["train"]),
            "val_positive_rate": safe_rate(split_positive_counts["val"], split_counts["val"]),
        },
    }
    if write_subsets:
        summary["train_subset_path"] = str(paths["train_subset"])
        summary["val_subset_path"] = str(paths["val_subset"])
    if extra_summary:
        summary.update(extra_summary)

    dump_summary(paths["summary"], summary)
    return summary


def build_stratified_split(
    train_path: Path,
    output_dir: Path,
    label_col: str,
    val_ratio: float,
    seed: int,
    max_rows: int | None,
    log_every: int,
    write_subsets: bool,
) -> dict:
    rng = random.Random(seed)
    label_to_indices: dict[str, array] = defaultdict(lambda: array("I"))

    log("[stratified] collecting row indices by label")
    with train_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if label_col not in reader.fieldnames:
            raise KeyError(f"Column '{label_col}' does not exist in {train_path}")

        total_rows = 0
        for row in reader:
            if should_stop(total_rows, max_rows):
                break
            label_to_indices[row[label_col]].append(total_rows)
            total_rows += 1
            maybe_log_progress(total_rows, log_every, "stratified:scan")

    val_mask = bytearray(total_rows)
    label_counts = {label: len(indices) for label, indices in label_to_indices.items()}
    val_counts = {}
    for label, indices in label_to_indices.items():
        shuffle_array(indices, rng)
        val_count = int(round(len(indices) * val_ratio))
        val_counts[label] = val_count
        for row_idx in indices[:val_count]:
            val_mask[row_idx] = 1

    log("[stratified] writing manifest and optional subset files")
    summary = write_row_assignments(
        train_path=train_path,
        output_dir=output_dir,
        method="stratified",
        max_rows=max_rows,
        log_every=log_every,
        split_selector=lambda row_idx, row: "val" if val_mask[row_idx] else "train",
        write_subsets=write_subsets,
        extra_summary={
            "config": {
                "val_ratio": val_ratio,
                "seed": seed,
            },
            "label_counts": label_counts,
            "label_val_counts": val_counts,
        },
    )
    return summary


def score_group_assignment(
    val_rows: int,
    val_positive: int,
    target_rows: float,
    target_positive: float,
) -> float:
    row_gap = abs(val_rows - target_rows) / max(target_rows, 1.0)
    positive_gap = abs(val_positive - target_positive) / max(target_positive, 1.0)
    return row_gap + positive_gap


def choose_validation_users(
    uid_stats: dict[str, tuple[int, int]],
    total_rows: int,
    total_positive: int,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], dict]:
    rng = random.Random(seed)
    items = list(uid_stats.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1][0], reverse=True)

    target_rows = total_rows * val_ratio
    target_positive = total_positive * val_ratio
    remaining_rows = total_rows
    remaining_positive = total_positive
    val_rows = 0
    val_positive = 0
    val_users: set[str] = set()

    for uid, (user_rows, user_positive) in items:
        remaining_rows -= user_rows
        remaining_positive -= user_positive

        current_score = score_group_assignment(val_rows, val_positive, target_rows, target_positive)
        candidate_score = score_group_assignment(
            val_rows + user_rows,
            val_positive + user_positive,
            target_rows,
            target_positive,
        )

        rows_needed = max(0.0, target_rows - val_rows)
        positives_needed = max(0.0, target_positive - val_positive)
        must_take_for_rows = remaining_rows < rows_needed
        must_take_for_positives = remaining_positive < positives_needed

        if must_take_for_rows or must_take_for_positives or candidate_score <= current_score:
            val_users.add(uid)
            val_rows += user_rows
            val_positive += user_positive

    group_summary = {
        "target_rows": target_rows,
        "target_positive": target_positive,
        "selected_val_users": len(val_users),
        "selected_val_rows": val_rows,
        "selected_val_positive": val_positive,
    }
    return val_users, group_summary


def build_group_by_uid_split(
    train_path: Path,
    output_dir: Path,
    uid_col: str,
    label_col: str,
    val_ratio: float,
    seed: int,
    max_rows: int | None,
    log_every: int,
    write_subsets: bool,
) -> dict:
    uid_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    log("[group_by_uid] collecting per-user row counts")
    with train_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [name for name in (uid_col, label_col) if name not in reader.fieldnames]
        if missing_columns:
            raise KeyError(f"Missing required columns in {train_path}: {missing_columns}")

        total_rows = 0
        total_positive = 0
        for row in reader:
            if should_stop(total_rows, max_rows):
                break
            uid = row[uid_col]
            positive = positive_int(row[label_col])
            uid_stats[uid][0] += 1
            uid_stats[uid][1] += positive
            total_rows += 1
            total_positive += positive
            maybe_log_progress(total_rows, log_every, "group_by_uid:scan")

    immutable_uid_stats = {uid: (values[0], values[1]) for uid, values in uid_stats.items()}
    val_users, group_summary = choose_validation_users(
        uid_stats=immutable_uid_stats,
        total_rows=total_rows,
        total_positive=total_positive,
        val_ratio=val_ratio,
        seed=seed,
    )

    log("[group_by_uid] writing manifest and optional subset files")
    summary = write_row_assignments(
        train_path=train_path,
        output_dir=output_dir,
        method="group_by_uid",
        max_rows=max_rows,
        log_every=log_every,
        split_selector=lambda row_idx, row: "val" if row[uid_col] in val_users else "train",
        write_subsets=write_subsets,
        extra_summary={
            "config": {
                "val_ratio": val_ratio,
                "seed": seed,
            },
            "group_stats": {
                "total_users": len(uid_stats),
                **group_summary,
            },
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Split configuration:")
    log(
        json.dumps(
            {
                "train_path": str(args.train_path),
                "output_dir": str(args.output_dir),
                "methods": list(args.methods),
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "write_subsets": args.write_subsets,
                "max_rows": args.max_rows,
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    summaries = {}
    for method in args.methods:
        if method == "stratified":
            summaries[method] = build_stratified_split(
                train_path=args.train_path,
                output_dir=output_dir,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                seed=args.seed,
                max_rows=args.max_rows,
                log_every=args.log_every,
                write_subsets=args.write_subsets,
            )
        elif method == "group_by_uid":
            summaries[method] = build_group_by_uid_split(
                train_path=args.train_path,
                output_dir=output_dir,
                uid_col=args.uid_col,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                seed=args.seed,
                max_rows=args.max_rows,
                log_every=args.log_every,
                write_subsets=args.write_subsets,
            )
        else:
            raise ValueError(f"Unsupported split method: {method}")

    log("Finished building splits. Summary files:")
    for method in args.methods:
        summary_path = output_dir / f"{method}_summary.json"
        log(f"  - {method}: {summary_path}")


if __name__ == "__main__":
    main()

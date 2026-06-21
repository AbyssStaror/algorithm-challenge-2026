from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


MISSING_TOKEN = "__MISSING__"
BASE_STAT_FIELDS = (
    "uid",
    "episode_id",
    "tab_name",
    "scene_name",
    "entrance_type",
    "address",
    "sex",
    "rg_source",
    "language",
    "primary_category",
)
BASE_CATEGORICAL_FIELDS = (
    "uid",
    "episode_id",
    "tab_name",
    "scene_name",
    "entrance_type",
    "address",
    "sex",
    "rg_source",
    "language",
    "primary_category",
)
NUMERIC_FEATURE_NAMES = (
    "uid_rate",
    "episode_rate",
    "tab_rate",
    "scene_rate",
    "entrance_rate",
    "address_rate",
    "sex_rate",
    "rg_source_rate",
    "language_rate",
    "primary_category_rate",
    "uid_count",
    "episode_count",
    "age",
    "membership_days",
    "duration_minutes",
    "category_count",
    "has_user_profile",
    "has_episode_profile",
    "has_scene_name",
    "uid_x_tab_rate",
    "uid_x_scene_rate",
    "uid_x_entrance_rate",
    "episode_x_tab_rate",
    "category_x_entrance_rate",
    "language_x_tab_rate",
)
CROSS_FIELD_SPECS = (
    ("uid", "tab_name"),
    ("uid", "scene_name"),
    ("uid", "entrance_type"),
    ("episode_id", "tab_name"),
    ("primary_category", "entrance_type"),
    ("language", "tab_name"),
)


def log(message: str) -> None:
    print(message, flush=True)


def clean_text(value: str | None) -> str:
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip()
    return text if text else MISSING_TOKEN


def positive_int(label_value: str) -> int:
    return 1 if str(label_value).strip() == "1" else 0


def safe_int(value: str | None, default: int = 0) -> int:
    text = clean_text(value)
    if text == MISSING_TOKEN:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def safe_float(value: str | None, default: float = 0.0) -> float:
    text = clean_text(value)
    if text == MISSING_TOKEN:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_date(text: str | None) -> datetime | None:
    value = clean_text(text)
    if value == MISSING_TOKEN:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def safe_ratio(positive: int, total: int) -> float:
    return positive / total if total else 0.0


def smoothed_rate(count: int, positive: int, prior: float, alpha: float) -> float:
    return (positive + alpha * prior) / (count + alpha)


def loo_smoothed_rate(count: int, positive: int, label: int, prior: float, alpha: float) -> float:
    adjusted_count = max(count - 1, 0)
    adjusted_positive = max(positive - label, 0)
    return (adjusted_positive + alpha * prior) / (adjusted_count + alpha)


def should_stop(row_idx: int, max_rows: int | None) -> bool:
    return max_rows is not None and row_idx >= max_rows


def maybe_log_progress(row_idx: int, every: int, stage: str) -> None:
    if row_idx > 0 and row_idx % every == 0:
        log(f"[{stage}] processed {row_idx:,} rows")


def scale_log_count(count: int) -> float:
    return min(math.log1p(max(count, 0)) / 10.0, 1.0)


def scale_age(age: int) -> float:
    return min(max(age, 0) / 100.0, 1.0)


def scale_membership_days(days: int) -> float:
    return min(max(days, 0) / 4000.0, 1.0)


def scale_duration_minutes(minutes: float) -> float:
    return min(math.log1p(max(minutes, 0.0)) / 8.0, 1.0)


def scale_category_count(category_count: int) -> float:
    return min(max(category_count, 0) / 10.0, 1.0)


def stable_hash(text: str, bucket_size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % bucket_size


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_user_features(path: Path, log_every: int) -> dict[str, dict]:
    log(f"Loading user features from {path}")
    features: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            rg_date = parse_date(row.get("rg_date"))
            exp_date = parse_date(row.get("exp_date"))
            membership_days = 0
            if rg_date and exp_date:
                membership_days = max((exp_date - rg_date).days, 0)

            uid = clean_text(row.get("uid"))
            features[uid] = {
                "address": clean_text(row.get("address")),
                "age": safe_int(row.get("age"), default=0),
                "sex": clean_text(row.get("sex")),
                "rg_source": clean_text(row.get("rg_source")),
                "membership_days": membership_days,
            }
            maybe_log_progress(row_idx, log_every, "user_features")
    return features


def load_episode_features(path: Path, log_every: int) -> dict[str, dict]:
    log(f"Loading episode features from {path}")
    features: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            category_ids = clean_text(row.get("category_ids"))
            categories = [token for token in category_ids.split("|") if token and token != MISSING_TOKEN]
            primary_category = categories[0] if categories else MISSING_TOKEN
            duration_minutes = safe_float(row.get("duration"), default=0.0) / 60000.0
            episode_id = clean_text(row.get("episode_id"))
            features[episode_id] = {
                "primary_category": primary_category,
                "category_count": len(categories),
                "duration_minutes": duration_minutes,
                "language": clean_text(row.get("language")),
            }
            maybe_log_progress(row_idx, log_every, "episode_features")
    return features


def build_context(
    row: dict[str, str],
    user_features: dict[str, dict],
    episode_features: dict[str, dict],
) -> dict[str, str | float | int]:
    uid = clean_text(row.get("uid"))
    episode_id = clean_text(row.get("episode_id"))
    user_info = user_features.get(uid)
    episode_info = episode_features.get(episode_id)

    return {
        "uid": uid,
        "episode_id": episode_id,
        "tab_name": clean_text(row.get("tab_name")),
        "scene_name": clean_text(row.get("scene_name")),
        "entrance_type": clean_text(row.get("entrance_type")),
        "address": clean_text(user_info["address"] if user_info else None),
        "sex": clean_text(user_info["sex"] if user_info else None),
        "rg_source": clean_text(user_info["rg_source"] if user_info else None),
        "language": clean_text(episode_info["language"] if episode_info else None),
        "primary_category": clean_text(episode_info["primary_category"] if episode_info else None),
        "age": int(user_info["age"]) if user_info else 0,
        "membership_days": int(user_info["membership_days"]) if user_info else 0,
        "duration_minutes": float(episode_info["duration_minutes"]) if episode_info else 0.0,
        "category_count": int(episode_info["category_count"]) if episode_info else 0,
        "has_user_profile": 1.0 if user_info else 0.0,
        "has_episode_profile": 1.0 if episode_info else 0.0,
        "has_scene_name": 0.0 if clean_text(row.get("scene_name")) == MISSING_TOKEN else 1.0,
    }


@dataclass
class AggregateStats:
    global_rows: int
    global_positive: int
    field_counts: dict[str, dict[str, list[int]]]

    @property
    def global_rate(self) -> float:
        return safe_ratio(self.global_positive, self.global_rows)


def build_aggregate_stats(
    train_csv: Path,
    user_features: dict[str, dict],
    episode_features: dict[str, dict],
    max_rows: int | None,
    log_every: int,
) -> AggregateStats:
    log(f"Building aggregate stats from {train_csv}")
    stat_fields = list(BASE_STAT_FIELDS) + [f"{left}__X__{right}" for left, right in CROSS_FIELD_SPECS]
    field_counts = {field: defaultdict(lambda: [0, 0]) for field in stat_fields}
    global_rows = 0
    global_positive = 0

    with train_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if should_stop(row_idx, max_rows):
                break
            label = positive_int(row["label"])
            context = build_context(row, user_features, episode_features)
            for field in BASE_STAT_FIELDS:
                field_counts[field][str(context[field])][0] += 1
                field_counts[field][str(context[field])][1] += label
            for left, right in CROSS_FIELD_SPECS:
                cross_name = f"{left}__X__{right}"
                cross_value = f"{context[left]}||{context[right]}"
                field_counts[cross_name][cross_value][0] += 1
                field_counts[cross_name][cross_value][1] += label
            global_rows += 1
            global_positive += label
            maybe_log_progress(global_rows, log_every, "aggregate_stats")

    return AggregateStats(
        global_rows=global_rows,
        global_positive=global_positive,
        field_counts={field: dict(values) for field, values in field_counts.items()},
    )


def build_numeric_features(
    context: dict[str, str | float | int],
    stats: AggregateStats,
    alpha: float,
    label: int | None = None,
    is_training_row: bool = False,
) -> np.ndarray:
    values: list[float] = []
    global_rate = stats.global_rate

    for field in BASE_STAT_FIELDS:
        key = str(context[field])
        count, positive = stats.field_counts[field].get(key, (0, 0))
        if is_training_row and label is not None:
            rate = loo_smoothed_rate(count, positive, label, global_rate, alpha)
        else:
            rate = smoothed_rate(count, positive, global_rate, alpha)
        values.append(rate)

    uid_count, _ = stats.field_counts["uid"].get(str(context["uid"]), (0, 0))
    episode_count, _ = stats.field_counts["episode_id"].get(str(context["episode_id"]), (0, 0))
    if is_training_row:
        uid_count = max(uid_count - 1, 0)
        episode_count = max(episode_count - 1, 0)

    values.extend(
        [
            scale_log_count(uid_count),
            scale_log_count(episode_count),
            scale_age(int(context["age"])),
            scale_membership_days(int(context["membership_days"])),
            scale_duration_minutes(float(context["duration_minutes"])),
            scale_category_count(int(context["category_count"])),
            float(context["has_user_profile"]),
            float(context["has_episode_profile"]),
            float(context["has_scene_name"]),
        ]
    )

    for left, right in CROSS_FIELD_SPECS:
        cross_name = f"{left}__X__{right}"
        cross_value = f"{context[left]}||{context[right]}"
        count, positive = stats.field_counts[cross_name].get(cross_value, (0, 0))
        if is_training_row and label is not None:
            rate = loo_smoothed_rate(count, positive, label, global_rate, alpha)
        else:
            rate = smoothed_rate(count, positive, global_rate, alpha)
        values.append(rate)

    return np.asarray(values, dtype=np.float32)


def build_sparse_indices(
    context: dict[str, str | float | int],
    bucket_size: int,
) -> np.ndarray:
    sparse_indices = []
    for field in BASE_CATEGORICAL_FIELDS:
        token = f"{field}={context[field]}"
        sparse_indices.append(stable_hash(token, bucket_size))

    for left, right in CROSS_FIELD_SPECS:
        token = f"{left}__X__{right}={context[left]}||{context[right]}"
        sparse_indices.append(stable_hash(token, bucket_size))

    return np.asarray(sparse_indices, dtype=np.int64)

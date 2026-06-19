#!/usr/bin/env python3
"""Create leakage-safe CFPB train/validation/test splits.

The zero-shot subset files are treated as immutable legacy test sets. Any row
in complaints.csv that matches a legacy example exactly, or whose narrative
starts with a 250-character legacy narrative, is excluded before splitting.

Rows sharing the same normalized narrative are assigned to the same split,
including rows whose labels disagree. This prevents duplicate text from
crossing train, validation, and internal-test boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "credit_reporting",
    "debt_collection",
    "mortgages_and_loans",
    "credit_card",
    "retail_banking",
]
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "internal_test": 0.10}
SUBSET_RE = re.compile(
    r"(?P<seed>[123])_subset_(?P<condition>balanced|imbalanced)_"
    r"(?P<max_chars>250|500)\.csv$"
)


def normalize_text(value: object) -> str:
    """Normalize text for duplicate detection without discarding punctuation."""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return " ".join(text.split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory containing complaints.csv and the legacy_subsets directory.",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing legacy subset CSV files. Defaults to "
            "<dataset-dir>/legacy_subsets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "splits",
        help="Directory in which split files and reports are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_legacy_subsets(legacy_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = sorted(legacy_dir.glob("[123]_subset_*.csv"))
    if len(files) != 12:
        raise ValueError(f"Expected 12 legacy subset files, found {len(files)}")

    for path in files:
        match = SUBSET_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected subset filename: {path.name}")
        frame = pd.read_csv(path)
        required = {"true_label", "narrative"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} is missing columns: {required - set(frame.columns)}")
        frame = frame[["true_label", "narrative"]].copy()
        frame["source_file"] = path.name
        frame["source_row"] = range(len(frame))
        frame["seed"] = int(match.group("seed"))
        frame["condition"] = match.group("condition")
        frame["max_chars"] = int(match.group("max_chars"))
        frames.append(frame)

    legacy = pd.concat(frames, ignore_index=True)
    if legacy[["true_label", "narrative"]].isna().any().any():
        raise ValueError("Legacy subsets contain missing labels or narratives")
    unknown = sorted(set(legacy["true_label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Legacy subsets contain unknown labels: {unknown}")

    legacy["narrative"] = legacy["narrative"].astype(str).str.strip()
    legacy["narrative_norm"] = legacy["narrative"].map(normalize_text)
    legacy["narrative_hash"] = legacy["narrative_norm"].map(digest)
    return legacy


def identify_legacy_rows(main: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return exclusion mask and match reason for rows related to legacy tests."""
    excluded = pd.Series(False, index=main.index)
    reason = pd.Series("", index=main.index, dtype="object")

    for label in LABELS:
        legacy_label = legacy.loc[legacy["true_label"].eq(label)]
        main_idx = main.index[main["product"].eq(label)]
        if not len(main_idx):
            continue

        exact_norm = set(legacy_label["narrative_norm"])
        prefix_raw = set(
            legacy_label.loc[
                legacy_label["max_chars"].eq(250)
                & legacy_label["narrative"].str.len().eq(250),
                "narrative",
            ]
        )
        prefix_norm = {normalize_text(text) for text in prefix_raw}

        main_raw = main.loc[main_idx, "narrative"].astype(str).str.strip()
        main_norm = main.loc[main_idx, "narrative_norm"]
        exact_mask = main_norm.isin(exact_norm)
        prefix_mask = main_raw.str[:250].isin(prefix_raw) | main_norm.str[:250].isin(prefix_norm)

        excluded.loc[main_idx] = (exact_mask | prefix_mask).to_numpy()
        label_reason = pd.Series("", index=main_idx, dtype="object")
        label_reason.loc[prefix_mask.index[prefix_mask]] = "legacy_250_prefix"
        label_reason.loc[exact_mask.index[exact_mask]] = "legacy_exact"
        reason.loc[main_idx] = label_reason

    # Apply the truncated-text guard independently of the label as well. A
    # complaint beginning with a legacy 250-character narrative must not enter
    # training merely because its stored label differs.
    all_prefix_raw = set(
        legacy.loc[
            legacy["max_chars"].eq(250) & legacy["narrative"].str.len().eq(250),
            "narrative",
        ]
    )
    all_prefix_norm = {normalize_text(text) for text in all_prefix_raw}
    cross_label_prefix = (
        main["narrative"].astype(str).str.strip().str[:250].isin(all_prefix_raw)
        | main["narrative_norm"].str[:250].isin(all_prefix_norm)
    ) & ~excluded
    excluded.loc[cross_label_prefix] = True
    reason.loc[cross_label_prefix] = "legacy_250_prefix_cross_label"

    # If an excluded narrative also occurs under another label, exclude the
    # entire normalized-narrative group. Otherwise the legacy test text itself
    # would still be visible during training, despite the conflicting label.
    matched_hashes = set(main.loc[excluded, "narrative_hash"])
    group_expansion = main["narrative_hash"].isin(matched_hashes) & ~excluded
    excluded.loc[group_expansion] = True
    reason.loc[group_expansion] = "legacy_narrative_group"

    return excluded, reason


def assign_grouped_splits(remaining: pd.DataFrame, seed: int) -> dict[str, str]:
    """Assign normalized-narrative groups with approximately stratified ratios."""
    grouped = (
        remaining.groupby("narrative_hash", sort=False)["product"]
        .agg(list)
        .to_dict()
    )
    by_primary_label: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for group_id, labels in grouped.items():
        counts = Counter(labels)
        primary_label = min(
            (label for label, count in counts.items() if count == max(counts.values())),
            key=LABELS.index,
        )
        by_primary_label[primary_label].append((group_id, len(labels)))

    assignment: dict[str, str] = {}
    for label in LABELS:
        groups = by_primary_label[label]
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(groups)
        total_rows = sum(size for _, size in groups)
        train_limit = total_rows * SPLIT_RATIOS["train"]
        validation_limit = total_rows * (
            SPLIT_RATIOS["train"] + SPLIT_RATIOS["validation"]
        )
        cumulative = 0
        for group_id, size in groups:
            midpoint = cumulative + size / 2
            if midpoint <= train_limit:
                split = "train"
            elif midpoint <= validation_limit:
                split = "validation"
            else:
                split = "internal_test"
            assignment[group_id] = split
            cumulative += size

    if len(assignment) != len(grouped):
        raise RuntimeError("Not every narrative group received a split assignment")
    return assignment


def overlap_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    return len(set(left["narrative_hash"]) & set(right["narrative_hash"]))


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["product"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    legacy_dir = (
        args.legacy_dir.resolve()
        if args.legacy_dir is not None
        else dataset_dir / "legacy_subsets"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_path = dataset_dir / "complaints.csv"
    complaints = pd.read_csv(main_path)
    required = {"product", "narrative"}
    if not required.issubset(complaints.columns):
        raise ValueError(f"{main_path} is missing columns: {required - set(complaints.columns)}")

    raw_rows = len(complaints)
    missing_mask = complaints["product"].isna() | complaints["narrative"].isna()
    complaints = complaints.loc[~missing_mask].copy()
    complaints["product"] = complaints["product"].astype(str).str.strip()
    complaints["narrative"] = complaints["narrative"].astype(str).str.strip()
    unknown = sorted(set(complaints["product"]) - set(LABELS))
    if unknown:
        raise ValueError(f"complaints.csv contains unknown labels: {unknown}")

    if "Unnamed: 0" in complaints.columns:
        complaints = complaints.rename(columns={"Unnamed: 0": "record_id"})
    elif "record_id" not in complaints.columns:
        complaints.insert(0, "record_id", complaints.index)

    complaints["narrative_norm"] = complaints["narrative"].map(normalize_text)
    complaints["narrative_hash"] = complaints["narrative_norm"].map(digest)

    legacy = load_legacy_subsets(legacy_dir)
    legacy_mask, match_reason = identify_legacy_rows(complaints, legacy)
    complaints["legacy_match_reason"] = match_reason
    legacy_excluded = complaints.loc[legacy_mask].copy()
    remaining = complaints.loc[~legacy_mask].copy()

    assignment = assign_grouped_splits(remaining, args.seed)
    remaining["split"] = remaining["narrative_hash"].map(assignment)
    splits = {
        name: remaining.loc[remaining["split"].eq(name)].copy()
        for name in SPLIT_RATIOS
    }

    # Hard failures: these invariants define a leakage-safe split.
    if overlap_count(splits["train"], splits["validation"]) != 0:
        raise RuntimeError("Narrative overlap detected between train and validation")
    if overlap_count(splits["train"], splits["internal_test"]) != 0:
        raise RuntimeError("Narrative overlap detected between train and internal test")
    if overlap_count(splits["validation"], splits["internal_test"]) != 0:
        raise RuntimeError("Narrative overlap detected between validation and internal test")
    if any(set(frame["narrative_hash"]) & set(legacy_excluded["narrative_hash"]) for frame in splits.values()):
        raise RuntimeError("Exact normalized legacy overlap survived exclusion")

    output_columns = ["record_id", "product", "narrative", "narrative_hash"]
    for name, frame in splits.items():
        frame[output_columns].to_csv(output_dir / f"{name}.csv", index=False)

    legacy_columns = [
        "source_file",
        "source_row",
        "seed",
        "condition",
        "max_chars",
        "true_label",
        "narrative",
        "narrative_hash",
    ]
    legacy[legacy_columns].to_csv(output_dir / "legacy_test_manifest.csv", index=False)
    legacy_excluded[
        output_columns + ["legacy_match_reason"]
    ].to_csv(output_dir / "legacy_excluded_from_training.csv", index=False)

    mixed_label_groups = int(
        (remaining.groupby("narrative_hash")["product"].nunique() > 1).sum()
    )
    report = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "source": {
            "complaints_csv_rows": raw_rows,
            "dropped_missing_rows": int(missing_mask.sum()),
            "usable_rows": len(complaints),
            "unique_normalized_narratives": int(complaints["narrative_hash"].nunique()),
        },
        "legacy": {
            "manifest_rows": len(legacy),
            "manifest_unique_label_narrative": int(
                legacy[["true_label", "narrative_norm"]].drop_duplicates().shape[0]
            ),
            "manifest_unique_narratives": int(legacy["narrative_hash"].nunique()),
            "excluded_main_rows": len(legacy_excluded),
            "excluded_unique_narratives": int(
                legacy_excluded["narrative_hash"].nunique()
            ),
            "excluded_class_counts": class_counts(legacy_excluded),
            "match_reason_counts": {
                key: int(value)
                for key, value in legacy_excluded["legacy_match_reason"]
                .value_counts()
                .items()
            },
        },
        "remaining_pool": {
            "rows": len(remaining),
            "unique_normalized_narratives": int(remaining["narrative_hash"].nunique()),
            "mixed_label_narrative_groups": mixed_label_groups,
        },
        "splits": {
            name: {
                "rows": len(frame),
                "fraction_of_remaining": len(frame) / len(remaining),
                "unique_normalized_narratives": int(frame["narrative_hash"].nunique()),
                "class_counts": class_counts(frame),
            }
            for name, frame in splits.items()
        },
        "overlap_checks": {
            "train_validation_narratives": overlap_count(
                splits["train"], splits["validation"]
            ),
            "train_internal_test_narratives": overlap_count(
                splits["train"], splits["internal_test"]
            ),
            "validation_internal_test_narratives": overlap_count(
                splits["validation"], splits["internal_test"]
            ),
            "exact_legacy_hashes_in_remaining_pool": len(
                set(remaining["narrative_hash"])
                & set(legacy_excluded["narrative_hash"])
            ),
        },
    }
    with (output_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

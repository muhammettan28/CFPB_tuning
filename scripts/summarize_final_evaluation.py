#!/usr/bin/env python3
"""Summarize final internal-test and legacy benchmark evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--pattern", default="qwen3_8b_final_eval_*")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen3_8b_final_evaluation_summary.csv"),
    )
    return parser.parse_args()


def infer_variant(path: Path, metadata: dict) -> str:
    if metadata.get("base_only"):
        return "base_zero_shot"
    adapter = str(metadata.get("adapter") or "")
    for seed in ("42", "43", "44"):
        if (
            f"seed{seed}" in adapter
            or f"_seed{seed}_" in adapter
            or f"fine_tuned_seed{seed}" in path.as_posix()
        ):
            return f"fine_tuned_seed{seed}"
    for part in path.parts:
        if part.startswith("fine_tuned_seed"):
            return part
    return "fine_tuned"


def infer_dataset_role(data_file: str) -> str:
    if "internal_test.csv" in data_file:
        return "internal_test"
    if "legacy_subsets" in data_file:
        return "legacy"
    return "unknown"


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.outputs_dir.glob(f"{args.pattern}/**/metrics.json")):
        eval_dir = metrics_path.parent
        metadata_path = eval_dir / "evaluation_metadata.json"
        if not metadata_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_file = str(metadata["data_file"])
        report = metrics["classification_report"]
        row = {
            "run_dir": eval_dir.relative_to(args.outputs_dir).as_posix(),
            "variant": infer_variant(eval_dir, metadata),
            "dataset_role": infer_dataset_role(data_file),
            "dataset_name": Path(data_file).name,
            "data_file": data_file,
            "seed": metadata["seed"],
            "base_only": metadata["base_only"],
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "off_label_count": metrics["off_label_count"],
            "mean_latency_seconds": metrics["mean_latency_seconds"],
        }
        for label in LABELS:
            row[f"{label}_f1"] = report[label]["f1-score"]
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["dataset_role"],
            row["dataset_name"],
            row["variant"],
            int(row["seed"]),
        )
    )
    if not rows:
        raise SystemExit("No final evaluation metrics found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"{row['dataset_name']} {row['variant']} "
            f"acc={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"weighted_f1={row['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

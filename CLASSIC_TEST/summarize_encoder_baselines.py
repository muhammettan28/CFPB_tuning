#!/usr/bin/env python3
"""Summarise encoder baseline result JSON files.

This is intentionally separate from ``run_encoder_baseline.py`` so that SLURM
array jobs can run one encoder model at a time and the final table can be
rebuilt deterministically after all jobs finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "outputs/encoder_baselines_10k_len512",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <results-dir>/encoder_summary.csv.",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("results_*.json")):
        data = json.loads(path.read_text())
        internal = data["internal_test"]
        per_class = internal.get("per_class_f1", {})
        rows.append(
            {
                "model": data["model"],
                "model_name": data["model_name"],
                "max_train": data["max_train"],
                "max_chars": data["max_chars"],
                "max_length": data["max_length"],
                "batch_size": data.get("batch_size"),
                "grad_accum": data.get("grad_accum"),
                "effective_batch_size": data.get("effective_batch_size"),
                "epochs": data["epochs"],
                "lr": data["lr"],
                "seed": data["seed"],
                "internal_accuracy": internal["accuracy"],
                "internal_macro_f1": internal["macro_f1"],
                "internal_weighted_f1": internal["weighted_f1"],
                "legacy_macro_f1_mean": data["legacy_mean_macro_f1"],
                "legacy_weighted_f1_mean": data["legacy_mean_weighted_f1"],
                "debt_collection_f1": per_class.get("debt_collection"),
            }
        )

    if not rows:
        raise SystemExit(f"No encoder result JSON files found in {args.results_dir}")

    df = pd.DataFrame(rows).sort_values(
        ["internal_macro_f1", "legacy_macro_f1_mean"], ascending=False
    )
    output = args.output or args.results_dir / "encoder_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize Qwen3 hyperparameter-search validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument(
        "--pattern", default="qwen3_8b_search_*"
    )
    parser.add_argument(
        "--pilot-pattern",
        default="qwen3_8b_qlora_pilot_*",
        help="Completed natural-sampling pilot(s) to include as E0 baselines.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/qwen3_8b_search_summary.csv")
    )
    args = parser.parse_args()

    rows = []
    experiment_dirs = {
        *args.outputs_dir.glob(args.pattern),
        *args.outputs_dir.glob(args.pilot_pattern),
    }
    for experiment_dir in sorted(experiment_dirs):
        metrics_path = experiment_dir / "validation_inference/metrics.json"
        metadata_path = experiment_dir / "run_metadata.json"
        state_path = experiment_dir / "trainer_state.json"
        if not (metrics_path.exists() and metadata_path.exists() and state_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        arguments = metadata["arguments"]
        report = metrics["classification_report"]
        rows.append(
            {
                "experiment_dir": experiment_dir.name,
                "sampling": arguments.get("train_sampling", "natural"),
                "learning_rate": arguments["learning_rate"],
                "lora_r": arguments["lora_r"],
                "lora_alpha": arguments["lora_alpha"],
                "epochs": arguments["epochs"],
                "seed": arguments["seed"],
                "best_checkpoint": state.get("best_model_checkpoint"),
                "best_eval_loss": state.get("best_metric"),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "retail_banking_f1": report["retail_banking"]["f1-score"],
                "credit_reporting_f1": report["credit_reporting"]["f1-score"],
                "credit_card_f1": report["credit_card"]["f1-score"],
                "mortgages_and_loans_f1": report["mortgages_and_loans"]["f1-score"],
                "debt_collection_f1": report["debt_collection"]["f1-score"],
                "off_label_count": metrics["off_label_count"],
                "mean_latency_seconds": metrics["mean_latency_seconds"],
            }
        )

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No completed search experiments found")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['experiment_dir']} "
            f"macro_f1={row['macro_f1']:.4f} "
            f"weighted_f1={row['weighted_f1']:.4f} "
            f"debt_f1={row['debt_collection_f1']:.4f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

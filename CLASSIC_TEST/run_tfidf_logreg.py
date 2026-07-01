#!/usr/bin/env python3
"""TF-IDF + Logistic Regression baseline for CFPB complaint classification.

This script intentionally lives outside PAPER/. It reads the leakage-controlled
dataset splits and legacy subsets, trains classical supervised baselines, and
writes metrics that can later be imported into the manuscript.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]


@dataclass
class RunConfig:
    name: str
    max_train_samples: int
    max_narrative_chars: int
    class_weight: str | None
    seed: int
    max_features: int
    ngram_max: int
    min_df: int
    c_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=ROOT / "dataset/splits/train.csv")
    parser.add_argument(
        "--internal-test-file",
        type=Path,
        default=ROOT / "dataset/splits/internal_test.csv",
    )
    parser.add_argument(
        "--legacy-dir", type=Path, default=ROOT / "dataset/legacy_subsets"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "CLASSIC_TEST/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-narrative-chars", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Use balanced class weights by default because the natural split is imbalanced.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=["10000", "full"],
        help="Training budgets to run. Use integer values or 'full'.",
    )
    return parser.parse_args()


def truncate_text(series: pd.Series, max_chars: int) -> pd.Series:
    return series.astype(str).str.slice(0, max_chars)


def load_supervised_frame(path: Path, label_column: str, max_chars: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {label_column, "narrative"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame[[label_column, "narrative"]].dropna().copy()
    frame = frame.loc[frame[label_column].isin(LABELS)].copy()
    frame["text"] = truncate_text(frame["narrative"], max_chars)
    frame["label"] = frame[label_column]
    return frame[["text", "label"]]


def sample_train(frame: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    if budget <= 0 or budget >= len(frame):
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    return frame.sample(n=budget, random_state=seed).reset_index(drop=True)


def make_pipeline(config: RunConfig) -> Pipeline:
    class_weight = None if config.class_weight in (None, "none") else config.class_weight
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, config.ngram_max),
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.c_value,
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                    n_jobs=-1,
                    random_state=config.seed,
                ),
            ),
        ]
    )


def metrics_dict(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro")),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted")
        ),
        "weighted_precision": float(
            precision_score(
                y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def evaluate_frame(model: Pipeline, frame: pd.DataFrame, output_path: Path) -> dict:
    prediction = model.predict(frame["text"])
    predictions = pd.DataFrame(
        {
            "true_label": frame["label"].to_numpy(),
            "prediction": prediction,
        }
    )
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "predictions.csv", index=False)
    metrics = metrics_dict(predictions["true_label"], predictions["prediction"])
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def budget_to_int(value: str) -> int:
    return 0 if value == "full" else int(value)


def run_one(config: RunConfig, args: argparse.Namespace) -> dict:
    train_full = load_supervised_frame(
        args.train_file, label_column="product", max_chars=config.max_narrative_chars
    )
    train = sample_train(train_full, config.max_train_samples, config.seed)
    internal = load_supervised_frame(
        args.internal_test_file,
        label_column="product",
        max_chars=config.max_narrative_chars,
    )
    run_dir = args.output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train["label"].value_counts().reindex(LABELS, fill_value=0).to_csv(
        run_dir / "train_class_counts.csv"
    )

    model = make_pipeline(config)
    model.fit(train["text"], train["label"])

    internal_metrics = evaluate_frame(model, internal, run_dir / "internal_test")

    legacy_rows = []
    for legacy_path in sorted(args.legacy_dir.glob("*.csv")):
        legacy = load_supervised_frame(
            legacy_path,
            label_column="true_label",
            max_chars=config.max_narrative_chars,
        )
        dataset_name = legacy_path.name
        metrics = evaluate_frame(
            model, legacy, run_dir / "legacy_subsets" / legacy_path.stem
        )
        legacy_rows.append(
            {
                "run": config.name,
                "dataset_name": dataset_name,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    legacy_summary = pd.DataFrame(legacy_rows)
    legacy_summary.to_csv(run_dir / "legacy_summary.csv", index=False)

    return {
        "run": config.name,
        "max_train_samples": config.max_train_samples or len(train_full),
        "class_weight": config.class_weight,
        "max_narrative_chars": config.max_narrative_chars,
        "internal_accuracy": internal_metrics["accuracy"],
        "internal_macro_f1": internal_metrics["macro_f1"],
        "internal_weighted_f1": internal_metrics["weighted_f1"],
        "legacy_accuracy_mean": legacy_summary["accuracy"].mean(),
        "legacy_macro_f1_mean": legacy_summary["macro_f1"].mean(),
        "legacy_weighted_f1_mean": legacy_summary["weighted_f1"].mean(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_weight = None if args.class_weight == "none" else args.class_weight
    summary_rows = []
    for budget in args.budgets:
        budget_int = budget_to_int(budget)
        name = f"tfidf_logreg_{budget}"
        config = RunConfig(
            name=name,
            max_train_samples=budget_int,
            max_narrative_chars=args.max_narrative_chars,
            class_weight=class_weight,
            seed=args.seed,
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            c_value=args.c_value,
        )
        print(f"Running {name}...")
        summary_rows.append(run_one(config, args))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

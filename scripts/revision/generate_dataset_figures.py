#!/usr/bin/env python3
"""Create dataset figures for the manuscript."""

from __future__ import annotations
from project_paths import project_path

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = project_path(ROOT, "dataset")
FIGURES = project_path(ROOT, "REVISION_PAPER", "figures")
LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]
DISPLAY = [
    "Retail\nbanking",
    "Credit\nreporting",
    "Credit\ncard",
    "Mortgages\nand loans",
    "Debt\ncollection",
]


def load_main() -> pd.DataFrame:
    parts = []
    for name, path in [
        ("train", DATASET / "splits" / "train.csv"),
        ("validation", DATASET / "splits" / "validation.csv"),
        ("internal test", DATASET / "splits" / "internal_test.csv"),
    ]:
        frame = pd.read_csv(path, usecols=["product", "narrative"])
        frame["split"] = name
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def narrative_length_histogram() -> None:
    frame = load_main()
    char_lengths = frame["narrative"].astype(str).str.len()
    word_lengths = frame["narrative"].astype(str).str.split().str.len()

    summary = pd.DataFrame(
        [
            {
                "measure": "characters",
                "mean": char_lengths.mean(),
                "median": char_lengths.median(),
                "p75": char_lengths.quantile(0.75),
                "p90": char_lengths.quantile(0.90),
                "p95": char_lengths.quantile(0.95),
                "p99": char_lengths.quantile(0.99),
                "share_le_500": (char_lengths <= 500).mean(),
            },
            {
                "measure": "whitespace_tokens",
                "mean": word_lengths.mean(),
                "median": word_lengths.median(),
                "p75": word_lengths.quantile(0.75),
                "p90": word_lengths.quantile(0.90),
                "p95": word_lengths.quantile(0.95),
                "p99": word_lengths.quantile(0.99),
                "share_le_512": (word_lengths <= 512).mean(),
            },
        ]
    )
    tables = project_path(ROOT, "REVISION_PAPER", "tables")
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "dataset_length_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(np.clip(char_lengths, 0, 4000), bins=60, color="#4C78A8")
    axes[0].axvline(500, color="#E45756", linestyle="--", linewidth=1.5)
    axes[0].set_title("Narrative length in characters")
    axes[0].set_xlabel("Characters (clipped at 4000)")
    axes[0].set_ylabel("Number of complaints")
    axes[0].text(520, axes[0].get_ylim()[1] * 0.82, "500-char\ntraining limit", color="#E45756")

    axes[1].hist(np.clip(word_lengths, 0, 800), bins=60, color="#59A14F")
    axes[1].set_title("Approximate sequence length")
    axes[1].set_xlabel("Whitespace tokens (clipped at 800)")
    axes[1].set_ylabel("Number of complaints")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "narrative_length_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def class_distribution() -> None:
    frame = load_main()
    split_counts = (
        frame.groupby(["split", "product"]).size().unstack(fill_value=0).reindex(columns=LABELS)
    )
    split_pct = split_counts.div(split_counts.sum(axis=1), axis=0) * 100
    tables = project_path(ROOT, "REVISION_PAPER", "tables")
    tables.mkdir(parents=True, exist_ok=True)
    split_counts.to_csv(tables / "dataset_class_counts.csv")
    split_pct.to_csv(tables / "dataset_class_percentages.csv")

    legacy_rows = []
    for path in sorted((DATASET / "legacy_subsets").glob("*.csv")):
        kind = "balanced" if "_balanced_" in path.name else "imbalanced"
        legacy = pd.read_csv(path, usecols=["true_label"])
        counts = legacy["true_label"].value_counts().reindex(LABELS, fill_value=0)
        pct = counts / counts.sum() * 100
        legacy_rows.append({"kind": kind, **pct.to_dict()})
    legacy_pct = pd.DataFrame(legacy_rows).groupby("kind")[LABELS].mean()
    legacy_pct.to_csv(tables / "legacy_class_percentages.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(LABELS))
    width = 0.25
    for offset, split in zip([-width, 0, width], ["train", "validation", "internal test"]):
        axes[0].bar(x + offset, split_pct.loc[split, LABELS], width=width, label=split)
    axes[0].set_title("Natural split class distribution")
    axes[0].set_ylabel("Share of examples (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(DISPLAY, fontsize=8)
    axes[0].legend(fontsize=8)

    width = 0.35
    axes[1].bar(x - width / 2, legacy_pct.loc["balanced", LABELS], width=width, label="legacy balanced")
    axes[1].bar(x + width / 2, legacy_pct.loc["imbalanced", LABELS], width=width, label="legacy imbalanced")
    axes[1].set_title("Legacy benchmark distributions")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DISPLAY, fontsize=8)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(65, split_pct.max().max() + 5))
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "class_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    narrative_length_histogram()
    class_distribution()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()

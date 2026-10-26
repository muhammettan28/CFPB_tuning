#!/usr/bin/env python3
"""Generate statistical tables for the CFPB QLoRA manuscript.

The script uses only archived experiment outputs under experiment_archive/outputs.
It produces CSV and LaTeX table fragments under PAPER/tables.
"""

from __future__ import annotations
from project_paths import project_path

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = project_path(ROOT, "experiment_archive", "outputs")
TABLES = project_path(ROOT, "REVISION_PAPER", "tables")
LABELS = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]
MODEL_SPECS = {
    "Qwen3-8B": {
        "internal_summary": "qwen3_8b_final_evaluation_summary.csv",
        "legacy_summary": "qwen3_8b_legacy_final_evaluation_summary.csv",
        "internal_glob": "qwen3_8b_final_eval_*/internal_test/predictions.csv",
        "train_glob": "qwen3_8b_*/*",
    },
    "Llama 3.1-8B": {
        "internal_summary": "llama31_8b_final_internal_summary.csv",
        "legacy_summary": "llama31_8b_legacy_final_summary.csv",
        "internal_glob": "llama31_8b_final_eval_*/internal_test/predictions.csv",
        "train_glob": "llama31_8b_fixed_protocol_seed*/*",
    },
}


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))
    f1s = []
    supports = []
    for label in LABELS:
        true_pos = int(np.sum((y_true == label) & (y_pred == label)))
        false_pos = int(np.sum((y_true != label) & (y_pred == label)))
        false_neg = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)
        supports.append(support)
    supports_array = np.asarray(supports, dtype=float)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports_array)),
    }


def pct(value: float) -> str:
    return f"{value:.4f}"


def ci_text(mean: float, low: float, high: float) -> str:
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def variant_from_path(path: Path) -> str:
    text = path.as_posix()
    if "base_zero_shot" in text:
        return "base_zero_shot"
    for seed in ("42", "43", "44"):
        if f"seed{seed}" in text:
            return f"fine_tuned_seed{seed}"
    return "fine_tuned"


def load_internal_predictions(pattern: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(ARCHIVE.glob(pattern)):
        variant = variant_from_path(path)
        frame = pd.read_csv(path)
        frames[variant] = frame
    expected = {
        "base_zero_shot",
        "fine_tuned_seed42",
        "fine_tuned_seed43",
        "fine_tuned_seed44",
    }
    missing = expected - set(frames)
    if missing:
        raise RuntimeError(f"Missing internal predictions for {pattern}: {missing}")
    return frames


def try_load_internal_predictions(pattern: str) -> dict[str, pd.DataFrame] | None:
    try:
        return load_internal_predictions(pattern)
    except RuntimeError as exc:
        print(f"WARNING: {exc}")
        return None


def bootstrap_internal_ci(
    model_name: str, frames: dict[str, pd.DataFrame], n_boot: int = 300
) -> list[dict[str, str]]:
    rng = np.random.default_rng(20260701)
    base = frames["base_zero_shot"]
    seed_frames = [
        frames["fine_tuned_seed42"],
        frames["fine_tuned_seed43"],
        frames["fine_tuned_seed44"],
    ]
    n = len(base)
    y_true = base["true_label"].to_numpy()
    base_pred = base["prediction"].to_numpy()
    seed_preds = [frame["prediction"].to_numpy() for frame in seed_frames]

    rows = []
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        base_samples = []
        ft_samples = []
        gain_samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b = metric_values(y_true[idx], base_pred[idx])[metric]
            seed_vals = [
                metric_values(y_true[idx], pred[idx])[metric] for pred in seed_preds
            ]
            f = float(np.mean(seed_vals))
            base_samples.append(b)
            ft_samples.append(f)
            gain_samples.append(f - b)
        b_mean = float(np.mean(base_samples))
        f_mean = float(np.mean(ft_samples))
        g_mean = float(np.mean(gain_samples))
        b_low, b_high = np.quantile(base_samples, [0.025, 0.975])
        f_low, f_high = np.quantile(ft_samples, [0.025, 0.975])
        g_low, g_high = np.quantile(gain_samples, [0.025, 0.975])
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "base_mean": b_mean,
                "base_low": float(b_low),
                "base_high": float(b_high),
                "fine_tuned_mean": f_mean,
                "fine_tuned_low": float(f_low),
                "fine_tuned_high": float(f_high),
                "gain_mean": g_mean,
                "gain_low": float(g_low),
                "gain_high": float(g_high),
            }
        )
    return rows


def write_internal_ci() -> pd.DataFrame:
    all_rows = []
    for model_name, spec in MODEL_SPECS.items():
        frames = try_load_internal_predictions(spec["internal_glob"])
        if frames is None:
            continue
        all_rows.extend(bootstrap_internal_ci(model_name, frames))
    out = pd.DataFrame(all_rows)
    out.to_csv(TABLES / "internal_bootstrap_ci.csv", index=False)

    tex_rows = []
    if all_rows:
        for row in all_rows:
            tex_rows.append(
                " & ".join(
                    [
                        row["model"],
                        row["metric"].replace("_", " "),
                        ci_text(row["base_mean"], row["base_low"], row["base_high"]),
                        ci_text(
                            row["fine_tuned_mean"],
                            row["fine_tuned_low"],
                            row["fine_tuned_high"],
                        ),
                        ci_text(row["gain_mean"], row["gain_low"], row["gain_high"]),
                    ]
                )
                + r" \\"
            )
    (TABLES / "table_internal_bootstrap_ci.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def write_internal_summary_table() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        summary = pd.read_csv(ARCHIVE / spec["internal_summary"])
        base = summary.loc[summary["variant"].eq("base_zero_shot")].iloc[0]
        ft = summary.loc[~summary["variant"].eq("base_zero_shot")]
        rows.append(
            {
                "model": model_name,
                "base_accuracy": base["accuracy"],
                "ft_accuracy_mean": ft["accuracy"].mean(),
                "accuracy_gain": ft["accuracy"].mean() - base["accuracy"],
                "base_macro_f1": base["macro_f1"],
                "ft_macro_f1_mean": ft["macro_f1"].mean(),
                "macro_f1_gain": ft["macro_f1"].mean() - base["macro_f1"],
                "base_weighted_f1": base["weighted_f1"],
                "ft_weighted_f1_mean": ft["weighted_f1"].mean(),
                "weighted_f1_gain": ft["weighted_f1"].mean() - base["weighted_f1"],
                "base_debt_f1": base["debt_collection_f1"],
                "ft_debt_f1_mean": ft["debt_collection_f1"].mean(),
                "debt_f1_gain": ft["debt_collection_f1"].mean()
                - base["debt_collection_f1"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "internal_test_summary.csv", index=False)
    tex_rows = []
    for row in rows:
        tex_rows.append(
            f"{row['model']} & {row['base_macro_f1']:.4f} & "
            f"{row['ft_macro_f1_mean']:.4f} & {row['macro_f1_gain']:.4f} & "
            f"{row['base_weighted_f1']:.4f} & {row['ft_weighted_f1_mean']:.4f} & "
            f"{row['weighted_f1_gain']:.4f} & {row['ft_debt_f1_mean']:.4f} \\\\"
        )
    (TABLES / "table_internal_summary.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def write_wilcoxon_tables() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        summary = pd.read_csv(ARCHIVE / spec["legacy_summary"])
        base = summary.loc[summary["variant"].eq("base_zero_shot")].copy()
        ft = summary.loc[~summary["variant"].eq("base_zero_shot")].copy()
        ft_mean = (
            ft.groupby("dataset_name")[["accuracy", "macro_f1", "weighted_f1"]]
            .mean()
            .reset_index()
        )
        paired = base[["dataset_name", "accuracy", "macro_f1", "weighted_f1"]].merge(
            ft_mean, on="dataset_name", suffixes=("_base", "_ft")
        )
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            diffs = paired[f"{metric}_ft"] - paired[f"{metric}_base"]
            stat, pvalue = wilcoxon(
                paired[f"{metric}_ft"],
                paired[f"{metric}_base"],
                alternative="greater",
                zero_method="wilcox",
            )
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "n_subsets": len(paired),
                    "base_mean": paired[f"{metric}_base"].mean(),
                    "fine_tuned_mean": paired[f"{metric}_ft"].mean(),
                    "mean_gain": diffs.mean(),
                    "median_gain": diffs.median(),
                    "wilcoxon_statistic": stat,
                    "p_value_greater": pvalue,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "legacy_wilcoxon.csv", index=False)
    tex_rows = []
    for row in rows:
        tex_rows.append(
            f"{row['model']} & {row['metric'].replace('_', ' ')} & "
            f"{row['base_mean']:.4f} & {row['fine_tuned_mean']:.4f} & "
            f"{row['mean_gain']:.4f} & {row['median_gain']:.4f} & "
            f"{row['p_value_greater']:.4g} \\\\"
        )
    (TABLES / "table_legacy_wilcoxon.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def write_legacy_summary_table() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        summary = pd.read_csv(ARCHIVE / spec["legacy_summary"])
        base = summary.loc[summary["variant"].eq("base_zero_shot")]
        ft = summary.loc[~summary["variant"].eq("base_zero_shot")]
        rows.append(
            {
                "model": model_name,
                "base_accuracy": base["accuracy"].mean(),
                "ft_accuracy": ft["accuracy"].mean(),
                "accuracy_gain": ft["accuracy"].mean() - base["accuracy"].mean(),
                "base_macro_f1": base["macro_f1"].mean(),
                "ft_macro_f1": ft["macro_f1"].mean(),
                "macro_f1_gain": ft["macro_f1"].mean() - base["macro_f1"].mean(),
                "base_weighted_f1": base["weighted_f1"].mean(),
                "ft_weighted_f1": ft["weighted_f1"].mean(),
                "weighted_f1_gain": ft["weighted_f1"].mean()
                - base["weighted_f1"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "legacy_summary.csv", index=False)
    tex_rows = []
    for row in rows:
        tex_rows.append(
            f"{row['model']} & {row['base_macro_f1']:.4f} & "
            f"{row['ft_macro_f1']:.4f} & {row['macro_f1_gain']:.4f} & "
            f"{row['base_weighted_f1']:.4f} & {row['ft_weighted_f1']:.4f} & "
            f"{row['weighted_f1_gain']:.4f} \\\\"
        )
    (TABLES / "table_legacy_summary.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Counter:
    counter = Counter()
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            counter[(true, pred)] += 1
    return counter


def write_error_analysis() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        frames = try_load_internal_predictions(spec["internal_glob"])
        if frames is None:
            continue
        base = frames["base_zero_shot"]
        base_counts = confusion_counts(base["true_label"], base["prediction"])
        for (true, pred), count in base_counts.most_common(8):
            rows.append(
                {
                    "model": model_name,
                    "variant": "base_zero_shot",
                    "true_label": true,
                    "predicted_label": pred,
                    "mean_count": float(count),
                }
            )
        aggregate = defaultdict(float)
        for variant in [
            "fine_tuned_seed42",
            "fine_tuned_seed43",
            "fine_tuned_seed44",
        ]:
            counts = confusion_counts(
                frames[variant]["true_label"], frames[variant]["prediction"]
            )
            for key, count in counts.items():
                aggregate[key] += count / 3.0
        for (true, pred), count in Counter(aggregate).most_common(8):
            rows.append(
                {
                    "model": model_name,
                    "variant": "fine_tuned_mean",
                    "true_label": true,
                    "predicted_label": pred,
                    "mean_count": float(count),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "internal_error_analysis_top_confusions.csv", index=False)
    tex_rows = []
    for row in rows:
        true_label = row["true_label"].replace("_", r"\_")
        predicted_label = row["predicted_label"].replace("_", r"\_")
        tex_rows.append(
            f"{row['model']} & {row['variant'].replace('_', ' ')} & "
            f"{true_label} $\\rightarrow$ "
            f"{predicted_label} & "
            f"{row['mean_count']:.1f} \\\\"
        )
    (TABLES / "table_error_analysis.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def write_latency_cost_table() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        internal = pd.read_csv(ARCHIVE / spec["internal_summary"])
        base = internal.loc[internal["variant"].eq("base_zero_shot")].iloc[0]
        ft = internal.loc[~internal["variant"].eq("base_zero_shot")]
        rows.append(
            {
                "model": model_name,
                "base_latency_seconds": base["mean_latency_seconds"],
                "fine_tuned_latency_seconds": ft["mean_latency_seconds"].mean(),
                "latency_delta_seconds": ft["mean_latency_seconds"].mean()
                - base["mean_latency_seconds"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "latency_summary.csv", index=False)
    tex_rows = []
    for row in rows:
        tex_rows.append(
            f"{row['model']} & {row['base_latency_seconds']:.3f} & "
            f"{row['fine_tuned_latency_seconds']:.3f} & "
            f"{row['latency_delta_seconds']:.3f} \\\\"
        )
    (TABLES / "table_latency.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_internal_summary_table()
    write_internal_ci()
    write_legacy_summary_table()
    write_wilcoxon_tables()
    write_error_analysis()
    write_latency_cost_table()
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()

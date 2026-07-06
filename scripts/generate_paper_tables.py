#!/usr/bin/env python3
"""Generate all supplementary LaTeX table files for the manuscript revision.

Produces:
  - table_off_label.tex          -- off-label counts (base vs FT, internal + legacy)
  - table_internal_summary_v2.tex -- internal summary with per-seed std dev
  - table_legacy_subset_overview.tex -- 12-subset structure
  - table_seed_stability.tex     -- per-seed macro-F1 and std for both models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1] / "experiment_archive" / "outputs"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "dataset" / "legacy_subsets"
OUT = Path(__file__).resolve().parents[1] / "PAPER" / "tables"
OUT.mkdir(exist_ok=True)

LABELS = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]


def load(fname: str) -> pd.DataFrame:
    return pd.read_csv(ARCHIVE / fname)


# ── 1. Load internal-test summaries ─────────────────────────────────────────
q_int = load("qwen3_8b_final_evaluation_summary.csv")
l_int = load("llama31_8b_final_internal_summary.csv")

q_base = q_int[q_int.base_only].iloc[0]
l_base = l_int[l_int.base_only].iloc[0]
q_ft   = q_int[~q_int.base_only]
l_ft   = l_int[~l_int.base_only]

# ── 2. Load legacy summaries ─────────────────────────────────────────────────
q_leg = load("qwen3_8b_legacy_final_evaluation_summary.csv")
l_leg = load("llama31_8b_legacy_final_summary.csv")

q_leg_base = q_leg[q_leg.variant == "base_zero_shot"]
l_leg_base = l_leg[l_leg.variant == "base_zero_shot"]
q_leg_ft   = q_leg[q_leg.variant.str.startswith("fine_tuned")]
l_leg_ft   = l_leg[l_leg.variant.str.startswith("fine_tuned")]


# ── 3. TABLE: Off-label rates ────────────────────────────────────────────────
n_int = int(q_base["n"])  # 15428
n_leg_total = 12 * 1000   # 12000 per evaluation pass; ×3 seeds for FT

rows = []
for model, base_int, ft_int, base_leg, ft_leg in [
    ("Qwen3-8B",      q_base, q_ft, q_leg_base, q_leg_ft),
    ("Llama 3.1-8B",  l_base, l_ft, l_leg_base, l_leg_ft),
]:
    base_off_int = int(base_int["off_label_count"])
    ft_off_int   = int(ft_int["off_label_count"].sum())   # sum over 3 seeds × n
    base_off_leg = int(base_leg["off_label_count"].sum())
    ft_off_leg   = int(ft_leg["off_label_count"].sum())

    ft_n_int = n_int * 3
    ft_n_leg = n_leg_total * 3

    rows.append(
        f"{model} & base & "
        f"{base_off_int} & {n_int} & {100*base_off_int/n_int:.3f}\\% & "
        f"{base_off_leg} & {n_leg_total} & {100*base_off_leg/n_leg_total:.3f}\\% \\\\"
    )
    rows.append(
        f"{model} & fine-tuned (mean) & "
        f"{ft_off_int} & {ft_n_int} & {100*ft_off_int/ft_n_int:.3f}\\% & "
        f"{ft_off_leg} & {ft_n_leg} & {100*ft_off_leg/ft_n_leg:.3f}\\% \\\\"
    )

off_label_tex = "\n".join(rows)
(OUT / "table_off_label.tex").write_text(off_label_tex)
print("Wrote table_off_label.tex")
print(off_label_tex)


# ── 4. TABLE: Internal summary with std dev ──────────────────────────────────
rows2 = []
for model, base_r, ft in [
    ("Qwen3-8B",     q_base, q_ft),
    ("Llama 3.1-8B", l_base, l_ft),
]:
    mf1_mean = ft.macro_f1.mean()
    mf1_std  = ft.macro_f1.std(ddof=0)
    wf1_mean = ft.weighted_f1.mean()
    wf1_std  = ft.weighted_f1.std(ddof=0)
    debt_mean = ft.debt_collection_f1.mean()
    debt_std  = ft.debt_collection_f1.std(ddof=0)

    gain_mf1 = mf1_mean - base_r.macro_f1
    gain_wf1 = wf1_mean - base_r.weighted_f1

    rows2.append(
        f"{model} & {base_r.macro_f1:.4f} & "
        f"{mf1_mean:.4f}$\\pm${mf1_std:.4f} & {gain_mf1:.4f} & "
        f"{base_r.weighted_f1:.4f} & "
        f"{wf1_mean:.4f}$\\pm${wf1_std:.4f} & {gain_wf1:.4f} & "
        f"{debt_mean:.4f}$\\pm${debt_std:.4f} \\\\"
    )

(OUT / "table_internal_summary_v2.tex").write_text("\n".join(rows2))
print("\nWrote table_internal_summary_v2.tex")
print("\n".join(rows2))


# ── 5. TABLE: Seed stability (per-seed results) ──────────────────────────────
rows3 = []
for model, ft in [("Qwen3-8B", q_ft), ("Llama 3.1-8B", l_ft)]:
    for _, row in ft.sort_values("seed").iterrows():
        seed = int(row["seed"])
        rows3.append(
            f"{model} & {seed} & "
            f"{row.accuracy:.4f} & {row.macro_f1:.4f} & {row.weighted_f1:.4f} & "
            f"{row.debt_collection_f1:.4f} \\\\"
        )
    ft_sorted = ft.sort_values("seed")
    rows3.append(
        f"\\quad mean$\\pm$std & -- & "
        f"{ft.accuracy.mean():.4f}$\\pm${ft.accuracy.std(ddof=0):.4f} & "
        f"{ft.macro_f1.mean():.4f}$\\pm${ft.macro_f1.std(ddof=0):.4f} & "
        f"{ft.weighted_f1.mean():.4f}$\\pm${ft.weighted_f1.std(ddof=0):.4f} & "
        f"{ft.debt_collection_f1.mean():.4f}$\\pm${ft.debt_collection_f1.std(ddof=0):.4f} \\\\"
    )
    rows3.append("\\hline")

(OUT / "table_seed_stability.tex").write_text("\n".join(rows3))
print("\nWrote table_seed_stability.tex")
print("\n".join(rows3))


# ── 6. TABLE: Legacy subset overview ────────────────────────────────────────
import os
sub_rows = []
for fname in sorted(os.listdir(LEGACY_DIR)):
    df = pd.read_csv(LEGACY_DIR / fname)
    vc = df["true_label"].value_counts()
    name = fname.replace(".csv", "")
    # parse fields
    parts = name.split("_")  # e.g. 1 subset balanced 250
    group = parts[0]
    sampling = parts[2]      # balanced / imbalanced
    max_chars = parts[3]     # 250 / 500
    n_total = len(df)
    n_per_cls = f"{vc.min()}--{vc.max()}" if vc.min() != vc.max() else str(vc.min())
    sub_rows.append(
        f"{group} & {sampling.capitalize()} & {max_chars} & {n_total} & {n_per_cls} \\\\"
    )

(OUT / "table_legacy_subset_overview.tex").write_text("\n".join(sub_rows))
print("\nWrote table_legacy_subset_overview.tex")
print("\n".join(sub_rows))


# ── 7. TABLE: Encoder baseline ────────────────────────────────────────────────
ENCODER_DIR = Path(__file__).resolve().parents[1] / "CLASSIC_TEST" / "outputs" / "encoder_baselines_10k_len512"
enc_rows = []
for short_name in ["BERT_base", "DistilBERT", "RoBERTa_base"]:
    jf = ENCODER_DIR / f"results_{short_name}.json"
    if jf.exists():
        import json
        r = json.loads(jf.read_text())
        it = r["internal_test"]
        debt_f1 = it["per_class_f1"].get("debt_collection", 0)
        display = r["model"]
        enc_rows.append(
            f"{display} & {it['accuracy']:.4f} & {it['macro_f1']:.4f} & "
            f"{it['weighted_f1']:.4f} & {r['legacy_mean_macro_f1']:.4f} & {debt_f1:.4f} \\\\"
        )
    else:
        display = short_name.replace("_", "-")
        enc_rows.append(f"{display} & -- & -- & -- & -- & -- \\\\")

(OUT / "table_encoder_baseline.tex").write_text("\n".join(enc_rows))
print("\nWrote table_encoder_baseline.tex")
print("\n".join(enc_rows))

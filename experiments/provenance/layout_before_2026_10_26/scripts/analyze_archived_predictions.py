#!/usr/bin/env python3
"""CPU-only paired cluster bootstrap and descriptive archived-prediction audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from audit_revision_data import ROOT, OUT, ARCHIVE, PROMPT, NORM, dependency_groups

LABELS = PROMPT["LABELS"]
VARIANTS = ["base_zero_shot", "fine_tuned_seed42", "fine_tuned_seed43", "fine_tuned_seed44"]
METRICS = ["accuracy", "macro_f1", "weighted_f1"]
SPECS = {
    "Qwen3-8B": ("qwen3_8b_final_evaluation_summary.csv", "qwen3_8b_legacy_final_evaluation_summary.csv"),
    "Llama-3.1-8B": ("llama31_8b_final_internal_summary.csv", "llama31_8b_legacy_final_summary.csv"),
}


def confusion_metrics(cm: np.ndarray) -> np.ndarray:
    """Accept (..., true=5, predicted=6), including off-label false negatives."""
    support = cm.sum(axis=-1)
    predicted = cm[..., :5].sum(axis=-2)
    tp = np.diagonal(cm[..., :5], axis1=-2, axis2=-1)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    n = support.sum(axis=-1)
    if np.any(n == 0):
        raise ValueError("Empty bootstrap replicate")
    return np.stack([tp.sum(axis=-1) / n, f1.mean(axis=-1), (f1 * support).sum(axis=-1) / n], axis=-1)


def group_confusions(y: np.ndarray, predictions: np.ndarray, groups: np.ndarray,
                     weights: np.ndarray | None = None) -> np.ndarray:
    _, indices = np.unique(groups, return_inverse=True)
    cm = np.zeros((indices.max() + 1, predictions.shape[1], 5, 6), dtype=float)
    for variant in range(predictions.shape[1]):
        np.add.at(cm[:, variant], (indices, y, predictions[:, variant]), 1 if weights is None else weights)
    return cm


def paired_cluster_bootstrap(cm: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample G clusters uniformly with replacement; keep each entire cluster.

    Identical cluster confusion profiles can be collapsed into multinomial
    categories without changing the ordinary nonparametric cluster-bootstrap
    distribution. All variants share every cluster multiplicity. Seeds stay
    fixed; this interval does not integrate training-seed uncertainty.
    """
    g, variants = cm.shape[:2]
    profiles, frequencies = np.unique(cm.reshape(g, -1), axis=0, return_counts=True)
    probabilities = frequencies / g
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, variants, len(METRICS)))
    for start in range(0, n_boot, 128):
        multiplicities = rng.multinomial(g, probabilities, size=min(128, n_boot - start))
        total = (multiplicities @ profiles).reshape(-1, variants, 5, 6)
        samples[start:start + len(total)] = confusion_metrics(total)
    return confusion_metrics(cm.sum(axis=0)), samples


def load_predictions(model: str, role: str, data: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    summary = pd.read_csv(ARCHIVE / SPECS[model][role == "legacy"])
    arrays = []
    evidence = []
    for variant in VARIANTS:
        is_base = variant == "base_zero_shot"
        seed = 42 if is_base else int(variant[-2:])
        rows = summary.loc[summary.base_only.eq(is_base) & summary.seed.eq(seed)]
        result = np.full(len(data), "__MISSING__", dtype=object)
        for row in rows.itertuples():
            path = ARCHIVE / row.run_dir / "predictions.csv"
            pred = pd.read_csv(path, keep_default_na=False)
            target = data if role == "internal" else data.loc[data.source_file.eq(row.dataset_name)]
            if role == "internal":
                if pred.record_id.duplicated().any():
                    raise ValueError(f"Duplicate record ID: {path}")
                if set(pred.record_id) != set(target.record_id):
                    raise ValueError(f"Record IDs do not match: {path}")
                pred = pred.set_index("record_id").loc[target.record_id].reset_index()
            # Legacy predictions have no record IDs: original evaluator loops
            # over all CSV rows, in order. Verify all labels and row counts.
            if len(pred) != len(target) or pred.true_label.tolist() != target.true_label.tolist():
                raise ValueError(f"Prediction alignment failure: {path}")
            result[target.index] = pred.prediction.to_numpy()
            measured = [accuracy_score(pred.true_label, pred.prediction),
                        f1_score(pred.true_label, pred.prediction, labels=LABELS, average="macro", zero_division=0),
                        f1_score(pred.true_label, pred.prediction, labels=LABELS, average="weighted", zero_division=0)]
            for metric, actual in zip(METRICS, measured):
                if not np.isclose(actual, getattr(row, metric), atol=1e-12, rtol=0):
                    raise ValueError(f"Archived metric mismatch: {path} {metric}")
            evidence.append({"model": model, "role": role, "variant": variant,
                             "path": str(path.relative_to(ROOT)), "rows": len(pred),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "alignment": "record_id + all labels" if role == "internal" else "source file row order + row count + all labels; no record IDs stored"})
        if (result == "__MISSING__").any():
            raise ValueError(f"Incomplete predictions: {model}, {role}, {variant}")
        unknown = set(result) - set(LABELS) - {"__off_label__"}
        if unknown:
            raise ValueError(f"Unexpected predictions: {unknown}")
        arrays.append(result)
    return np.stack(arrays, axis=1), evidence


def encode(labels: np.ndarray) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(LABELS + ["__off_label__"])}
    return np.vectorize(mapping.__getitem__, otypes=[int])(labels)


def summarize_bootstrap(model: str, analysis: str, y: np.ndarray, predictions: np.ndarray,
                        groups: np.ndarray, n_boot: int, seed: int,
                        weights: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    cm = group_confusions(y, predictions, groups, weights)
    point, samples = paired_cluster_bootstrap(cm, n_boot, seed)
    # Both point estimates and CIs use the same metric computation.
    categories = np.stack([samples[:, 0], samples[:, 1:].mean(axis=1), samples[:, 1:].mean(axis=1) - samples[:, 0]], axis=1)
    observed = np.stack([point[0], point[1:].mean(axis=0), point[1:].mean(axis=0) - point[0]], axis=0)
    low, high = np.quantile(categories, [.025, .975], axis=0)
    rows, seed_rows = [], []
    for k, metric in enumerate(METRICS):
        rows.append({"model": model, "analysis": analysis, "metric": metric,
                     "rows": len(y), "groups": len(cm), "bootstrap_replicates": n_boot,
                     "base_point": observed[0, k], "base_low": low[0, k], "base_high": high[0, k],
                     "ft_seed_mean_point": observed[1, k], "ft_low": low[1, k], "ft_high": high[1, k],
                     "paired_gain_point": observed[2, k], "gain_low": low[2, k], "gain_high": high[2, k],
                     "ft_seed_sd_ddof1": point[1:, k].std(ddof=1)})
        for j, variant in enumerate(VARIANTS):
            seed_rows.append({"model": model, "analysis": analysis, "variant": variant, "metric": metric, "value": point[j, k]})
    np.savez_compressed(OUT / f"bootstrap_{model}_{analysis}.npz", samples=categories,
                        category=np.array(["base", "ft_seed_mean", "paired_gain"]), metric=np.array(METRICS))
    print(model, analysis, "rows", len(y), "groups", len(cm), "macro-F1 gain", observed[2, 1], "CI", low[2, 1], high[2, 1], flush=True)
    return rows, seed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    membership = pd.read_csv(OUT / "input_membership.csv.gz")
    internal = pd.read_csv(ROOT / "dataset/splits/internal_test.csv").rename(columns={"product": "true_label"})
    keys = membership.loc[membership.dataset.eq("internal_test")].set_index("record_id")
    for col in ["full_normalized", "normalized_500", "qwen_tokens512", "llama_tokens512"]:
        internal[col] = internal.record_id.map(keys[col])
    internal["group"] = dependency_groups(internal, ["full_normalized", "normalized_500", "qwen_tokens512", "llama_tokens512"])
    unsafe = np.zeros(len(internal), dtype=bool)
    # Common conservative sensitivity subset, fixed across models and seeds.
    # Excludes matches anywhere in the train OR validation pools, beyond the
    # actually selected 10k/2k, and propagates exclusions to whole components.
    for col in ["normalized_500", "qwen_tokens512", "llama_tokens512"]:
        pool = set(membership.loc[membership.dataset.isin(["train", "validation"]), col])
        unsafe |= internal[col].isin(pool).to_numpy()
    clean = ~internal.group.isin(set(internal.loc[unsafe, "group"])).to_numpy()
    internal[["record_id", "group"]].assign(conservative_clean=clean).to_csv(OUT / "internal_sensitivity_manifest.csv", index=False)
    legacy = pd.read_csv(ROOT / "dataset/splits/legacy_test_manifest.csv")
    groups = pd.read_csv(OUT / "legacy_groups.csv")
    assert legacy[["source_file", "source_row", "true_label"]].equals(groups[["source_file", "source_row", "true_label"]])
    legacy["full_normalized"] = groups.full_normalized
    legacy["group"] = groups.prefix250_group
    pool_prefixes = set()
    for role in ["train", "validation"]:
        source = pd.read_csv(ROOT / f"dataset/splits/{role}.csv", usecols=["narrative"])
        pool_prefixes.update(source.narrative.astype(str).str[:250].map(NORM))
    prefix_hit = legacy.narrative.astype(str).str[:250].map(NORM).isin(pool_prefixes)
    legacy_clean = ~legacy.group.isin(set(legacy.loc[prefix_hit, "group"])).to_numpy()
    legacy[["source_file", "source_row", "group"]].assign(conservative_prefix_clean=legacy_clean).to_csv(OUT / "legacy_sensitivity_manifest.csv", index=False)
    # Equal weight per distinct (label, normalized visible narrative) unit;
    # repeated subset appearances share that unit's weight. Different 250/500
    # inputs remain separate units but their dependency cluster stays together.
    unit_counts = legacy.groupby(["true_label", "full_normalized"])["source_row"].transform("size")
    unique_weights = 1.0 / unit_counts.to_numpy()
    all_rows, all_seeds, evidence, strata, conflicts = [], [], [], [], []
    for model in SPECS:
        int_pred, ev = load_predictions(model, "internal", internal)
        evidence.extend(ev)
        legacy_pred, ev = load_predictions(model, "legacy", legacy)
        evidence.extend(ev)
        for variant_index, variant in enumerate(VARIANTS):
            temp = legacy[["true_label", "full_normalized"]].assign(prediction=legacy_pred[:, variant_index])
            conflicts.append({"model": model, "variant": variant,
                              "normalized_units_with_different_predictions_across_appearances": int((temp.groupby(["true_label", "full_normalized"]).prediction.nunique() > 1).sum())})
        for name, data, pred, mask, weights in [
            ("internal_original", internal, int_pred, np.ones(len(internal), dtype=bool), None),
            ("internal_conservative_clean", internal, int_pred, clean, None),
            ("legacy_unique_input_weighted", legacy, legacy_pred, np.ones(len(legacy), dtype=bool), unique_weights),
            ("legacy_conservative_prefix_clean", legacy, legacy_pred, legacy_clean, unique_weights),
        ]:
            rows, seeds = summarize_bootstrap(model, name, encode(data.loc[mask, "true_label"].to_numpy()),
                encode(pred[mask]), data.loc[mask, "group"].to_numpy(), args.replicates, args.seed,
                None if weights is None else weights[mask])
            all_rows.extend(rows); all_seeds.extend(seeds)
        # Descriptive strata do not claim an experimentally controlled context
        # ablation: original complaint length/content and subset composition vary.
        for role, data, pred, stratum in [
            ("internal", internal, int_pred, pd.cut(internal.narrative.str.len(), [0, 250, 500, np.inf], labels=["<=250_chars", "251-500_chars", ">500_chars"], include_lowest=True)),
            ("legacy", legacy, legacy_pred, legacy.max_chars.astype(str) + "chars_" + legacy.condition),
        ]:
            for name in sorted(stratum.unique()):
                mask = (stratum == name).to_numpy()
                cm = group_confusions(encode(data.loc[mask, "true_label"].to_numpy()), encode(pred[mask]), np.arange(mask.sum())).sum(axis=0)
                values = confusion_metrics(cm)
                for j, variant in enumerate(VARIANTS):
                    strata.append({"model": model, "role": role, "stratum": str(name), "variant": variant, "rows": int(mask.sum()), **dict(zip(METRICS, values[j]))})
    pd.DataFrame(all_rows).to_csv(OUT / "paired_group_bootstrap.csv", index=False)
    pd.DataFrame(all_seeds).to_csv(OUT / "per_seed_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "descriptive_truncation_strata.csv", index=False)
    pd.DataFrame(conflicts).to_csv(OUT / "legacy_repeat_prediction_consistency.csv", index=False)
    (OUT / "prediction_sources.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (OUT / "statistical_method.json").write_text(json.dumps({
        "replicates": args.replicates, "rng_seed": args.seed, "interval": "95% paired cluster percentile bootstrap",
        "internal_groups": "Connected components sharing full normalized narrative OR normalized raw500 text OR either model's reconstructed 512-token input; labels never used to split groups.",
        "legacy_groups": "Conservative components sharing normalized first 250 raw characters or full normalized narrative; not verified complaint identity.",
        "legacy_estimand": "Equal weight per unique (true label, full normalized visible narrative) unit. Repeated appearances have inverse-frequency weights; different 250/500 visible texts are separate units clustered together. Different predictions for repeated normalized inputs, if any, are averaged at the confusion-count level, not silently discarded.",
        "internal_estimand": "Original row-weighted metrics; all rows of a resampled dependency group retained.",
        "ft_estimator": "Arithmetic mean of the three independently scored fixed training seeds, not ensemble voting and not pooled confusion counts.",
        "seed_variation": "Sample SD across three fixed seed scores (ddof=1), reported separately. Bootstrap conditions on these trained models; it does not estimate training-seed uncertainty.",
        "point_estimates": "Computed on observed data using the same function as bootstrap replicates, never the bootstrap sample mean.",
        "sensitivity": {"internal_original_rows": len(internal), "internal_clean_rows": int(clean.sum()), "removed_rows": int((~clean).sum()),
                        "legacy_original_rows": len(legacy), "legacy_prefix_clean_rows": int(legacy_clean.sum()), "legacy_removed_rows": int((~legacy_clean).sum()),
                        "limit": "Conservative removal using full train/validation pools and reconstructed model inputs. This does not repair training, selection contamination, certify all possible semantic leakage, or substitute for newly trained leakage-controlled experiments."},
        "alignment_limit": "Legacy CSV predictions contain no IDs or text; matching verified using source row order, all labels, row counts and archived metrics, not full independent record identity.",
        "method_reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
    }, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()

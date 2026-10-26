#!/usr/bin/env python3
"""Freeze model-visible separation and baseline search before new fitting.

Protect existing evaluation sets; quarantine, never reassign, conflicting
train/validation records. No model weights or GPU libraries are loaded.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from audit_revision_data import ROOT, OUT, NORM, NATURAL_SAMPLE, sha, dependency_groups

DATA = ROOT / "REVISION_PAPER/data/revision_v1"
STUDY = ROOT / "REVISION_PAPER/analysis/revision_v1"
KEYS = ["full_normalized", "normalized_500", "qwen_tokens512", "llama_tokens512"]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_snapshot() -> dict:
    files = sorted(p for p in (ROOT / "ORJ_PAPER").rglob("*") if p.is_file())
    files += sorted(p for p in (ROOT / "dataset").rglob("*") if p.is_file())
    files += [ROOT / "scripts/train_qlora.py", ROOT / "scripts/evaluate_adapter.py", ROOT / "experiment_archive/SHA256SUMS"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in files}


def assign_roles(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Quarantine conflicts with evaluation first, then validation (no labels)."""
    eval_groups = set(frame.loc[frame.original_role.isin(["internal_test", "legacy"]), "group"])
    val_groups = set(frame.loc[frame.original_role.eq("validation"), "group"]) - eval_groups
    role = frame.original_role.copy()
    reason = pd.Series("retained", index=frame.index)
    eval_conflict = frame.original_role.isin(["train", "validation"]) & frame.group.isin(eval_groups)
    val_conflict = frame.original_role.eq("train") & frame.group.isin(val_groups)
    role.loc[eval_conflict | val_conflict] = "quarantine"
    reason.loc[eval_conflict] = "component_touches_preserved_evaluation"
    reason.loc[val_conflict] = "component_touches_retained_validation"
    return role, reason


def search_grid() -> list[dict]:
    return [dict(zip(["ngram_max", "max_features", "min_df", "C", "class_weight"], values))
            for values in itertools.product([1, 2], [50000, 200000], [2, 5], [.25, 1., 4.], [None, "balanced"])]


def main():
    DATA.mkdir(parents=True, exist_ok=True); STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY / "protocol.json").exists():
        raise RuntimeError("Protocol already frozen; do not silently overwrite or tune on test outcomes")
    before = protected_snapshot()
    (STUDY / "source_integrity_before.json").write_text(json.dumps(before, indent=2) + "\n")
    old_run = json.loads((OUT / "audit_run.json").read_text())
    for name, expected in old_run["inputs_sha256"].items():
        assert file_sha(ROOT / name) == expected, f"Source changed since tokenizer audit: {name}"
    implementation = json.loads((OUT / "archived_implementation_details.json").read_text())
    for name, expected in implementation["source_sha256"].items():
        assert file_sha(ROOT / name) == expected, f"Prompt/training source changed: {name}"
    token_cache = pd.read_csv(OUT / "unique_input_keys.csv.gz").set_index("raw_500")
    originals = {name: pd.read_csv(ROOT / f"dataset/splits/{name}.csv") for name in ["train", "validation", "internal_test"]}
    legacy = pd.read_csv(ROOT / "dataset/splits/legacy_test_manifest.csv")
    pieces = []
    for name, f in [*originals.items(), ("legacy", legacy)]:
        f = f.copy()
        f["original_role"] = name; f["source_index"] = np.arange(len(f))
        f["true_label"] = f["product"] if "product" in f else f["true_label"]
        f["full_normalized"] = f.narrative.map(NORM).map(sha)
        assert (f.full_normalized == f.narrative_hash).all()
        f["normalized_500"] = f.narrative.astype(str).str[:500].map(NORM).map(sha)
        f["raw_500"] = f.narrative.astype(str).str[:500].map(sha)
        f["prefix250"] = f.narrative.astype(str).str[:250].map(NORM).map(sha)
        for key in ["qwen_tokens512", "llama_tokens512"]:
            f[key] = f.raw_500.map(token_cache[key])
            assert f[key].notna().all()
        pieces.append(f)
    all_rows = pd.concat(pieces, ignore_index=True)
    legacy_prefixes = set(all_rows.loc[all_rows.original_role.eq("legacy"), "prefix250"])
    all_rows["legacy_prefix_guard"] = [p if p in legacy_prefixes else f"unmatched:{i}" for i, p in enumerate(all_rows.prefix250)]
    all_rows["group"] = dependency_groups(all_rows, KEYS + ["legacy_prefix_guard"])
    all_rows["revision_role"], all_rows["reason"] = assign_roles(all_rows)
    manifest_cols = ["original_role", "source_index", "record_id", "source_file", "source_row", "true_label", "group", "revision_role", "reason"]
    all_rows[manifest_cols].to_csv(STUDY / "role_assignment_manifest.csv", index=False)
    all_rows[["original_role", "source_index", "group", *KEYS, "prefix250"]].to_csv(DATA / "audit_keys.csv.gz", index=False)
    sizes = []
    for name in ["train", "validation", "internal_test"]:
        selected = all_rows.loc[all_rows.revision_role.eq(name)]
        indices = selected.source_index.astype(int)
        f = originals[name].iloc[indices].copy()
        if name == "internal_test":
            shutil.copy2(ROOT / "dataset/splits/internal_test.csv", DATA / "internal_test.csv")
        else:
            f.to_csv(DATA / f"{name}.csv", index=False)
        f[["record_id", "product", "narrative_hash"]].assign(group=selected.group.to_numpy()).to_csv(STUDY / f"{name}_manifest.csv", index=False)
        sizes.append({"partition": name, "original_rows": len(originals[name]), "retained_rows": len(f),
                      "quarantined_rows": len(originals[name]) - len(f), "groups": selected.group.nunique(),
                      "class_counts": f["product"].value_counts().to_dict()})
    for name, budget in [("train", 10000), ("validation", 2000)]:
        f = pd.read_csv(DATA / f"{name}.csv")
        selected = NATURAL_SAMPLE(f, budget, 42)
        assert len(selected) == budget
        selected.to_csv(DATA / f"{name}_{budget}.csv", index=False)
        selected[["record_id", "product", "narrative_hash"]].to_csv(STUDY / f"{name}_{budget}_manifest.csv", index=False)
    (DATA / "legacy_subsets").mkdir(exist_ok=True)
    for path in sorted((ROOT / "dataset/legacy_subsets").glob("*.csv")):
        shutil.copy2(path, DATA / "legacy_subsets" / path.name)
    shutil.copy2(ROOT / "dataset/splits/legacy_test_manifest.csv", DATA / "legacy_test_manifest.csv")
    checks = []
    for a, b in [("train", "validation"), ("train", "internal_test"), ("validation", "internal_test"), ("train", "legacy"), ("validation", "legacy")]:
        left = all_rows.loc[all_rows.revision_role.eq(a)]; right = all_rows.loc[all_rows.revision_role.eq(b)]
        for key in ["group", *KEYS]:
            n = len(set(left[key]) & set(right[key]))
            checks.append({"left": a, "right": b, "representation": key, "shared": n})
            assert n == 0, (a, b, key, n)
    for name in ["train", "validation"]:
        assert not set(all_rows.loc[all_rows.revision_role.eq(name), "prefix250"]) & legacy_prefixes
    pd.DataFrame(checks).to_csv(STUDY / "separation_checks.csv", index=False)
    overlap = len(set(all_rows.loc[all_rows.revision_role.eq("internal_test"), "group"]) & set(all_rows.loc[all_rows.revision_role.eq("legacy"), "group"]))
    selected_source = [DATA / n for n in ["train.csv", "validation.csv", "internal_test.csv", "train_10000.csv", "validation_2000.csv", "legacy_test_manifest.csv"]]
    selected_source += sorted((DATA / "legacy_subsets").glob("*.csv"))
    protocol = {
        "version": "revision_v1", "frozen_date": "2026-10-20", "gpu_allowed": False,
        "evaluation_policy": "Original internal test and 12 legacy CSVs retained byte-for-byte. They were previously inspected; this is retrospective re-evaluation, not a newly unseen test.",
        "grouping": "Label-blind connected components of full normalized narrative, normalized first 500 raw characters, both archived-tokenizer 512-token inputs; additionally link any row sharing a normalized raw250 prefix with a legacy row. No empirical test score used in this rule.",
        "allocation_priority": "Preserve internal_test and legacy; quarantine train/validation components touching either; retain remaining validation and quarantine training components touching it. Never move an evaluation row into training.",
        "split_sizes": sizes, "internal_legacy_shared_components": overlap,
        "evaluation_dependence_limit": "Internal and legacy evaluation may share conservative groups; do not combine them as independent replications.",
        "fixed_selections": {"selection_seed": 42, "training_rows": 10000, "validation_rows": 2000,
                             "policy": "Same fixed selected rows for every model and repetition; distinguish optimisation seeds from data selection."},
        "tfidf_search": {"budgets": ["10000", "full"], "candidates_per_budget": len(search_grid()), "grid": search_grid(),
                         "validation": "Same frozen 2000 validation rows; maximise macro-F1. Ties within 1e-12: smallest candidate_id in preregistered enumeration.",
                         "preprocessing": "Only training texts fit vocabulary and IDF; validation/test only transformed. Narrative[:500], lowercase, Unicode accent stripping, sublinear_tf, L2-normalized TF-IDF.",
                         "solver": "lbfgs", "max_iter": 2000, "tol": 1e-5, "repetition_seeds": [42, 43, 44],
                         "determinism_note": "lbfgs ignores random_state; with fixed data these three seed-labelled refits are determinism checks, not independent stochastic runs. Report identical-score SD=0 transparently; do not manufacture training variability.",
                         "final_evaluation": "Write selected config after validation search for each budget, before opening any test/legacy CSV for prediction. No selection on final metrics."},
        "files_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in selected_source},
        "cached_tokenizer_scope": json.loads((OUT / "tokenizer_audit_evidence.json").read_text()),
        "limits": ["Existing original legacy exclusion of 8123 rows is retained; no excluded rows reintroduced.",
                   "Qwen input keys rely on an archived pilot tokenizer; final checkpoint identity is not independently certified.",
                   "These checks cover stated normalization/truncation/serialization, not arbitrary paraphrases or semantic duplication.",
                   "Archived models trained on the previous selections do not become corrected models merely by re-scoring."]}
    (STUDY / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (STUDY / "protocol.sha256").write_text(file_sha(STUDY / "protocol.json") + "  protocol.json\n")
    assert protected_snapshot() == before
    print(json.dumps({"sizes": sizes, "internal_legacy_shared_components": overlap, "checks": len(checks), "search_candidates_per_budget": len(search_grid())}, indent=2), flush=True)


if __name__ == "__main__":
    main()

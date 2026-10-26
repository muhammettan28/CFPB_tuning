#!/usr/bin/env python3
"""Read-only CPU audit of archived CFPB data and model-visible inputs.

Writes new analysis files only under REVISION_PAPER/analysis/cpu_revision.
No model weights are loaded. Original splits and archives are never rewritten.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "REVISION_PAPER/analysis/cpu_revision"
ARCHIVE = ROOT / "experiment_archive/outputs"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dependency_groups(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Connected components of shared identities, ignoring outcome labels."""
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in columns:
        first = {}
        for i, value in enumerate(frame[col]):
            if value in first:
                a, b = find(i), find(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = i
    return np.array([find(i) for i in range(len(frame))])


def repo_definitions(path: str, names: list[str]) -> dict:
    """Load only literal definitions/functions, without importing GPU modules."""
    tree = ast.parse((ROOT / path).read_text())
    selected = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name in names:
            selected.append(node)
    import unicodedata
    namespace = {"pd": pd, "unicodedata": unicodedata, "hashlib": hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)
    return {name: namespace[name] for name in names}


NORM = repo_definitions("scripts/prepare_splits.py", ["normalize_text"])["normalize_text"]
PROMPT = repo_definitions("scripts/evaluate_adapter.py", ["LABELS", "SYSTEM_PROMPT"])
NATURAL_SAMPLE = repo_definitions("scripts/train_qlora.py", ["natural_sample"])["natural_sample"]


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame["narrative"].isna().any():
        raise ValueError("Unexpected missing narrative in prepared data")
    if "product" in frame:
        frame["true_label"] = frame["product"]
    frame["full_normalized"] = frame.narrative.map(NORM).map(sha)
    if "narrative_hash" in frame:
        assert frame.narrative_hash.equals(frame.full_normalized)
    frame["raw_500"] = frame.narrative.astype(str).str[:500].map(sha)
    frame["normalized_500"] = frame.narrative.astype(str).str[:500].map(NORM).map(sha)
    return frame


def provenance(frames: dict[str, pd.DataFrame]) -> dict:
    raw = pd.read_csv(ROOT / "dataset/complaints.csv")
    missing = raw[["product", "narrative"]].isna().any(axis=1)
    usable = raw.loc[~missing].copy()
    usable["product"] = usable["product"].str.strip()
    usable["narrative"] = usable["narrative"].str.strip()
    index_col = "Unnamed: 0" if "Unnamed: 0" in raw else raw.columns[0]
    usable["record_id"] = usable[index_col] if index_col not in {"product", "narrative"} else usable.index
    excluded = pd.read_csv(ROOT / "dataset/splits/legacy_excluded_from_training.csv")
    pieces = pd.concat([frames[n] for n in ["train", "validation", "internal_test"]], ignore_index=True)
    assert pieces.record_id.is_unique and usable.record_id.is_unique
    assert set(pieces.record_id).isdisjoint(excluded.record_id)
    assert set(pieces.record_id) | set(excluded.record_id) == set(usable.record_id)
    joined = pieces.merge(usable[["record_id", "product", "narrative"]], on="record_id", suffixes=("", "_source"), validate="one_to_one")
    assert (joined["product"] == joined["product_source"]).all()
    assert (joined.narrative == joined.narrative_source).all()
    report = {
        "raw_rows": len(raw), "missing_required_rows": int(missing.sum()),
        "usable_rows": len(usable), "excluded_legacy_rows": len(excluded),
        "remaining_rows": len(pieces),
        "splits": {n: {"rows": len(frames[n]), "unique_full_narratives": frames[n].full_normalized.nunique()} for n in ["train", "validation", "internal_test"]},
        "split_rows_match_source_by_record_id_label_and_text": True,
        "upstream_limit": "complaints.csv already has five mapped labels and preprocessed narratives. The original CFPB extraction, preprocessing and original-product mapping cannot be recovered from this file alone.",
    }
    (OUT / "data_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame({n: f.true_label.value_counts() for n, f in frames.items() if n in ["train", "validation", "internal_test"]}).to_csv(OUT / "split_label_counts.csv")
    return report


def legacy_overlap(legacy: pd.DataFrame) -> dict:
    names = sorted(legacy.source_file.unique())
    sets = {
        "narrative": {n: set(legacy.loc[legacy.source_file.eq(n), "full_normalized"]) for n in names},
        "label_narrative": {n: set(zip(g.true_label, g.full_normalized)) for n in names for g in [legacy.loc[legacy.source_file.eq(n)]]},
    }
    for kind, groups in sets.items():
        counts = pd.DataFrame([[len(groups[a] & groups[b]) for b in names] for a in names], index=names, columns=names)
        counts.to_csv(OUT / f"legacy_overlap_{kind}_counts.csv")
        counts.div(pd.Series({n: len(groups[n]) for n in names}), axis=0).to_csv(OUT / f"legacy_overlap_{kind}_row_fraction.csv")
        union = pd.DataFrame([[len(groups[a] | groups[b]) for b in names] for a in names], index=names, columns=names)
        counts.div(union).to_csv(OUT / f"legacy_overlap_{kind}_jaccard.csv")
    # Conservative dependency groups: shared normalized raw 250-character
    # prefixes or full normalized text form one component. Links short/long versions,
    # they are different 500-character model inputs. It can over-group distinct
    # complaints with a common prefix, so also report exact-text groups.
    legacy["prefix250_key"] = legacy.narrative.astype(str).str[:250].map(NORM).map(sha)
    legacy["prefix250_group"] = dependency_groups(legacy, ["full_normalized", "prefix250_key"])
    report = {"rows": len(legacy), "unique_normalized_narratives": legacy.full_normalized.nunique(),
              "unique_label_narrative_pairs": len(legacy[["true_label", "full_normalized"]].drop_duplicates()),
              "conservative_prefix250_groups": legacy.prefix250_group.nunique(),
              "group_definition": "Connected components sharing full normalized text OR normalized first 250 raw characters; NFKC/lowercase/whitespace normalization, label excluded. Conservative dependency grouping, not verified complaint identity."}
    (OUT / "legacy_overlap_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_selections(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for seed in [42, 43, 44]:
        for role, budget in [("train", 10000), ("validation", 2000)]:
            selected = NATURAL_SAMPLE(frames[role], budget, seed)
            matches = list(ARCHIVE.glob(f"llama31_8b_fixed_protocol_seed{seed}_*/{role}_selection_manifest.csv"))
            if len(matches) != 1:
                raise ValueError(f"Expected one recorded Llama selection: {seed} {role}")
            recorded = pd.read_csv(matches[0])
            assert selected.record_id.tolist() == recorded.record_id.tolist()
            assert selected.narrative_hash.tolist() == recorded.narrative_hash.tolist()
            name = f"selected_{role}_seed{seed}"
            frames[name] = selected.copy()
            rows.append({"selection": name, "rows": len(selected), "llama_evidence": str(matches[0].relative_to(ROOT)),
                         "llama_manifest_matches_reconstruction": True,
                         "qwen_evidence": "Reconstructed from current natural_sample implementation, seed and Slurm protocol; final Qwen selection manifests absent. Not independently verified for archived final Qwen runs."})
    pd.DataFrame(rows).to_csv(OUT / "selection_manifest_validation.csv", index=False)
    return rows


def audit_short_prefixes(frames: dict[str, pd.DataFrame], legacy: pd.DataFrame) -> None:
    """Also test shared 250-character prefixes, not only equal visible inputs.

    A 250-character legacy text and a longer training text need not have equal
    500-character model inputs, yet can still derive from the same complaint.
    Prefix matches are conservative flags, not proof of complaint identity.
    """
    rows = []
    for name, frame in frames.items():
        left = frame.narrative.astype(str).str[:250].map(NORM)
        for source, group in legacy.groupby("source_file"):
            right = group.narrative.astype(str).str[:250].map(NORM)
            common = set(left) & set(right)
            rows.append({"left": name, "legacy_subset": source,
                         "shared_normalized_prefix250": len(common),
                         "left_rows_flagged": int(left.isin(common).sum()),
                         "legacy_rows_flagged": int(right.isin(common).sum())})
    pd.DataFrame(rows).to_csv(OUT / "legacy_short_prefix_audit.csv", index=False)


def token_keys(frames: dict[str, pd.DataFrame], legacy: pd.DataFrame) -> list[dict]:
    from transformers import AutoTokenizer, __version__
    texts = pd.concat([frames[n].narrative for n in ["train", "validation", "internal_test"]] + [legacy.narrative]).astype(str).str[:500].drop_duplicates().tolist()
    mapping = pd.DataFrame({"raw_500": [sha(t) for t in texts]})
    evidence = []
    model_paths = {
        "qwen": ARCHIVE / "qwen3_8b_qlora_pilot_5952287/best_adapter",
        "llama": ARCHIVE / "llama31_8b_fixed_protocol_seed42_5994012_0/best_adapter",
    }
    for model, path in model_paths.items():
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        token_hashes, prompt_hashes, counts = [], [], []
        for start in range(0, len(texts), 512):
            prompts = [tokenizer.apply_chat_template([
                {"role": "system", "content": PROMPT["SYSTEM_PROMPT"]},
                {"role": "user", "content": "Complaint:\n" + t}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for t in texts[start:start + 512]]
            # Reproduce evaluate_adapter.py exactly, including its default
            # add_special_tokens=True on the separate tokenizer call.
            ids = tokenizer(prompts, truncation=False, add_special_tokens=True)["input_ids"]
            counts.extend(map(len, ids))
            prompt_hashes.extend(sha(p) for p in prompts)
            for seq in ids:
                visible = seq[:512] if tokenizer.truncation_side == "right" else seq[-512:]
                token_hashes.append(sha(json.dumps(visible, separators=(",", ":"))))
        mapping[f"{model}_prompt"] = prompt_hashes
        mapping[f"{model}_tokens512"] = token_hashes
        mapping[f"{model}_untruncated_tokens"] = counts
        evidence.append({"model": model, "tokenizer_path": str(path.relative_to(ROOT)),
                         "transformers": __version__, "unique_raw500_inputs": len(texts),
                         "inputs_exceeding_512_tokens": int((np.asarray(counts) > 512).sum()),
                         "truncation_side": tokenizer.truncation_side,
                         "tokenizer_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}},
                         "coverage_limit": "Evaluation prompt serialization and 512-token IDs reproduced from archived tokenizer. Qwen tokenizer is from pilot, not final checkpoints; base-model tokenizer identity and exact SFT trainer completion/truncation are not independently certified."})
        print(f"Token audit {model}: {len(texts)} unique inputs, {evidence[-1]['inputs_exceeding_512_tokens']} over 512 tokens", flush=True)
    mapping.to_csv(OUT / "unique_input_keys.csv.gz", index=False)
    lookup = mapping.set_index("raw_500")
    for f in [*frames.values(), legacy]:
        for col in mapping.columns[1:]:
            f[col] = f.raw_500.map(lookup[col])
            assert f[col].notna().all()
    (OUT / "tokenizer_audit_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def audit_pairs(frames: dict[str, pd.DataFrame], legacy: pd.DataFrame) -> pd.DataFrame:
    frames = dict(frames)
    for name, g in legacy.groupby("source_file", sort=True):
        frames[name] = g
    pairs = [("train", "validation"), ("train", "internal_test"), ("validation", "internal_test")]
    for a in ["train", "validation", "internal_test"] + [n for n in frames if n.startswith("selected_")]:
        targets = ["internal_test"] if a.startswith("selected_") else []
        pairs.extend((a, b) for b in targets + sorted(legacy.source_file.unique()))
    columns = ["full_normalized", "raw_500", "normalized_500", "qwen_prompt", "qwen_tokens512", "llama_prompt", "llama_tokens512"]
    summary, hits = [], []
    for a, b in pairs:
        left, right = frames[a], frames[b]
        for col in columns:
            common = set(left[col]) & set(right[col])
            summary.append({"left": a, "right": b, "representation": col, "shared_unique_inputs": len(common),
                            "left_rows_affected": int(left[col].isin(common).sum()),
                            "right_rows_affected": int(right[col].isin(common).sum())})
            if common:
                for row in right.loc[right[col].isin(common)].itertuples():
                    hits.append({"left": a, "right": b, "representation": col,
                                 "right_record_id": getattr(row, "record_id", ""),
                                 "right_source_row": getattr(row, "source_row", ""),
                                 "input_hash": getattr(row, col), "true_label": row.true_label})
    result = pd.DataFrame(summary)
    result.to_csv(OUT / "leakage_overlap_summary.csv", index=False)
    pd.DataFrame(hits).to_csv(OUT / "leakage_affected_evaluation_rows.csv.gz", index=False)
    membership = []
    for name, f in frames.items():
        cols = [c for c in ["record_id", "source_row", "true_label", "full_normalized", "normalized_500", "raw_500", "qwen_tokens512", "llama_tokens512"] if c in f]
        membership.append(f[cols].assign(dataset=name))
    pd.concat(membership, ignore_index=True).to_csv(OUT / "input_membership.csv.gz", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {n: add_keys(pd.read_csv(ROOT / f"dataset/splits/{n}.csv")) for n in ["train", "validation", "internal_test"]}
    legacy = add_keys(pd.read_csv(ROOT / "dataset/splits/legacy_test_manifest.csv"))
    for name, g in legacy.groupby("source_file"):
        original = pd.read_csv(ROOT / "dataset/legacy_subsets" / name)
        assert g.narrative.tolist() == original.narrative.astype(str).str.strip().tolist()
        assert g.true_label.tolist() == original.true_label.tolist()
    prov = provenance(frames)
    overlap = legacy_overlap(legacy)
    selections = load_selections(frames)
    audit_short_prefixes(frames, legacy)
    tokens = token_keys(frames, legacy)
    result = audit_pairs(frames, legacy)
    legacy[["source_file", "source_row", "true_label", "full_normalized", "prefix250_group"]].to_csv(OUT / "legacy_groups.csv", index=False)
    inputs = [ROOT / "dataset/complaints.csv", ROOT / "dataset/splits/legacy_test_manifest.csv"] + [ROOT / f"dataset/splits/{n}.csv" for n in ["train", "validation", "internal_test"]]
    (OUT / "audit_run.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "provenance": prov, "legacy": overlap, "selections": selections, "tokenizers": tokens}, indent=2) + "\n")
    print(result.loc[result.right.eq("internal_test") & result.representation.isin(["raw_500", "normalized_500", "qwen_tokens512", "llama_tokens512"])].to_string(index=False))


if __name__ == "__main__":
    main()

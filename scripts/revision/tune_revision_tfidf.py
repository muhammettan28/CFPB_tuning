#!/usr/bin/env python3
"""Validation-only search and frozen-seed refits of corrected-data TF-IDF LR."""
from __future__ import annotations
from project_paths import project_path

import argparse
import gc
import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(name, "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from threadpoolctl import threadpool_limits

from prepare_revision_protocol import ROOT, DATA, STUDY, file_sha
from audit_revision_data import PROMPT

EXP = project_path(ROOT, "experiments/tfidf_revision_v1")
LABELS = PROMPT["LABELS"]


def score(y, prediction):
    return {"n": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, prediction, labels=LABELS, average="weighted", zero_division=0))}


def vectorizer(config):
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                           ngram_range=(1, config["ngram_max"]), min_df=config["min_df"],
                           max_features=config["max_features"], dtype=np.float64)


def fit_classifier(x, y, config, protocol, seed):
    model = LogisticRegression(C=config["C"], class_weight=config["class_weight"],
        solver=protocol["solver"], max_iter=protocol["max_iter"], tol=protocol["tol"], random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def assert_source_hashes(protocol):
    for name, expected in protocol["files_sha256"].items():
        assert file_sha(project_path(ROOT, name)) == expected, f"Frozen data changed: {name}"


def search(budget, protocol):
    out = EXP / f"budget_{budget}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selected_config.json").exists():
        print(f"Reusing frozen validation selection for budget {budget}", flush=True)
        return
    path = DATA / ("train_10000.csv" if budget == "10000" else "train.csv")
    train = pd.read_csv(path, usecols=["narrative", "product"])
    valid = pd.read_csv(DATA / "validation_2000.csv", usecols=["narrative", "product"])
    train_text = train.narrative.astype(str).str[:500]
    valid_text = valid.narrative.astype(str).str[:500]
    rows, best = [], None
    x_train = x_valid = vec = None
    last_features = None
    started = time.monotonic()
    settings = protocol["tfidf_search"]
    for i, config in enumerate(settings["grid"]):
        features = tuple(config[k] for k in ["ngram_max", "max_features", "min_df"])
        if features != last_features:
            del x_train, x_valid, vec
            gc.collect()
            vec = vectorizer(config)
            x_train = vec.fit_transform(train_text)  # training only
            x_valid = vec.transform(valid_text)
            last_features = features
        then = time.monotonic()
        model = fit_classifier(x_train, train["product"], config, settings, 42)
        prediction = model.predict(x_valid)
        metrics = score(valid["product"], prediction)
        row = {"candidate_id": i, **config, **metrics, "vocabulary_size": len(vec.vocabulary_),
               "iterations": int(model.n_iter_.max()), "fit_predict_seconds": time.monotonic() - then}
        rows.append(row)
        if best is None or row["macro_f1"] > best["validation_metrics"]["macro_f1"] + 1e-12:
            best = {"budget": budget, "candidate_id": i, "config": config,
                    "validation_metrics": metrics, "train_rows": len(train), "validation_rows": len(valid),
                    "protocol_sha256": file_sha(STUDY / "protocol.json"),
                    "selection_rule": settings["validation"],
                    "test_data_opened_during_search": False}
            joblib.dump(vec, out / "selected_vectorizer.joblib", compress=3)
        pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
        print(f"budget={budget} candidate={i+1}/{len(settings['grid'])} val_macro_f1={metrics['macro_f1']:.6f} fit_s={row['fit_predict_seconds']:.2f}", flush=True)
        del model
    best["search_elapsed_seconds"] = time.monotonic() - started
    (out / "selected_config.json").write_text(json.dumps(best, indent=2) + "\n")
    (out / "selected_config.sha256").write_text(file_sha(out / "selected_config.json") + "  selected_config.json\n")
    print(f"FROZEN budget={budget} best candidate={best['candidate_id']} val_macro_f1={best['validation_metrics']['macro_f1']:.6f}", flush=True)


def final_evaluation(budget, protocol):
    out = EXP / f"budget_{budget}"
    selection = json.loads((out / "selected_config.json").read_text())
    assert selection["protocol_sha256"] == file_sha(STUDY / "protocol.json")
    assert (out / "selected_config.sha256").read_text().split()[0] == file_sha(out / "selected_config.json")
    path = DATA / ("train_10000.csv" if budget == "10000" else "train.csv")
    train = pd.read_csv(path)
    stored_vec = joblib.load(out / "selected_vectorizer.joblib")
    vec = vectorizer(selection["config"])
    # Match search's numerical path exactly. fit_transform and transform on
    # training can differ at ~1e-16 and in CSR ordering, changing lbfgs stopping.
    x_train = vec.fit_transform(train.narrative.astype(str).str[:500])
    assert vec.vocabulary_ == stored_vec.vocabulary_
    np.testing.assert_array_equal(vec.idf_, stored_vec.idf_)
    valid = pd.read_csv(DATA / "validation_2000.csv")
    x_valid = vec.transform(valid.narrative.astype(str).str[:500])
    # Final examples are only loaded after the configuration is frozen.
    internal = pd.read_csv(DATA / "internal_test.csv")
    legacy = pd.read_csv(DATA / "legacy_test_manifest.csv")
    x_internal = vec.transform(internal.narrative.astype(str).str[:500])
    x_legacy = vec.transform(legacy.narrative.astype(str).str[:500])
    results, pred_hashes = [], []
    for seed in protocol["tfidf_search"]["repetition_seeds"]:
        dest = out / f"seed_{seed}"; dest.mkdir(exist_ok=True)
        then = time.monotonic()
        model = fit_classifier(x_train, train["product"], selection["config"], protocol["tfidf_search"], seed)
        actual_validation = score(valid["product"], model.predict(x_valid))
        assert actual_validation == selection["validation_metrics"], (budget, seed, actual_validation)
        joblib.dump(model, dest / "classifier.joblib", compress=3)
        hashes = {}
        for role, frame, x in [("internal_test", internal, x_internal), ("legacy", legacy, x_legacy)]:
            prediction = model.predict(x)
            result = score(frame["product"] if role == "internal_test" else frame.true_label, prediction)
            result.update(budget=budget, seed=seed, role=role)
            results.append(result)
            columns = ["record_id", "product"] if role == "internal_test" else ["source_file", "source_row", "true_label"]
            predictions = frame[columns].rename(columns={"product": "true_label"}).copy()
            predictions["prediction"] = prediction
            predictions.to_csv(dest / f"{role}_predictions.csv.gz", index=False)
            hashes[role] = hashlib.sha256("\n".join(prediction).encode()).hexdigest()
            (dest / f"{role}_metrics.json").write_text(json.dumps({**result,
                "classification_report": classification_report(predictions.true_label, prediction, labels=LABELS, output_dict=True, zero_division=0)}, indent=2) + "\n")
        pred_hashes.append({"seed": seed, **hashes})
        print(f"FINAL budget={budget} seed={seed} internal_macro_f1={results[-2]['macro_f1']:.6f} elapsed_s={time.monotonic()-then:.2f}", flush=True)
    pd.DataFrame(results).to_csv(out / "final_metrics.csv", index=False)
    identical = all(len({row[role] for row in pred_hashes}) == 1 for role in ["internal_test", "legacy"])
    (out / "repetition_checks.json").write_text(json.dumps({
        "prediction_hashes": pred_hashes, "identical_predictions_all_three_seeds": identical,
        "interpretation": protocol["tfidf_search"]["determinism_note"]}, indent=2) + "\n")
    if not identical:
        raise RuntimeError("Unexpected lbfgs determinism difference; investigate before reporting SD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "evaluate", "all"], default="all")
    args = parser.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert_source_hashes(protocol)
    (EXP / "environment.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "cpu_threads_limit": 4,
        "gpu_used": False, "script_sha256": file_sha(Path(__file__))}, indent=2) + "\n")
    # Freeze BOTH budgets using validation only before any final predictions.
    if args.stage in ["search", "all"]:
        for budget in protocol["tfidf_search"]["budgets"]:
            search(budget, protocol)
    if args.stage in ["evaluate", "all"]:
        for budget in protocol["tfidf_search"]["budgets"]:
            assert (EXP / f"budget_{budget}/selected_config.json").exists()
        for budget in protocol["tfidf_search"]["budgets"]:
            final_evaluation(budget, protocol)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()

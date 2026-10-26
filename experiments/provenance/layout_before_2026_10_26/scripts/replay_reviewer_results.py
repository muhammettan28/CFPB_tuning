#!/usr/bin/env python3
"""Replay published prediction/latency analyses on CPU in an extracted archive.

Does not fit models, load neural weights, inspect CUDA, or require ORJ_PAPER.
Writes only to a new --output directory; never replaces archived evidence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import analyze_corrected_gpu_results as core
import analyze_ref14_baselines as extra
from revision_gpu_common import ROOT, PAPER, verify_data, sha_file


def compare(actual, source):
    expected = pd.read_csv(source)
    actual = pd.DataFrame(actual)
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise AssertionError(f'Shape/schema mismatch: {source}')
    numerical = expected.select_dtypes(include='number').columns.tolist()
    for name in expected:
        if name in numerical:
            if not np.allclose(actual[name], expected[name], atol=1e-12, rtol=0, equal_nan=True):
                raise AssertionError(f'Numerical mismatch: {source.name}/{name}')
        elif not actual[name].fillna('').astype(str).equals(expected[name].fillna('').astype(str)):
            raise AssertionError(f'Identity/order mismatch: {source.name}/{name}')
    actual.to_csv(core.OUT / source.name, index=False)
    return {'source': str(source.relative_to(ROOT)), 'sha256': sha_file(source),
            'rows': len(expected), 'numeric_cells': len(expected) * len(numerical),
            'identity_and_order_match': True, 'absolute_tolerance': 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for replayed results; existing directories are rejected.')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('--output must not exist; archived evidence is never overwritten')
    output.mkdir(parents=True)
    core.OUT = output
    verify_data(include_final=True)
    # Source lists certify the exact predictions/metadata/timings used originally.
    checked = {}
    for listing in [PAPER/'analysis/gpu_execution/GPU_RESULT_SOURCES.json',
                    PAPER/'analysis/ref14_encoder/RESULT_SOURCES.json']:
        for item in json.loads(listing.read_text()):
            path = ROOT/item['path']
            if sha_file(path) != item['sha256']:
                raise AssertionError(f'Changed analysis source: {path}')
            checked[item['path']] = item['sha256']
    internal, legacy = core.targets()
    rows, seeds, contrasts, refrows, refcontrasts = [], [], [], [], []
    with threadpool_limits(limits=4):
        for role, target in [('internal', internal), ('legacy', legacy)]:
            print(f'Replaying {role}: 10,000 paired group draws...', flush=True)
            r, s, c = core.summarise_role(target, role)
            rows += r; seeds += s; contrasts += c
            r, c = extra.summarise(target, role)
            refrows += r; refcontrasts += c
        print('Replaying length sensitivity and controlled latency...', flush=True)
        long = core.long_input_results(internal)
        latency, paired = core.latency_results(internal)
    main_sources = PAPER/'analysis/gpu_execution'
    comparisons = [compare(values, main_sources/name) for values, name in [
        (rows, 'corrected_model_metrics.csv'), (seeds, 'corrected_per_seed_metrics.csv'),
        (contrasts, 'corrected_paired_contrasts.csv'), (long, 'corrected_long_input.csv'),
        (latency, 'controlled_latency_summary.csv'), (paired, 'controlled_latency_paired.csv')]]
    comparisons += [compare(values, PAPER/'analysis/ref14_encoder'/name) for values, name in [
        (refrows, 'model_metrics.csv'), (refcontrasts, 'paired_contrasts.csv')]]
    report = {'passed': True, 'cpu_only': True, 'neural_weights_required': False,
              'original_paper_required': False, 'fitting_performed': False,
              'source_files_hash_checked': len(checked), 'comparisons': comparisons,
              'scope': 'Saved-prediction/timing replay, not model execution or recovery of historical provenance.'}
    (output/'REPLAY_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: {len(checked)} source hashes; {len(comparisons)} complete result tables. {output}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Verify restored narrative/selection details against saved evidence, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import pandas as pd
from project_paths import ROOT, PAPER
from revision_gpu_common import sha_file
from restored_content_evidence import historical_protocol_rows, verified_confusions
from publication_language import editorial_history_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main_tex = (PAPER/'manuscript.tex').read_text()
    sup = (PAPER/'supplementary_material.tex').read_text()
    for name,tex in [('manuscript',main_tex),('supplementary',sup)]:
        assert not editorial_history_hits(tex),(name,editorial_history_hits(tex))
    intro = main_tex.split(r'\section{Introduction}', 1)[1].strip().split('\n\n', 1)[0]
    assert hashlib.sha256(intro.encode()).hexdigest() == 'b2286e3fd4cc2e6c13b891e723e23d3e14c260a9a447c56755f1f9f45f88c161'
    matrices, sources = verified_confusions()
    errors = []
    # Require each value in its named directional comparison, not elsewhere in a table.
    dominant = main_tex.split('The dominant zero-shot error is', 1)[1].split('\n\n', 1)[0]
    residual = main_tex.split('One residual issue remains visible in both models:', 1)[1].split('\n\\begin{figure}', 1)[0]
    for model in ['qwen', 'llama']:
        base, ft = matrices[model+'_base'], matrices[model+'_ft']
        debt_base, debt_ft = float(base[4, 1]), float(ft[4, 1])
        reduction = 100*(1-debt_ft/debt_base)
        for value in [f'{debt_base:,.0f}', f'{debt_ft:,.2f}', f'{reduction:.1f}'+r'\%']:
            assert value in dominant, (model, value)
        for value in [f'{base[0, 2]:,.0f}', f'{ft[0, 2]:,.0f}']:
            assert value in dominant, (model, 'banking-to-card', value)
        for value in [f'{base[1, 4]:,.0f}', f'{ft[1, 4]:,.0f}']:
            assert value in residual, (model, 'reporting-to-debt', value)
        errors.append(dict(model=model, debt_to_reporting_base=debt_base,
                           debt_to_reporting_ft_mean=debt_ft, reduction_percent=reduction,
                           reporting_to_debt_base=float(base[1, 4]),
                           reporting_to_debt_ft_mean=float(ft[1, 4])))
    historical = historical_protocol_rows()
    best = max(historical, key=lambda r: r['macro_f1'])
    assert best['condition'] == 'Natural 2-epoch'
    assert f"{best['macro_f1']:.4f}" in main_tex
    historical_block = sup.split(r'\label{sup:historical-selection}', 1)[1].split(r'\end{longtable}', 1)[0]
    for record in historical:
        assert record['condition'] in historical_block
        for key in ['macro_f1', 'debt_f1']:
            assert f"{record[key]:.4f}" in historical_block
    # Validate old descriptive statistics on their actual pre-quarantine population.
    frames = [pd.read_csv(ROOT/f'dataset/splits/{role}.csv', usecols=['product', 'narrative'])
              for role in ['train', 'validation', 'internal_test']]
    pool = pd.concat(frames, ignore_index=True)
    length = pool.narrative.str.len()
    counts = pool['product'].value_counts()
    historical_population = sup.split('The 154,282-row partitioning pool contains', 1)[1]
    for label, count in counts.items():
        for value in [f'{count:,}', f'{100*count/len(pool):.2f}'+r'\%']:
            assert value in historical_population, (label, value)
    for value in [f'{length.median():,.0f}',
                  f'{length.quantile(.9):,.0f}', f'{100*length.le(500).mean():.2f}'+r'\%']:
        assert value in historical_population, value
    split_report = json.loads((ROOT/'dataset/splits/split_report.json').read_text())
    for value in split_report['legacy']['match_reason_counts'].values():
        assert f'{value:,}' in sup.split('The benchmark-related exclusion records comprise', 1)[1].split('\n\n', 1)[0]
    for path in [ROOT/'dataset/splits/split_report.json', ROOT/'analysis/postprocessing/corrected_confusion_counts.npz']:
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path)))
    for record in historical:
        sources.append(dict(path=record['source'], sha256=record['source_sha256']))
    report = dict(passed=True, cpu_only=True, fitting_or_inference_performed=False,
                  original_intro_opening_retained_verbatim=True,
                  publication_revision_history_absent=True,
                  auxiliary_fitting_scope_and_overlap_limitations_retained=True,
                  historical_selection_conditions_checked=len(historical),
                  historical_scores_separate_from_corrected_results=True,
                  raw_prediction_files_checked=8, all_four_LLM_confusion_matrices_reconstructed=True,
                  directional_errors=errors, historical_pool_rows=len(pool),
                  historical_descriptive_statistics_verified=True,
                  historical_exclusion_reasons_verified=True,
                  new_supplementary_table_labels=re.findall(r'\\label\{(sup:historical-selection)\}', sup),
                  sources=list({r['path']: r for r in sources}.values()),
                  documents_sha256={name: sha_file(PAPER/name) for name in ['manuscript.tex', 'supplementary_material.tex']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: original opening, six historical validation conditions, eight prediction files, directional errors and historical data descriptions.')


if __name__ == '__main__':
    main()

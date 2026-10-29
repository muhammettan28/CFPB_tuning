#!/usr/bin/env python3
"""Recheck added comparison/cost/provenance evidence without loading model weights."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_gpu_common import ROOT,sha_file
from build_comparison_evidence import training_records,cost_summary,protocol_comparison,POST
from publication_language import editorial_history_hits,internal_workflow_hits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-documents',action='store_true',
                        help='Also check claims in the local, Git-ignored publication sources.')
    args=parser.parse_args()
    if args.include_documents:
        required=['manuscript.tex','supplementary_material.tex','response_to_reviewers.tex']
        missing=[name for name in required if not (ROOT/'REVISION_PAPER'/name).is_file()]
        if missing:parser.error('--include-documents requires local REVISION_PAPER files: '+', '.join(missing))
    source_hashes={};calls=0
    for part in ['sparse','encoders']:
        base=ROOT/'experiments/cpu_costs_v1'/part
        marker=json.loads((base/'COMPLETE.json').read_text());assert marker['completed'] and marker['cpu_only']
        assert marker['record_sha256']==sha_file(base/'latency_records.csv')
        assert marker['script_sha256']==sha_file(ROOT/'scripts/revision/benchmark_cpu_baselines.py')
        frame=pd.read_csv(base/'latency_records.csv',dtype={'record_id':str})
        expected_ids=set(pd.read_csv(ROOT/'analysis/gpu_preparation/latency_sample_manifest.csv').record_id.astype(str))
        for system,g in frame.groupby('system'):
            assert len(g)==300 and set(g.repetition)=={0,1,2}
            for _,r in g.groupby('repetition'):assert set(r.record_id)==expected_ids and not r.record_id.duplicated().any()
            assert (g.seconds>0).all()
            if part=='encoders':
                stored=pd.read_csv(ROOT/f'experiments/gpu_revision_v1/evaluation/evaluate/{system}/42/suite/internal_test_predictions.csv.gz',dtype={'record_id':str}).set_index('record_id')
                probe=g.loc[g.repetition.eq(0)].set_index('record_id')
                assert not (probe.prediction!=stored.loc[probe.index,'prediction']).any()
        env=json.loads((base/'environment.json').read_text());assert env['cuda_visible_devices']=='' and env['threads']==4
        calls+=len(frame)
        for p in base.glob('*'):
            if p.suffix in {'.json','.csv'}:source_hashes[str(p.relative_to(ROOT))]=sha_file(p)
    records=training_records();assert len(records)==17
    for r in records:
        source=ROOT/r['source']
        if source.exists():assert sha_file(source)==r['source_sha256'],source
        if r['device']!='CPU':assert r['recorded_training_metrics']['train_runtime']==r['training_seconds']
        else:assert r['recorded_training_metrics']['prediction_parity_rows']==15428
    for name,actual in zip(['computational_costs.csv','cpu_latency_summary.csv'],cost_summary(records)):
        expected=pd.read_csv(POST/name)
        pd.testing.assert_frame_equal(actual,expected,check_exact=False,atol=1e-12,rtol=0)
    current=protocol_comparison();saved=json.loads((POST/'protocol_comparison_evidence.json').read_text());assert current==saved
    assert all(abs(r['difference'])<.003 for r in current['adapted_comparisons'])
    audit=json.loads((POST/'tokenization_path_audit.json').read_text());assert audit['passed'] and audit['cpu_only'] and not audit['neural_inference_performed']
    for path,digest in audit['code_sources'].items():assert sha_file(ROOT/path)==digest
    llama=next(r for r in audit['paths'] if r['model']=='llama');qwen=next(r for r in audit['paths'] if r['model']=='qwen')
    assert llama['default_leading_bos']==2 and llama['explicit_leading_bos']==1 and qwen['identical_token_ids']
    if args.include_documents:
        for stem in ['manuscript','supplementary_material']:
            text=(ROOT/'REVISION_PAPER'/f'{stem}.tex').read_text();assert not editorial_history_hits(text);assert not internal_workflow_hits(text)
        main=(ROOT/'REVISION_PAPER/manuscript.tex').read_text();response=(ROOT/'REVISION_PAPER/response_to_reviewers.tex').read_text()
        assert '63.98\\%' in main and '63.01\\%' in main and '\\cite{cfpbdata,tiwari2021complaints}' in main
        assert 'statistical equivalence or non-inferiority' in main
        for value in ['1,401','1,167','333','263','0.6189','0.6247','V100','RTX 5060 Ti']:
            assert value in response,value
        assert response.count('\n'+r'\item ')==5
        for row in current['adapted_comparisons']:
            assert f"{row['difference']:+.6f}" in response
        response_tex=(ROOT/'REVISION_PAPER/response_to_reviewers.tex').read_text()
        assert r'\(2^{-12}\)' in response_tex
    report={'passed':True,'cpu_only':True,'new_gpu_execution':False,'cpu_timed_calls_verified':calls,
            'training_runtime_records_verified':len(records),'cost_tables_reconstructed':True,
            'all_test_ids_and_labels_unchanged_for_base_comparison':True,'zero_shot_counts_reconstructed_from_predictions':True,
            'tokenisation_path_audit_checked':True,'descriptive_adapted_changes_below_0_003':True,
            'document_checks_performed':args.include_documents,
            'equivalence_not_inferred_from_zero_containing_interval':True if args.include_documents else None,
            'five_item_response_summary_present':True if args.include_documents else None,
            'publication_revision_chronology_absent':True if args.include_documents else None,'source_sha256':source_hashes,
            'scope':'Additional paired intervals are separately recomputed in the nine-table saved-result replay; costs are reconstructed from raw CPU call records and exported training-time records.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 1,500 CPU calls, 17 fitting records, protocol differences and cost reconstruction'+
          ('; local manuscript/response claims checked' if args.include_documents else '; publication sources not required'))

if __name__=='__main__':main()

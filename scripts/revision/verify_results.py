#!/usr/bin/env python3
"""Recheck added comparison/cost/provenance evidence without loading model weights."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_gpu_common import ROOT,sha_file
from build_comparison_evidence import training_records,cost_summary,protocol_comparison,POST


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
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
    report={'passed':True,'cpu_only':True,'cpu_timed_calls_verified':calls,
            'training_runtime_records_verified':len(records),'cost_tables_reconstructed':True,
            'tokenisation_path_audit_checked':True,'source_sha256':source_hashes}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 1,500 CPU calls, 17 fitting records, protocol differences and cost reconstruction')

if __name__=='__main__':main()

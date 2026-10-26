#!/usr/bin/env python3
"""Assemble added comparisons, provenance and cost summaries from recorded evidence."""
import argparse, ast, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from revision_gpu_common import ROOT, PAPER, EXP, DATA, LABELS, load_frame, sha_file
from comparison_extensions import encoder_contrasts

POST = ROOT/'analysis/postprocessing'


def training_records():
    exported = POST/'training_runtime_evidence.json'
    if exported.exists():
        return json.loads(exported.read_text())
    records = []
    for model in ['qwen','llama','bert','distilbert','roberta']:
        for seed in [42,43,44]:
            if model in ['qwen','llama']:
                source = EXP/'qlora'/model/f'seed_{seed}'/'checkpoints/trainer_state.json'
                log = json.loads(source.read_text())['log_history']
                metrics = next(r for r in log if 'train_runtime' in r)
                system = model+'_ft'
            else:
                source = ROOT/'analysis/gpu_execution/logs'/f'encoder-refit__{model}__{seed}__main.log'
                candidates = re.findall(r"\{[^{}\n]*['\"]train_runtime['\"][^{}\n]*\}",source.read_text())
                assert len(candidates) == 1,source
                metrics = ast.literal_eval(candidates[0]); system = model
            records.append(dict(system=system,seed=seed,train_rows=10000,device='RTX 5060 Ti',
                                training_seconds=metrics['train_runtime'],source=str(source.relative_to(ROOT)),
                                source_sha256=sha_file(source),recorded_training_metrics=metrics))
    for budget in ['10000','full']:
        source = ROOT/'experiments/cpu_costs_v1/sparse/training_records.csv'
        row = pd.read_csv(source).set_index('system').loc['tfidf_'+budget]
        records.append(dict(system='tfidf_'+budget,seed=42,train_rows=int(row.train_rows),device='CPU',
                            training_seconds=float(row.training_seconds),source=str(source.relative_to(ROOT)),
                            source_sha256=sha_file(source),recorded_training_metrics={'scope':row.scope,'prediction_parity_rows':int(row.prediction_parity_rows)}))
    return records


def encoder_search_records():
    exported = POST/'training_search_runtime_evidence.json'
    if exported.exists():
        return json.loads(exported.read_text())
    records = {}
    for model in ['bert', 'distilbert', 'roberta']:
        source = ROOT/'analysis/gpu_execution/logs'/f'encoder-search__{model}__base__main.log'
        matches = re.findall(r"\{[^{}\n]*['\"]train_runtime['\"][^{}\n]*\}", source.read_text())
        assert len(matches) == 6, source
        records[model] = {'source':str(source.relative_to(ROOT)), 'source_sha256':sha_file(source),
                          'candidate_training_records':[ast.literal_eval(m) for m in matches]}
    exported.write_text(json.dumps(records, indent=2)+'\n')
    return records


def protocol_comparison():
    listing = json.loads((ROOT/'analysis/cpu_revision/prediction_sources.json').read_text())
    target = load_frame(DATA/'internal_test.csv'); ids = target.record_id.astype(str).tolist(); truth = target['product'].tolist()
    rows = []
    for model,title in [('qwen','Qwen3-8B'),('llama','Llama-3.1-8B')]:
        item = next(r for r in listing if r['model']==title and r['role']=='internal' and r['variant']=='base_zero_shot')
        source = ROOT/item['path']; assert sha_file(source) == item['sha256']
        old = pd.read_csv(source,dtype={'record_id':str},keep_default_na=False).set_index('record_id').loc[ids]
        new_source = EXP/'evaluation/evaluate'/model/'base/suite/internal_test_predictions.csv.gz'
        new = pd.read_csv(new_source,dtype={'record_id':str},keep_default_na=False).set_index('record_id').loc[ids]
        assert old.true_label.tolist()==new.true_label.tolist()==truth
        cm_old = confusion_matrix(truth,old.prediction,labels=LABELS+['__off_label__'])
        cm_new = confusion_matrix(truth,new.prediction,labels=LABELS+['__off_label__'])
        rows.append(dict(model=model,rows=len(ids),identical_record_ids_and_labels=True,
                         auxiliary_base_macro_f1=f1_score(truth,old.prediction,labels=LABELS,average='macro',zero_division=0),
                         primary_base_macro_f1=f1_score(truth,new.prediction,labels=LABELS,average='macro',zero_division=0),
                         differing_predictions=int((old.prediction!=new.prediction).sum()),
                         auxiliary_debt_to_reporting=int(cm_old[4,1]),primary_debt_to_reporting=int(cm_new[4,1]),
                         auxiliary_banking_to_card=int(cm_old[0,2]),primary_banking_to_card=int(cm_new[0,2]),
                         sources=[{'path':str(p.relative_to(ROOT)),'sha256':sha_file(p)} for p in [source,new_source]]))
    historical = pd.read_csv(ROOT/'analysis/cpu_revision/paired_group_bootstrap.csv')
    primary = pd.read_csv(ROOT/'analysis/gpu_execution/corrected_model_metrics.csv')
    adapted = []
    for model,title in [('qwen','Qwen3-8B'),('llama','Llama-3.1-8B')]:
        old = historical.loc[historical.analysis.eq('internal_original')&historical.model.eq(title)&historical.metric.eq('macro_f1')].iloc[0]
        new = primary.loc[primary.role.eq('internal')&primary.system.eq(model+'_ft')&primary.metric.eq('macro_f1')].iloc[0]
        adapted.append(dict(model=model,auxiliary_point=old.ft_seed_mean_point,primary_point=new.point,difference=new.point-old.ft_seed_mean_point))
    filtered = historical.loc[historical.analysis.eq('internal_conservative_clean')&historical.metric.eq('macro_f1')].to_dict('records')
    return dict(base_comparisons=rows,adapted_comparisons=adapted,filtered_auxiliary_results=filtered,
                interpretation='Descriptive cross-implementation differences; membership, serialization, artifact identities and hardware are not held fixed. This is not a causal overlap-removal ablation.')


def cost_summary(records):
    search_records = encoder_search_records()
    frames = [pd.read_csv(ROOT/f'experiments/cpu_costs_v1/{part}/latency_records.csv') for part in ['sparse','encoders']]
    cpu = pd.concat(frames,ignore_index=True); summaries = []; cost = []
    latency = pd.read_csv(ROOT/'analysis/gpu_execution/controlled_latency_summary.csv',dtype={'variant':str})
    for system in ['qwen_ft','llama_ft','bert','distilbert','roberta','tfidf_10000','tfidf_full']:
        fits = [r for r in records if r['system']==system]; fit_seconds = np.mean([r['training_seconds'] for r in fits])
        if system.endswith('_ft'):
            g = latency.loc[latency.model.eq(system.split('_')[0])&latency.variant.ne('base')&latency.condition.eq('chars500_tokens512')&latency.timing.eq('end_to_end_seconds')]
            seconds = g['mean'].mean(); inference_device = 'RTX 5060 Ti'; search_seconds = None
        else:
            g = cpu.loc[cpu.system.eq(system)]; seconds = g.seconds.mean(); inference_device = 'CPU'
            summaries.append(dict(system=system,calls=len(g),mean_ms=1000*seconds,median_ms=1000*g.seconds.median(),p95_ms=1000*g.seconds.quantile(.95)))
            if system.startswith('tfidf_'):
                budget=system.removeprefix('tfidf_');selection=json.loads((ROOT/f'experiments/tfidf_revision_v1/budget_{budget}/selected_config.json').read_text());search_seconds=selection['search_elapsed_seconds']
            else:
                search_seconds=sum(r['train_runtime'] for r in search_records[system]['candidate_training_records'])
        cost.append(dict(system=system,train_rows=fits[0]['train_rows'],fit_runs=len(fits),training_device=fits[0]['device'],mean_fit_seconds=fit_seconds,
                         mean_fit_minutes=fit_seconds/60,mean_fit_device_hours=fit_seconds/3600,search_seconds=search_seconds,
                         inference_device=inference_device,mean_e2e_ms=1000*seconds))
    return pd.DataFrame(cost),pd.DataFrame(summaries)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap-directory',type=Path,required=True);args=p.parse_args()
    encoder_contrasts(args.bootstrap_directory).to_csv(POST/'qlora_encoder_contrasts.csv',index=False)
    records=training_records();(POST/'training_runtime_evidence.json').write_text(json.dumps(records,indent=2)+'\n')
    (POST/'protocol_comparison_evidence.json').write_text(json.dumps(protocol_comparison(),indent=2)+'\n')
    costs,latency=cost_summary(records);costs.to_csv(POST/'computational_costs.csv',index=False);latency.to_csv(POST/'cpu_latency_summary.csv',index=False)
    sources=json.loads((POST/'DOCUMENT_NUMERICAL_SOURCES.json').read_text())
    for name in ['qlora_encoder_contrasts.csv','computational_costs.csv','cpu_latency_summary.csv','training_runtime_evidence.json','training_search_runtime_evidence.json','protocol_comparison_evidence.json','tokenization_path_audit.json','cpu_inference_precision_check.json']:
        path=POST/name;sources=[r for r in sources if r['path']!=str(path.relative_to(ROOT))];sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha_file(path)))
    (POST/'DOCUMENT_NUMERICAL_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(costs.to_string(index=False));print('PASS: additional contrasts, training excerpts, protocol comparison and CPU cost evidence generated')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Validate all corrected-data GPU outputs; generate paired uncertainty and reports.

CPU only. Runs after all 88 GPU jobs, never selects models using final metrics.
"""
from __future__ import annotations
from project_paths import project_path
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from revision_gpu_common import (CONFIG,DATA,EXP,LABELS,PAPER,PREP,PROTOCOL_SHA,ROOT,
                                 json_text,load_frame,load_plan,pick_candidate,sha_file,verify_data,versions)
from analyze_archived_predictions import encode,group_confusions,paired_cluster_bootstrap,confusion_metrics,METRICS

OUT=project_path(PAPER, 'analysis/gpu_execution')
REPLICATES=10000
RNG_SEED=20261020
SOURCES=[]


def completed(path):
    done=json.loads((path/'COMPLETE.json').read_text())
    if not done['completed']:raise RuntimeError(f'Incomplete job: {path}')
    meta=json.loads((path/'run_metadata.json').read_text())
    if meta['protocol_sha256']!=PROTOCOL_SHA or meta['plan_sha256']!=sha_file(CONFIG):
        raise RuntimeError(f'Wrong protocol/plan: {path}')
    return meta


def remember(path,kind):
    SOURCES.append({'path':str(path.relative_to(ROOT)),'sha256':sha_file(path),'kind':kind})


def validate_training_job(job):
    path=Path(job['out']);meta=completed(path)
    if meta['arguments']['action']!=job['action'] or meta['arguments']['model']!=job['model']:
        raise RuntimeError('Job identity differs from metadata')
    if job['action'] not in {'smoke','qlora','encoder-search','encoder-refit'}:return
    train=load_frame(DATA/'train_10000.csv');val=load_frame(DATA/'validation_2000.csv')
    if job['action']=='smoke':train,val=train.head(32),val.head(8)
    if meta['actual_train_rows']!=len(train) or meta['actual_validation_rows']!=len(val):
        raise RuntimeError('Actual fitting budget differs from frozen selected data')
    columns=['record_id','product','narrative_hash']
    for role,frame,source in [('train',train,'train_10000.csv'),('validation',val,'validation_2000.csv')]:
        manifest=pd.read_csv(path/f'{role}_selection_manifest.csv',dtype=str)
        if not manifest[columns].equals(frame[columns].astype(str).reset_index(drop=True)):
            raise RuntimeError('Fitting selection order/IDs/labels/narrative hashes differ')
        if meta[f'{role}_csv_sha256']!=sha_file(DATA/source):raise RuntimeError('Fitting data hash differs')
    if job['action']=='encoder-search':
        search_path=path/'validation_search.json';selected_path=path/'selected_config.json'
        rows=json.loads(search_path.read_text());selected=json.loads(selected_path.read_text())
        grid=load_plan()['encoder']['grid']
        if len(rows)!=6 or {r['candidate_id'] for r in rows}!={c['candidate_id'] for c in grid}:
            raise RuntimeError('Encoder search is incomplete')
        if sha_file(selected_path)!=selected_path.with_suffix('.sha256').read_text().split()[0] or sha_file(search_path)!=selected['validation_search_sha256']:
            raise RuntimeError('Encoder selection signatures differ')
        for row in rows:
            candidate=next(c for c in grid if c['candidate_id']==row['candidate_id'])
            if any(row[k]!=v for k,v in candidate.items()):raise RuntimeError('Encoder grid differs')
            source=path/f'candidate_{row["candidate_id"]:02d}'/'validation_predictions.csv'
            pred=pd.read_csv(source,dtype={'record_id':str}).rename(columns={'product':'true_label'})
            aligned=align_predictions(pred,val.rename(columns={'product':'true_label'}),'internal')
            actual=check_metrics(aligned.true_label,aligned.prediction,row['validation_metrics'])
            if abs(actual[1]-row['validation_macro_f1'])>1e-12:raise RuntimeError('Encoder search score differs')
            remember(source,'encoder_validation_search')
        if pick_candidate(rows)['candidate_id']!=selected['candidate']['candidate_id']:
            raise RuntimeError('Encoder selected candidate violates validation rule')
        remember(search_path,'encoder_validation_search');remember(selected_path,'encoder_selected_configuration')


def align_predictions(pred,target,role):
    if role=='internal':
        pred=pred.copy();pred['record_id']=pred.record_id.astype(str)
        ids=target.record_id.astype(str).tolist()
        if pred.record_id.duplicated().any() or set(pred.record_id)!=set(ids):
            raise RuntimeError('Internal prediction record IDs differ')
        pred=pred.set_index('record_id').loc[ids].reset_index()
    else:
        pred=pred.copy();pred['source_row']=pred.source_row.astype(int)
        idx=pd.MultiIndex.from_frame(target[['source_file','source_row']])
        if pred.duplicated(['source_file','source_row']).any() or set(map(tuple,pred[['source_file','source_row']].values))!=set(idx):
            raise RuntimeError('Legacy prediction view IDs differ')
        pred=pred.set_index(['source_file','source_row']).loc[idx].reset_index()
    if pred.true_label.tolist()!=target.true_label.tolist():raise RuntimeError('Prediction labels differ from frozen data')
    unknown=set(pred.prediction)-set(LABELS)-{'__off_label__'}
    if unknown:raise RuntimeError(f'Unknown predictions: {unknown}')
    return pred


def source_predictions(model,variant,target,role,action='evaluate',condition=None):
    path=EXP/'evaluation'/action/model/variant/(condition if action=='long-input' else 'suite')
    meta=completed(path)
    if load_plan()['models'][model]['family']=='encoder':
        parity_path=path/'validation_inference_parity.json'
        parity=json.loads(parity_path.read_text())
        reference=project_path(ROOT, parity['reference_path'])
        if (not parity['passed'] or parity['validation_rows']!=2000 or parity['prediction_differences']!=0
                or parity['final_test_read_before_check'] or sha_file(reference)!=parity['reference_sha256']):
            raise RuntimeError('Encoder validation/final inference parity not established')
        remember(parity_path,'encoder_validation_inference_parity')
    files=[path/'internal_test_predictions.csv.gz'] if role=='internal' else sorted(path.glob('*_predictions.csv.gz'))
    if role=='legacy':files=[p for p in files if p.name!='internal_test_predictions.csv.gz']
    if len(files)!=(1 if role=='internal' else 12):raise RuntimeError(f'Missing predictions: {path}')
    frames=[]
    for file in files:
        frames.append(pd.read_csv(file,keep_default_na=False,dtype={'record_id':str}))
        remember(file,'new_gpu_prediction')
    pred=align_predictions(pd.concat(frames,ignore_index=True),target,role)
    saved=json.loads((path/'metrics.json').read_text())
    if role=='internal':
        check_metrics(target.true_label,pred.prediction,saved['internal_test'])
    else:
        for name,group in pred.groupby('source_file',sort=False):
            check_metrics(group.true_label,group.prediction,saved[Path(name).stem])
    return pred.prediction.to_numpy()


def check_metrics(true,pred,saved,weights=None):
    actual=[accuracy_score(true,pred,sample_weight=weights),
            f1_score(true,pred,labels=LABELS,average='macro',zero_division=0,sample_weight=weights),
            f1_score(true,pred,labels=LABELS,average='weighted',zero_division=0,sample_weight=weights)]
    for key,value in zip(METRICS,actual):
        if not np.isclose(value,saved[key],atol=1e-12,rtol=0):raise RuntimeError(f'Metric mismatch: {key}')
    return np.asarray(actual)


def targets():
    internal=load_frame(DATA/'internal_test.csv').rename(columns={'product':'true_label'})
    manifest=pd.read_csv(project_path(PAPER, 'analysis/revision_v1/internal_test_manifest.csv'),dtype={'record_id':str})
    mapping=manifest.set_index('record_id')['group']
    internal['group']=internal.record_id.astype(str).map(mapping)
    if internal.group.isna().any():raise RuntimeError('Missing internal group')
    internal['group']=internal.group.astype(int)
    legacy=pd.read_csv(DATA/'legacy_test_manifest.csv')
    groups=pd.read_csv(project_path(PAPER, 'analysis/cpu_revision/legacy_groups.csv'))
    if not legacy[['source_file','source_row','true_label']].equals(groups[['source_file','source_row','true_label']]):
        raise RuntimeError('Legacy group alignment mismatch')
    legacy['group']=groups.prefix250_group
    legacy['full_normalized']=groups.full_normalized
    sizes=legacy.groupby(['true_label','full_normalized']).source_row.transform('size')
    legacy['unit_weight']=1.0/sizes
    return internal,legacy


def systems_and_predictions(target,role):
    arrays=[];systems=[];variant_names=[]
    for model,title in [('qwen','Qwen3-8B'),('llama','Llama-3.1-8B-Instruct')]:
        for variant in ['base','42','43','44']:
            arrays.append(source_predictions(model,variant,target,role));variant_names.append(f'{model}_{variant}')
        start=len(arrays)-4
        systems += [dict(system=model+'_base',name=title+' zero-shot',indices=[start],train_rows=0,stochastic_runs=1),
                    dict(system=model+'_ft',name=title+' QLoRA',indices=[start+1,start+2,start+3],train_rows=10000,stochastic_runs=3)]
    for model,title in [('bert','BERT-base'),('distilbert','DistilBERT'),('roberta','RoBERTa-base')]:
        start=len(arrays)
        for variant in ['42','43','44']:
            arrays.append(source_predictions(model,variant,target,role));variant_names.append(f'{model}_{variant}')
        systems.append(dict(system=model,name=title,indices=list(range(start,start+3)),train_rows=10000,stochastic_runs=3))
    for budget,count in [('10000',10000),('full',120502)]:
        checks=[]
        for seed in [42,43,44]:
            file=project_path(PAPER, f'experiments/tfidf_revision_v1/budget_{budget}/seed_{seed}', 'internal_test_predictions.csv.gz' if role=='internal' else 'legacy_predictions.csv.gz')
            pred=align_predictions(pd.read_csv(file,keep_default_na=False,dtype={'record_id':str}),target,role)
            checks.append(pred.prediction.to_numpy());remember(file,'existing_corrected_tfidf_prediction')
        if not all(np.array_equal(checks[0],p) for p in checks[1:]):raise RuntimeError('TF-IDF determinism check changed')
        start=len(arrays);arrays.append(checks[0]);variant_names.append('tfidf_'+budget)
        systems.append(dict(system='tfidf_'+budget,name='TF-IDF/LR '+('10k' if budget=='10000' else 'full'),indices=[start],train_rows=count,stochastic_runs=0))
    return np.stack(arrays,axis=1),systems,variant_names


def summarise_role(target,role):
    predictions,systems,names=systems_and_predictions(target,role)
    weights=target.unit_weight.to_numpy() if role=='legacy' else None
    y=encode(target.true_label.to_numpy());pred=encode(predictions)
    cm=group_confusions(y,pred,target.group.to_numpy(),weights)
    point,samples=paired_cluster_bootstrap(cm,REPLICATES,RNG_SEED)
    # One common group draw across all nineteen variants, enabling paired contrasts.
    rows=[];seeds=[];lookup={}
    for spec in systems:
        idx=spec['indices'];observed=point[idx].mean(axis=0);draws=samples[:,idx].mean(axis=1)
        lookup[spec['system']]=(observed,draws,spec)
        low,high=np.quantile(draws,[.025,.975],axis=0)
        for k,metric in enumerate(METRICS):
            sd=float(point[idx,k].std(ddof=1)) if len(idx)==3 else 0.0 if spec['stochastic_runs']==0 else None
            rows.append({'role':role,'system':spec['system'],'name':spec['name'],'train_rows':spec['train_rows'],
                'metric':metric,'point':float(observed[k]),'low':float(low[k]),'high':float(high[k]),
                'seed_sd_ddof1':sd,'rows':len(target),'groups':len(cm),'replicates':REPLICATES})
        for index in idx:
            expected=check_metrics(target.true_label,predictions[:,index],dict(zip(METRICS,point[index])),weights)
            for key,value in zip(METRICS,expected):seeds.append({'role':role,'variant':names[index],'metric':key,'value':float(value)})
    contrasts=[]
    pairs=[('qwen_base','qwen_ft','within_model_adaptation'),('llama_base','llama_ft','within_model_adaptation')]
    pairs += [('tfidf_10000',key,'matched_10k_baseline_comparison') for key in ['qwen_ft','llama_ft','bert','distilbert','roberta']]
    pairs += [('tfidf_full',key,'unequal_training_budget_context') for key in ['qwen_ft','llama_ft']]
    for reference,comparison,kind in pairs:
        a,ad,aspec=lookup[reference];b,bd,bspec=lookup[comparison]
        # Subtract the common reference before averaging seeds; exact ties then
        # remain zero rather than acquiring roundoff from a three-value mean.
        paired_point=(point[bspec['indices']]-point[aspec['indices'][0]]).mean(axis=0)
        paired_draws=(samples[:,bspec['indices']]-samples[:,aspec['indices'][0],None]).mean(axis=1)
        low,high=np.quantile(paired_draws,[.025,.975],axis=0)
        for k,metric in enumerate(METRICS):contrasts.append({'role':role,'reference':reference,'comparison':comparison,
            'kind':kind,'metric':metric,'paired_difference':float(paired_point[k]),'low':float(low[k]),'high':float(high[k])})
    np.savez_compressed(OUT/f'corrected_bootstrap_{role}.npz',samples=samples,variant=np.asarray(names),metric=np.asarray(METRICS))
    return rows,seeds,contrasts


def long_input_results(internal):
    rows=[]
    conditions=load_plan()['long_input']['conditions']
    for model in ['qwen','llama']:
        arrays=[]
        for variant in ['base','42','43','44']:
            per_condition=[source_predictions(model,variant,internal,'internal','long-input',c['name']) for c in conditions]
            if not np.array_equal(per_condition[0],per_condition[1]):raise RuntimeError(f'{model}/{variant}: token-cap control predictions differ')
            arrays.extend(per_condition)
        predictions=np.stack(arrays,axis=1)
        for cohort,mask in [('all',np.ones(len(internal),dtype=bool)),('full_narrative_gt500',internal.narrative.str.len().gt(500).to_numpy())]:
            target=internal.loc[mask]
            cm=group_confusions(encode(target.true_label.to_numpy()),encode(predictions[mask]),target.group.to_numpy())
            point,samples=paired_cluster_bootstrap(cm,REPLICATES,RNG_SEED)
            for variant,indices in [('base',[0]),('ft_seed_mean',[3,6,9])]:
                for a,b,label in [(0,1,'token_cap_control'),(1,2,'500_to_1500_chars')]:
                    ai=[i+a for i in indices];bi=[i+b for i in indices]
                    x=point[ai].mean(axis=0);y=point[bi].mean(axis=0)
                    gains=(samples[:,bi]-samples[:,ai]).mean(axis=1)
                    lo,hi=np.quantile(gains,[.025,.975],axis=0)
                    for k,metric in enumerate(METRICS):
                        rows.append({'model':model,'variant':variant,'cohort':cohort,'comparison':label,
                            'reference_condition':conditions[a]['name'],'comparison_condition':conditions[b]['name'],
                            'metric':metric,'reference_point':float(x[k]),'comparison_point':float(y[k]),
                            'paired_difference':float((point[bi,k]-point[ai,k]).mean()),'low':float(lo[k]),'high':float(hi[k]),
                            'rows':len(target),'groups':len(cm),'replicates':REPLICATES})
    return rows


def latency_results(internal):
    manifest=pd.read_csv(PREP/'latency_sample_manifest.csv',dtype={'record_id':str})
    expected=manifest.record_id.tolist()
    group_map=dict(zip(internal.record_id.astype(str),internal.group))
    groups=np.asarray([group_map[x] for x in expected]);unique,inv=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(RNG_SEED)
    # Paired fixed group multiplicities; repeats are averaged within each record first.
    multiplicities=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=REPLICATES)
    counts=np.bincount(inv);denominator=multiplicities@counts
    summaries=[];paired=[]
    for model in ['qwen','llama']:
        for condition in load_plan()['long_input']['conditions']:
            per_record=[]
            for variant in ['base','42','43','44']:
                path=EXP/'latency'/model/variant/condition['name'];completed(path)
                file=path/'latency_records.csv';remember(file,'controlled_latency')
                frame=pd.read_csv(file,dtype={'record_id':str})
                if len(frame)!=300 or frame.duplicated(['record_id','repetition']).any():raise RuntimeError('Incomplete/duplicate latency records')
                if set(frame.record_id)!=set(expected) or set(frame.repetition)!={0,1,2}:raise RuntimeError('Latency IDs/repeats differ')
                if not np.isfinite(frame[['generation_seconds','end_to_end_seconds']].to_numpy()).all() or (frame[['generation_seconds','end_to_end_seconds']]<=0).any().any():raise RuntimeError('Invalid latency time')
                means=frame.groupby('record_id')[['generation_seconds','end_to_end_seconds']].mean().loc[expected]
                per_record.append(means.to_numpy())
                for key in ['generation_seconds','end_to_end_seconds']:
                    summaries.append({'model':model,'variant':variant,'condition':condition['name'],'timing':key,
                        'mean':float(frame[key].mean()),'median':float(frame[key].median()),'p95':float(frame[key].quantile(.95)),
                        'records':100,'repetitions':3,'timed_calls':300,
                        'serial_requests_per_second':300/float(frame.end_to_end_seconds.sum()),
                        'mean_input_tokens':float(frame.input_tokens.mean()),'mean_new_tokens':float(frame.new_tokens.mean())})
            times=np.stack(per_record,axis=1)
            aggregate=np.zeros((len(unique),4,2));np.add.at(aggregate,inv,times)
            draws=np.einsum('bg,gvm->bvm',multiplicities,aggregate)/denominator[:,None,None]
            diff=draws[:,1:].mean(axis=1)-draws[:,0]
            lo,hi=np.quantile(diff,[.025,.975],axis=0)
            observed=times[:,1:].mean(axis=1).mean(axis=0)-times[:,0].mean(axis=0)
            for k,timing in enumerate(['generation_seconds','end_to_end_seconds']):
                paired.append({'model':model,'condition':condition['name'],'timing':timing,
                    'ft_seed_mean_minus_base_seconds':float(observed[k]),'low':float(lo[k]),'high':float(hi[k]),
                    'records':100,'groups':len(unique),'replicates':REPLICATES})
    return summaries,paired


def table_tex(frame,filename):
    rows=[]
    for r in frame.itertuples(index=False):
        sd='---' if pd.isna(r.seed_sd_ddof1) else f'{r.seed_sd_ddof1:.4f}'
        name=r.name.replace('_',r'\_')
        rows.append(f'{name} & {r.train_rows:,} & {r.point:.4f} & [{r.low:.4f}, {r.high:.4f}] & {sd} '+r'\\')
    (project_path(PAPER, 'tables', filename)).write_text('\n'.join(rows)+'\n'+r'\hline'+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);verify_data(include_final=True)
    # All GPU jobs must finish before reading any final outputs for inference.
    from run_gpu_study import jobs
    for job in jobs():validate_training_job(job)
    original=json.loads((project_path(PAPER, 'analysis/revision_v1/source_integrity_before.json')).read_text())
    for name,expected in original.items():
        if sha_file(project_path(ROOT, name))!=expected:raise RuntimeError(f'Protected original source changed: {name}')
    internal,legacy=targets()
    estimates=[];seed_rows=[];contrasts=[]
    with threadpool_limits(limits=4):
        for role,target in [('internal',internal),('legacy',legacy)]:
            r,s,c=summarise_role(target,role);estimates+=r;seed_rows+=s;contrasts+=c
        long=long_input_results(internal)
        latency,latency_paired=latency_results(internal)
    estimates=pd.DataFrame(estimates);seeds=pd.DataFrame(seed_rows);contrasts=pd.DataFrame(contrasts)
    estimates.to_csv(OUT/'corrected_model_metrics.csv',index=False)
    seeds.to_csv(OUT/'corrected_per_seed_metrics.csv',index=False)
    contrasts.to_csv(OUT/'corrected_paired_contrasts.csv',index=False)
    pd.DataFrame(long).to_csv(OUT/'corrected_long_input.csv',index=False)
    pd.DataFrame(latency).to_csv(OUT/'controlled_latency_summary.csv',index=False)
    pd.DataFrame(latency_paired).to_csv(OUT/'controlled_latency_paired.csv',index=False)
    table_tex(estimates.loc[estimates.role.eq('internal') & estimates.metric.eq('macro_f1')],'table_corrected_internal_macro.tex')
    table_tex(estimates.loc[estimates.role.eq('legacy') & estimates.metric.eq('macro_f1')],'table_corrected_legacy_macro.tex')
    source_by_path={r['path']:r for r in SOURCES}
    (OUT/'GPU_RESULT_SOURCES.json').write_text(json_text(list(source_by_path.values())))
    method={'bootstrap_replicates':REPLICATES,'rng_seed':RNG_SEED,
        'analysis_library_versions':versions(),
        'internal':'Row-weighted frozen internal groups, observed point estimates; 19 variants share each group-bootstrap draw.',
        'legacy':'Equal total weight per (true label, normalized visible narrative); preserve all appearances in 3919 prefix/full dependency groups.',
        'seed':'Sample SD across three actual same-data optimisation refits; deterministic TF-IDF checks reported separately.',
        'long_input':'Matched same-checkpoint conditions; all IDs and predefined >500-character cohort; token-cap control checked for equal predictions.',
        'latency':'Three repeats averaged within record before group bootstrap; common group multiplicities across base/FT. Means/median/p95 remain conditional on observed hardware/job blocks.',
        'limits':['Retrospective evaluation on previously inspected records; prospective generalisation is not established.','Fixed-trained-model bootstrap excludes training/model-selection uncertainty.','Latency variants are separate job blocks, not a fully randomized crossover.','No deployment acceptability, regulatory readiness or independent-source generalization claim.']}
    method['limits'].append('Auxiliary and primary experiments differ in data allocation, model identities, serialization and hardware; their score difference does not isolate the effect of overlap screening.')
    (OUT/'GPU_STATISTICAL_METHOD.json').write_text(json_text(method))
    today=datetime.now(ZoneInfo('Europe/Istanbul')).date().isoformat()
    lines=['# Main model comparison: results and evidence',
        '', 'All systems below use the fixed, overlap-screened allocation. Base LLMs have no adaptation budget; full-data TF-IDF uses a larger budget and is contextual. Checkpoint provenance and actual source/config versions are retained per run.',
        '', '## Internal test macro-F1','', '| Model | Train rows | Macro-F1 | 95% paired-group CI | Run SD |','| --- | ---: | ---: | --- | ---: |']
    for r in estimates.loc[estimates.role.eq('internal') & estimates.metric.eq('macro_f1')].itertuples(index=False):
        sd='—' if pd.isna(r.seed_sd_ddof1) else f'{r.seed_sd_ddof1:.4f}'
        lines.append(f'| {r.name} | {r.train_rows:,} | {r.point:.4f} | [{r.low:.4f}, {r.high:.4f}] | {sd} |')
    lines += ['', '## What is recorded','',
        '- `corrected_model_metrics.csv`: internal and unique-input-weighted benchmark point estimates, intervals and run SD.',
        '- `corrected_paired_contrasts.csv`: within-model adaptation and matched-10k TF-IDF comparisons, with paired intervals. Full-data contrasts are explicitly unequal-budget context.',
        '- `corrected_long_input.csv`: matched inference-length effects and token-cap control; no longer-context retraining claim.',
        '- `controlled_latency_summary.csv` / `controlled_latency_paired.csv`: synchronized real calls, all 100 IDs and three repetitions; cache bypassed.',
        '- `GPU_RESULT_SOURCES.json`: SHA-256 evidence for every prediction and latency source.',
        '', '## Interpretation limits','']
    lines += ['- '+x for x in method['limits']]
    lines += ['', 'Code, data selections, configurations and saved predictions are available at https://github.com/muhammettan28/CFPB_tuning. CPU verification commands are documented in the root README and docs/REPRODUCIBILITY.md.']
    (OUT/'GPU_RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n')
    validation={'date':today,'all_88_gpu_jobs_complete':True,'prediction_sources_checked':len(source_by_path),
        'all_fitting_budgets_manifests_verified':True,'encoder_validation_searches_independently_rescored':3,
        'protected_original_files_unchanged':len(original),
        'sklearn_point_checks':'all 19 variants in both roles within 1e-12','bootstrap_replicates':REPLICATES,
        'protocol_sha256':PROTOCOL_SHA,'plan_sha256':sha_file(CONFIG),'gpu_experiments_completed':True,
        'manuscript_final_review_completed':False,'uploaded':False}
    (OUT/'GPU_RESULT_VALIDATION.json').write_text(json_text(validation))
    from prepare_reviewer_crosswalk import main as update_responses
    update_responses()
    from build_local_reviewer_package import main as update_package
    update_package()
    print('Validated corrected GPU predictions, paired uncertainty, long-input conditions and controlled latency. Report written.',flush=True)


if __name__=='__main__':main()

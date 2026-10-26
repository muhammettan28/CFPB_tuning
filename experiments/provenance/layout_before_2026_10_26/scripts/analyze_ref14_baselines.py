#!/usr/bin/env python3
"""Paired analysis of the predeclared reference-parameter encoder refits, CPU only."""
from __future__ import annotations
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import analyze_corrected_gpu_results as core
from analyze_archived_predictions import encode,group_confusions,paired_cluster_bootstrap,METRICS
from revision_gpu_common import ROOT,PAPER,DATA,EXP,load_frame,json_text,sha_file,PROTOCOL_SHA,versions
from run_ref14_baselines import OUT,CONFIG_EXTRA,config,job_path,guard


def validate_fit(model,seed):
    path=job_path('fit',model,seed);meta=core.completed(path)
    spec=config()['models'][model]
    if meta['actual_settings']!=spec or meta['optimization_seed']!=seed or meta['supplementary_configuration_sha256']!=sha_file(CONFIG_EXTRA):
        raise RuntimeError('Supplementary fitted settings/seed differ')
    cols=['record_id','product','narrative_hash']
    for role,name,size in [('train','train_10000.csv',10000),('validation','validation_2000.csv',2000)]:
        frozen=load_frame(DATA/name)
        saved=pd.read_csv(path/f'{role}_selection_manifest.csv',dtype=str)
        if (meta[f'actual_{role}_rows']!=size or meta[f'{role}_csv_sha256']!=sha_file(DATA/name)
                or not saved[cols].equals(frozen[cols].astype(str).reset_index(drop=True))):
            raise RuntimeError('Supplementary fitting membership/order/label/hash/budget differs')
    saved=pd.read_csv(path/'validation_predictions.csv',dtype={'record_id':str}).rename(columns={'product':'true_label'})
    aligned=core.align_predictions(saved,load_frame(DATA/'validation_2000.csv').rename(columns={'product':'true_label'}),'internal')
    score=json.loads((path/'validation_metrics.json').read_text())
    core.check_metrics(aligned.true_label,aligned.prediction,score)
    core.remember(path/'validation_predictions.csv','supplementary_validation')
    return score


def predictions(model,seed,target,role):
    path=job_path('evaluation',model,seed);meta=core.completed(path)
    if meta['supplementary_configuration_sha256']!=sha_file(CONFIG_EXTRA):raise RuntimeError('Evaluation configuration differs')
    parity_path=path/'validation_inference_parity.json';parity=json.loads(parity_path.read_text())
    source=job_path('fit',model,seed)/'validation_predictions.csv'
    if (not parity['passed'] or parity['validation_rows']!=2000 or parity['prediction_differences']!=0
            or parity['final_test_read_before_check'] or parity['max_length']!=config()['models'][model]['max_length']
            or parity['reference_path']!=str(source.relative_to(ROOT)) or parity['reference_sha256']!=sha_file(source)):
        raise RuntimeError('Supplementary reload inference parity not established')
    core.remember(parity_path,'supplementary_validation_inference_parity')
    files=[path/'internal_test_predictions.csv.gz'] if role=='internal' else [p for p in sorted(path.glob('*_predictions.csv.gz')) if p.name!='internal_test_predictions.csv.gz']
    if len(files)!=(1 if role=='internal' else 12):raise RuntimeError('Missing supplementary prediction views')
    frames=[]
    for file in files:
        frames.append(pd.read_csv(file,keep_default_na=False,dtype={'record_id':str}));core.remember(file,'supplementary_final_prediction')
    pred=core.align_predictions(pd.concat(frames,ignore_index=True),target,role)
    saved=json.loads((path/'metrics.json').read_text())
    if role=='internal':core.check_metrics(pred.true_label,pred.prediction,saved['internal_test'])
    else:
        for name,group in pred.groupby('source_file',sort=False):core.check_metrics(group.true_label,group.prediction,saved[name[:-4]])
    return pred.prediction.to_numpy()


def summarise(target,role):
    arrays=[];systems=[]
    for model in config()['models']:
        for setting in ['main','ref14']:
            idx=[]
            for seed in [42,43,44]:
                idx.append(len(arrays))
                arrays.append(core.source_predictions(model,str(seed),target,role) if setting=='main' else predictions(model,seed,target,role))
            systems.append((model,setting,idx))
    predicted=np.stack(arrays,axis=1)
    weights=target.unit_weight.to_numpy() if role=='legacy' else None
    cm=group_confusions(encode(target.true_label.to_numpy()),encode(predicted),target.group.to_numpy(),weights)
    point,draws=paired_cluster_bootstrap(cm,10000,20261020)
    rows=[];contrasts=[]
    for model,setting,idx in systems:
        mean=point[idx].mean(axis=0);low,high=np.quantile(draws[:,idx].mean(axis=1),[.025,.975],axis=0)
        for k,metric in enumerate(METRICS):rows.append({'role':role,'model':model,'setting':setting,'metric':metric,
            'point':float(mean[k]),'low':float(low[k]),'high':float(high[k]),'seed_sd_ddof1':float(point[idx,k].std(ddof=1)),
            'rows':len(target),'groups':len(cm),'train_rows':10000,'replicates':10000})
        for index in idx:core.check_metrics(target.true_label,predicted[:,index],dict(zip(METRICS,point[index])),weights)
    for model in config()['models']:
        a=next(idx for m,s,idx in systems if m==model and s=='main')
        b=next(idx for m,s,idx in systems if m==model and s=='ref14')
        delta=(point[b]-point[a]).mean(axis=0)
        low,high=np.quantile((draws[:,b]-draws[:,a]).mean(axis=1),[.025,.975],axis=0)
        for k,metric in enumerate(METRICS):contrasts.append({'role':role,'model':model,'metric':metric,
            'ref14_minus_main':float(delta[k]),'low':float(low[k]),'high':float(high[k]),'replicates':10000})
    return rows,contrasts


def main():
    cfg=config();core.SOURCES.clear()
    if not json.loads((PAPER/'analysis/gpu_execution/GPU_RESULT_VALIDATION.json').read_text())['gpu_experiments_completed']:
        raise RuntimeError('Core analysis must complete first')
    validation=[]
    for model in cfg['models']:
        guard(model)
        for seed in [42,43,44]:
            result=validate_fit(model,seed)
            validation.append({'model':model,'seed':seed,**result})
    protected=json.loads((PAPER/'analysis/revision_v1/source_integrity_before.json').read_text())
    for name,digest in protected.items():
        if sha_file(ROOT/name)!=digest:raise RuntimeError(f'Protected original changed: {name}')
    internal,legacy=core.targets();rows=[];contrasts=[]
    with threadpool_limits(limits=4):
        for role,target in [('internal',internal),('legacy',legacy)]:
            r,c=summarise(target,role);rows+=r;contrasts+=c
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'model_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'validation_metrics.csv',index=False)
    today=datetime.now(ZoneInfo('Europe/Istanbul')).date().isoformat()
    lines=['# Reference [14] encoder-parameter sensitivity','',f'Local completion date: {today}.',
        '', 'BERT/RoBERTa parameters in Tables III–IV were specified before supplementary final predictions. All six three-epoch refits use the same frozen 10,000 training and 2,000 validation examples. The published learning rates and epochs were already the main validation winners; the added conditions change physical batch to 16 (accumulation 1) and token limits to 128/100. Their joint effect is measured, not attributed to an individual hyperparameter.',
        '', 'This is a parameter-sensitivity extension, not an exact reproduction of the paper. We retain 500-character inputs, corrected split, pinned checkpoints, PyTorch/FP16 and the main warmup/weight-decay/scheduler settings. Source Table IV reports TensorFlow for RoBERTa. The source uses another split and preprocessing. All conditions are reported without selecting a replacement baseline on test scores.',
        '', '| Evaluation | Model | Setting | Macro-F1 | 95% group-bootstrap CI | Run SD |',
        '| --- | --- | --- | ---: | --- | ---: |']
    for r in frame.loc[frame.metric.eq('macro_f1')].itertuples(index=False):
        lines.append(f'| {r.role} | {r.model} | {r.setting} | {r.point:.4f} | [{r.low:.4f}, {r.high:.4f}] | {r.seed_sd_ddof1:.4f} |')
    lines += ['', 'Intervals share 10,000 frozen-group bootstrap draws across original and added predictions. Legacy views receive the same unique-normalized-input weights as in the core analysis. Run SD is reported separately; the intervals condition on fitted checkpoints and exclude training/model-selection uncertainty. Evaluation is retrospective, not a new unseen prospective test. Read paired_contrasts.csv for reference-setting minus main-setting intervals. No numerical superiority over the published paper is inferred.']
    (OUT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n')
    sources={r['path']:r for r in core.SOURCES}
    (OUT/'RESULT_SOURCES.json').write_text(json_text(list(sources.values())))
    (OUT/'RESULT_VALIDATION.json').write_text(json_text({'completed':True,'date':today,'all_12_gpu_jobs_complete':True,
        'actual_six_refits':True,'all_frozen_memberships_verified':True,'reload_parity_verified':6,
        'protected_original_files_unchanged':len(protected),'configuration_sha256':sha_file(CONFIG_EXTRA),
        'analysis_code_sha256':sha_file(__file__),'analysis_library_versions':versions(),
        'point_metrics_independently_verified':True,'uploaded':False}))
    with (PAPER/'REVISION_LOG.md').open('a') as f:
        f.write(f'\n## Ref [14] ek encoder deneyleri — otomatik tamamlanma ({today})\n\nBERT/RoBERTa kaynak tablo ayarlarıyla altı gerçek refit ve altı evaluation tamamlandı; üyelik, inference eşliği, metrikler ve orijinal dosya bütünlüğü doğrulandı. Sonuçlar: [RESULTS_REPORT.md](analysis/ref14_encoder/RESULTS_REPORT.md), paired_contrasts.csv ve RESULT_VALIDATION.json. Birebir replikasyon veya test üzerinden baseline seçimi değildir.\n')
    from prepare_reviewer_crosswalk import main as responses
    responses()
    from build_local_reviewer_package import main as package
    package()
    print('Reference-parameter sensitivity completed and independently checked.',flush=True)


if __name__=='__main__':main()

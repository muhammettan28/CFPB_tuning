#!/usr/bin/env python3
"""Separate literature-parameter sensitivity; default is a CPU-only queue preview."""
from __future__ import annotations
import argparse
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from revision_gpu_common import (ROOT,PAPER,DATA,CONFIG,PROTOCOL_SHA,artifact_entry,audit_guard,
    json_text,load_frame,load_plan,metrics,sha_file,sha_text,snapshot_hashes,verify_data)
from audit_gpu_tokenizers import separation_failures

CONFIG_EXTRA=PAPER/'configs/ref14_encoder_v1.json'
OUT=PAPER/'analysis/ref14_encoder'
EXP=PAPER/'experiments/ref14_encoder_v1'
REGISTRY=PAPER/'configs/model_registry.local.json'


def config():
    if sha_file(CONFIG_EXTRA)!=CONFIG_EXTRA.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('Supplementary configuration changed')
    cfg=json.loads(CONFIG_EXTRA.read_text())
    if cfg['parent_gpu_plan_sha256']!=sha_file(CONFIG) or cfg['parent_protocol_sha256']!=PROTOCOL_SHA:
        raise RuntimeError('Parent protocol changed')
    return cfg


def prepare():
    """Token-only audit, including shorter-cap separation and bootstrap groups."""
    from transformers import AutoTokenizer
    verify_data(include_final=True);cfg=config();OUT.mkdir(parents=True,exist_ok=True)
    groups=pd.read_csv(PAPER/'analysis/revision_v1/internal_test_manifest.csv',dtype={'record_id':str}).set_index('record_id')['group']
    legacy_groups=pd.read_csv(PAPER/'analysis/cpu_revision/legacy_groups.csv').set_index(['source_file','source_row']).prefix250_group
    frames=[('train',DATA/'train_10000.csv'),('validation',DATA/'validation_2000.csv'),('internal_test',DATA/'internal_test.csv')]
    frames += [('legacy',p) for p in sorted((DATA/'legacy_subsets').glob('*.csv'))]
    reports=[]
    for model,spec in cfg['models'].items():
        artifact,snapshot,_=audit_guard(REGISTRY,model)
        if artifact['commit']!=spec['commit'] or artifact['repo_id']!=spec['repo_id']:raise RuntimeError('Supplementary model pin differs')
        tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
        tok.truncation_side='right';rows=[]
        for role,path in frames:
            frame=load_frame(path)
            texts=frame.narrative.str[:500].tolist()
            for start in range(0,len(frame),512):
                batch=texts[start:start+512]
                encoded=tok(batch,truncation=True,max_length=spec['max_length'],return_offsets_mapping=True)
                full=tok(batch,truncation=False)['input_ids']
                for i,(ids,offsets,full_ids) in enumerate(zip(encoded['input_ids'],encoded['offset_mapping'],full)):
                    pos=start+i
                    rows.append({'role':role,'source_file':path.name,'source_row':pos,
                        'record_id':str(frame.record_id.iloc[pos]) if 'record_id' in frame else f'{path.name}:{pos}',
                        'token_hash':sha_text(json.dumps(ids)),'tokens_before_cap':len(full_ids),'tokens_retained':len(ids),
                        'retained_narrative_chars':max(b for a,b in offsets)})
        keys=pd.DataFrame(rows);failures=separation_failures(keys)
        violations={}
        for role in ['internal_test','legacy']:
            part=keys.loc[keys.role.eq(role)].copy()
            part['group']=part.record_id.map(groups) if role=='internal_test' else [legacy_groups.loc[(r.source_file,r.source_row)] for r in part.itertuples(index=False)]
            if part.group.isna().any():raise RuntimeError('Missing frozen dependency group')
            violations[role]=int(part.groupby('token_hash').group.nunique().gt(1).sum())
        path=OUT/f'input_keys_{model}.csv.gz';keys.to_csv(path,index=False)
        report={'model':model,'artifact':artifact,'configuration_sha256':sha_file(CONFIG_EXTRA),
                'parent_protocol_sha256':PROTOCOL_SHA,'keys_sha256':sha_file(path),'rows':len(keys),
                'max_length':spec['max_length'],'clipped_rows':int(keys.tokens_before_cap.gt(spec['max_length']).sum()),
                'separation_failures':failures,'token_keys_spanning_multiple_frozen_groups':violations,
                'passed':not failures and not any(violations.values()),
                'scope':'Actual 500-character input with reference token cap; no model/GPU load. Exact-token separation, not semantic independence.'}
        (OUT/f'input_audit_{model}.json').write_text(json_text(report));reports.append(report)
        print(f'{model}: {len(keys)} rows; separation={failures}, group violations={violations}',flush=True)
    (OUT/'PREPARATION.json').write_text(json_text({'configuration_sha256':sha_file(CONFIG_EXTRA),'passed':all(r['passed'] for r in reports),'models':reports}))
    if not all(r['passed'] for r in reports):raise RuntimeError('Shorter-cap audit failed; amendment required before training')


def guard(model):
    cfg=config();spec=cfg['models'][model]
    artifact,snapshot,audit_hash=audit_guard(REGISTRY,model)
    report=json.loads((OUT/f'input_audit_{model}.json').read_text())
    if (not report['passed'] or report['artifact']!=artifact or report['configuration_sha256']!=sha_file(CONFIG_EXTRA)
            or report['keys_sha256']!=sha_file(OUT/f'input_keys_{model}.csv.gz') or artifact['commit']!=spec['commit']):
        raise RuntimeError('Supplementary input audit/provenance mismatch')
    return cfg,spec,artifact,snapshot,audit_hash


def job_path(action,model,seed):return EXP/action/model/f'seed_{seed}'


def execute_job(action,model,seed):
    from types import SimpleNamespace
    from run_revision_gpu import (metadata,new_output,fit_encoder,predict_encoder,
                                  require_encoder_validation_parity,tokenizer_at)
    cfg,spec,artifact,snapshot,audit_hash=guard(model)
    verify_data(include_final=action=='evaluation')
    import torch
    import transformers
    if transformers.__version__!='4.57.6' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('One CUDA GPU and the fixed compatible environment are required')
    out=new_output(job_path(action,model,seed))
    args=SimpleNamespace(action='ref14-'+action,model=model,seed=seed,variant=str(seed),registry=REGISTRY,execute_gpu=True)
    info=metadata(out,args,artifact,audit_hash,snapshot,torch)
    info.update(study='ref14_encoder_v1',supplementary_configuration_sha256=sha_file(CONFIG_EXTRA),
                supplementary_runner_sha256=sha_file(Path(__file__)),actual_settings=spec,optimization_seed=seed)
    (out/'run_metadata.json').write_text(json_text(info))
    shutil.copy2(__file__,out/'code_snapshot/run_ref14_baselines.py')
    shutil.copy2(CONFIG_EXTRA,out/'ref14_encoder_v1.json')
    if action=='fit':
        result=fit_encoder(snapshot,out,spec,seed,load_plan(),settings=spec)
        (out/'validation_metrics.json').write_text(json_text(result))
        saved=snapshot_hashes(out/'model')
    else:
        from transformers import AutoModelForSequenceClassification
        trained=job_path('fit',model,seed)
        complete=json.loads((trained/'COMPLETE.json').read_text())
        fitted_meta=json.loads((trained/'run_metadata.json').read_text())
        if fitted_meta['artifact']!=artifact or fitted_meta['supplementary_configuration_sha256']!=sha_file(CONFIG_EXTRA):raise RuntimeError('Fitted provenance differs')
        if snapshot_hashes(trained/'model')!=complete['saved_model_files_sha256']:raise RuntimeError('Fitted weights changed')
        if snapshot_hashes(snapshot)!=json.loads((trained/'base_snapshot_files.sha256.json').read_text()):raise RuntimeError('Base model changed')
        tok=tokenizer_at(snapshot)
        model_object=AutoModelForSequenceClassification.from_pretrained(trained/'model',local_files_only=True).to('cuda').eval()
        require_encoder_validation_parity(args,load_plan(),model_object,tok,out,
            reference_path=trained/'validation_predictions.csv',max_length=spec['max_length'])
        result={}
        for path in [DATA/'internal_test.csv',*sorted((DATA/'legacy_subsets').glob('*.csv'))]:
            frame=load_frame(path)
            predictions,lengths=predict_encoder(model_object,tok,frame,16,spec['max_length'])
            rows=[{'record_id':getattr(r,'record_id',f'{path.name}:{i}'),'source_file':path.name,'source_row':i,
                   'true_label':r.product,'prediction':predictions[i],'input_tokens':lengths[i]}
                  for i,r in enumerate(frame.itertuples(index=False))]
            pd.DataFrame(rows).to_csv(out/f'{path.stem}_predictions.csv.gz',index=False)
            result[path.stem]=metrics(frame['product'].tolist(),predictions)
        (out/'metrics.json').write_text(json_text(result));saved=None
        del model_object;gc.collect();torch.cuda.empty_cache()
    (out/'COMPLETE.json').write_text(json_text({'completed':True,'action':action,'result':result,'saved_model_files_sha256':saved}))
    print(f'Completed reference sensitivity: {action}/{model}/{seed}',flush=True)


def state(status,**details):
    path=OUT/'RUN_STATUS.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    if status!='failed':old.pop('error',None)
    if status!='waiting_for_core_study':old.pop('core_job',None)
    info={**old,'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),'coordinator_pid':os.getpid(),**details}
    temp=path.with_suffix('.json.tmp');temp.write_text(json_text(info));temp.replace(path)
    (OUT/'REF14_BASELINE_STATUS.md').write_text('# Ref [14] hiperparametre ek deneyleri\n\n'+
        f"- Durum: **{status}**\n- Son kayıt (UTC): {info['updated_utc']}\n- Tamamlanan ek GPU aşaması: {info.get('completed_jobs',0)} / 12\n- Aktif aşama: `{info.get('current_job','—')}`\n\n"+
        'Ana 88-aşamalı çalışma ayrı izlenir; bu ek çalışma GPU çakışmasını önlemek için ana koordinatör kilidini bekler. Tamamlanmış sonuç ile bekleyen iş aynı sayılmaz.\n')


def run_study(wait):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    own=(OUT/'coordinator.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for model in config()['models']:guard(model)
    core=PAPER/'analysis/gpu_execution'
    core_lock=(core/'orchestrator.lock').open('a+')
    while True:
        current=json.loads((core/'GPU_RUN_STATUS.json').read_text())
        if current['status'] in {'failed','interrupted','stopped_between_jobs'}:
            state('blocked_core_not_complete',core_status=current['status']);return
        if current['status']=='experiments_and_analysis_completed':
            try:fcntl.flock(core_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        if not wait:raise RuntimeError('Core study is still active; use --wait-for-core')
        state('waiting_for_core_study',completed_jobs=0,current_job=None,core_job=current.get('current_job'))
        time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1');env.pop('USE_TORCH',None)
    queue=[(action,model,seed) for action in ['fit','evaluation'] for model in config()['models'] for seed in config()['optimization_seeds']]
    try:
        for i,(action,model,seed) in enumerate(queue):
            out=job_path(action,model,seed);name=f'{action}__{model}__{seed}'
            if (out/'COMPLETE.json').exists():
                recorded=json.loads((out/'run_metadata.json').read_text())
                if recorded['supplementary_configuration_sha256']!=sha_file(CONFIG_EXTRA):raise RuntimeError('Existing supplementary run differs')
                state('running',completed_jobs=i+1,current_job=name);continue
            if out.exists():raise RuntimeError(f'Incomplete output preserved; investigate before retry: {out}')
            with (OUT/'logs'/f'{name}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',__file__,'--job',action,model,str(seed),'--execute'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    state('running',completed_jobs=i,current_job=name,child_pid=child.pid);time.sleep(10)
                if child.returncode or not (out/'COMPLETE.json').exists():raise RuntimeError(f'{name} failed; see supplementary logs')
            state('running',completed_jobs=i+1,current_job=name,child_pid=None)
        state('cpu_analysis_running',current_job='paired result analysis')
        env['USE_TORCH']='0'
        with (OUT/'logs/result_analysis.log').open('a') as log:
            subprocess.run(['/tmp/cfpb-revision-venv/bin/python',str(PAPER/'scripts/analyze_ref14_baselines.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('experiments_and_analysis_completed',completed_jobs=12,current_job=None,child_pid=None)
    except BaseException as error:
        state('failed',error=str(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--wait-for-core',action='store_true');p.add_argument('--job',nargs=3,metavar=('ACTION','MODEL','SEED'))
    a=p.parse_args()
    if a.prepare:prepare();return
    if a.job:
        action,model,seed=a.job
        if not a.execute or action not in {'fit','evaluation'} or model not in config()['models'] or int(seed) not in config()['optimization_seeds']:
            raise ValueError('Invalid or unauthorised supplementary job')
        execute_job(action,model,int(seed));return
    if not a.execute:print(json_text(config()));return
    run_study(a.wait_for_core)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run the authorised GPU study serially, with durable status and restart support."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from revision_gpu_common import CONFIG,DATA,EXP,PAPER,PREP,ROOT,artifact_entry,load_plan,sha_file,verify_data

EXEC=PAPER/'analysis/gpu_execution'
RUNNER=PAPER/'scripts/run_revision_gpu.py'


def now():return datetime.now(timezone.utc).isoformat()


def jobs():
    plan=load_plan();result=[]
    def add(action,model,seed=42,variant='base',condition='chars500_tokens512'):
        family=plan['models'][model]['family']
        args=[action,'--model',model]
        if action in {'qlora','encoder-refit'}:args+=['--seed',str(seed)]
        if action in {'evaluate','long-input','latency'}:args+=['--variant',variant]
        if action in {'long-input','latency'}:args+=['--condition',condition]
        if action=='smoke':out=EXEC/'smoke'/model
        elif action=='qlora':out=EXP/'qlora'/model/f'seed_{seed}'
        elif action=='encoder-search':out=EXP/'encoder'/model/'search'
        elif action=='encoder-refit':out=EXP/'encoder'/model/f'seed_{seed}'
        elif action=='latency':out=EXP/'latency'/model/variant/condition
        else:out=EXP/'evaluation'/action/model/variant/(condition if action=='long-input' else 'suite')
        result.append({'key':'__'.join([action,model,str(seed) if action in {'qlora','encoder-refit'} else variant,condition if action in {'long-input','latency'} else 'main']),
                       'action':action,'model':model,'args':args,'out':str(out)})
    for model in ['qwen','llama','bert','distilbert','roberta']:add('smoke',model)
    # All validation selections/refits precede all final predictions.
    for model in ['bert','distilbert','roberta']:add('encoder-search',model)
    for model in ['bert','distilbert','roberta']:
        for seed in [42,43,44]:add('encoder-refit',model,seed=seed)
    for model in ['qwen','llama']:
        for seed in [42,43,44]:add('qlora',model,seed=seed)
    for model in ['qwen','llama','bert','distilbert','roberta']:
        for variant in (['base','42','43','44'] if model in ['qwen','llama'] else ['42','43','44']):
            add('evaluate',model,variant=variant)
    for model in ['qwen','llama']:
        for variant in ['base','42','43','44']:
            for condition in plan['long_input']['conditions']:
                add('long-input',model,variant=variant,condition=condition['name'])
    for model in ['qwen','llama']:
        for variant in ['base','42','43','44']:
            for condition in plan['long_input']['conditions']:
                add('latency',model,variant=variant,condition=condition['name'])
    return result


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def status(state):
    state['heartbeat_utc']=now()
    atomic_json(EXEC/'GPU_RUN_STATUS.json',state)
    current=state.get('current_job') or '—'
    text=f'''# GPU deneylerinin güncel durumu

Son kayıt (UTC): {state['heartbeat_utc']}

- Durum: **{state['status']}**
- Tamamlanan GPU aşaması: **{state['completed_jobs']} / {state['total_gpu_jobs']}** (smoke dahil; bir encoder-search aşaması altı eğitim içerir)
- Aktif aşama: `{current}`
- Koordinatör PID: `{state.get('orchestrator_pid')}`
- Aktif iş PID: `{state.get('child_pid')}`
- Plan SHA-256: `{state['plan_sha256']}`

Ayrıntılı olaylar: `analysis/gpu_execution/RUN_EVENTS.jsonl`. İş logları: `analysis/gpu_execution/logs/`. Bu durum kaydı, tamamlanmamış deneyleri bitmiş saymaz. Son bilimsel rapor bütün final tahminler ve latency kayıtları hazır olduğunda üretilir.
'''
    if state.get('error'):text+='\nHata: '+state['error']+'\n'
    (EXEC/'GPU_RUN_STATUS.md').write_text(text)


def event(kind,**kwargs):
    with (EXEC/'RUN_EVENTS.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'event':kind,**kwargs},ensure_ascii=False)+'\n')


def checkpoint_for(job):
    if job['action']!='qlora':return None
    checkpoints=[]
    for p in (Path(job['out'])/'checkpoints').glob('checkpoint-*'):
        if (p/'trainer_state.json').exists() and (p/'optimizer.pt').exists():
            checkpoints.append(p)
    return max(checkpoints,key=lambda p:int(p.name.split('-')[-1])) if checkpoints else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--python',type=Path,default=Path.home()/'.venvs/cfpb-gpu-revision/bin/python')
    parser.add_argument('--archive-incomplete',action='store_true',help='Preserve failed partial output before restarting a non-resumable stage')
    args=parser.parse_args();queue=jobs()
    if not args.execute:
        print(json.dumps(queue,indent=2));return
    EXEC.mkdir(parents=True,exist_ok=True);(EXEC/'logs').mkdir(exist_ok=True)
    lock=(EXEC/'orchestrator.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A study orchestrator is already active')
    verify_data(include_final=True)
    authorization=json.loads((EXEC/'GPU_AUTHORIZATION_2026-10-21.json').read_text())
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
    env.pop('USE_TORCH',None)
    state={'status':'running','started_utc':now(),'orchestrator_pid':os.getpid(),'child_pid':None,
        'plan_sha256':sha_file(CONFIG),'authorization_file':'GPU_AUTHORIZATION_2026-10-21.json',
        'total_gpu_jobs':len(queue),'completed_jobs':0,'current_job':None,'gpu_available_authorized':True}
    atomic_json(EXEC/'EXECUTION_QUEUE.json',queue)
    event('orchestrator_started',pid=os.getpid(),jobs=len(queue),plan_sha256=sha_file(CONFIG))
    try:
        for i,job in enumerate(queue):
            out=Path(job['out'])
            if (out/'COMPLETE.json').exists():
                meta=json.loads((out/'run_metadata.json').read_text())
                artifact,_=artifact_entry(PAPER/'configs/model_registry.local.json',job['model'])
                if meta['plan_sha256']!=sha_file(CONFIG) or meta['artifact']!=artifact or meta['arguments']['action']!=job['action']:
                    raise RuntimeError(f'Existing completion provenance differs: {out}')
                state['completed_jobs']=i+1;event('verified_existing_complete',job=job['key']);continue
            if (EXEC/'STOP_AFTER_CURRENT').exists():
                state['status']='stopped_between_jobs';state['current_job']=None;status(state);event('stop_file_respected');return
            resume=checkpoint_for(job) if out.exists() else None
            if out.exists() and resume is None:
                if not args.archive_incomplete:raise RuntimeError(f'Incomplete existing output: {out}; preserve/review it before retrying')
                archive=EXEC/'failed_attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/job['key']
                archive.parent.mkdir(parents=True,exist_ok=True);out.rename(archive)
                event('incomplete_output_preserved',job=job['key'],path=str(archive))
            command=[str(args.python),'-u',str(RUNNER),*job['args'],'--execute-gpu']
            if resume:command+=['--resume-from-checkpoint',str(resume)]
            state.update(current_job=job['key'],job_number=i+1,job_started_utc=now(),status='running')
            log=EXEC/'logs'/(job['key']+'.log')
            event('job_started',job=job['key'],log=str(log.relative_to(ROOT)),resume=str(resume) if resume else None)
            with log.open('a') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;status(state)
                while child.poll() is None:
                    time.sleep(10);status(state)
                code=child.returncode
            state['child_pid']=None
            if code or not (out/'COMPLETE.json').exists():
                raise RuntimeError(f'{job["key"]} failed (exit {code}); see {log.relative_to(ROOT)}')
            state['completed_jobs']=i+1
            event('job_completed',job=job['key'],output=str(out.relative_to(ROOT)))
            status(state)
        state.update(status='gpu_jobs_completed_cpu_analysis_running',current_job='CPU result analysis')
        status(state);event('gpu_jobs_completed')
        analysis=PAPER/'scripts/analyze_corrected_gpu_results.py'
        if not analysis.exists():raise RuntimeError('GPU outputs complete; CPU result-analysis script is not yet installed')
        log=EXEC/'logs/corrected_result_analysis.log'
        with log.open('a') as stream:
            analysis_env=env.copy();analysis_env['USE_TORCH']='0'
            child=subprocess.Popen(['/tmp/cfpb-revision-venv/bin/python',str(analysis)],cwd=ROOT,env=analysis_env,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;status(state)
            while child.poll() is None:time.sleep(10);status(state)
            if child.returncode:raise RuntimeError(f'CPU analysis failed; see {log.relative_to(ROOT)}')
        state.update(status='experiments_and_analysis_completed',current_job=None,child_pid=None,finished_utc=now())
        status(state);event('study_completed',report='analysis/gpu_execution/GPU_RESULTS_REPORT.md')
        with (PAPER/'REVISION_LOG.md').open('a') as f:
            f.write('\n## GPU deneyleri ve analiz — otomatik tamamlanma kaydı\n\nTamamlanma UTC: '+now()+'. Planlanan bütün GPU aşamaları ve CPU sonuç analizi tamamlandı. Kanıt: [GPU_RESULTS_REPORT.md](analysis/gpu_execution/GPU_RESULTS_REPORT.md), [GPU_RUN_STATUS.md](GPU_RUN_STATUS.md) ve RUN_EVENTS.jsonl. Makalenin nihai metin/PDF ve yazar/görsel kontrolü ayrıca doğrulanmalıdır; bu kayıt hakemlere gönderim değildir.\n')
    except BaseException as error:
        state.update(status='failed' if not isinstance(error,KeyboardInterrupt) else 'interrupted',error=str(error))
        status(state);event('orchestrator_stopped',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()

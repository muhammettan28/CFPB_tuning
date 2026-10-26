#!/usr/bin/env python3
"""Resume the authorised revision task after both experiment chains validate."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from revision_gpu_common import ROOT,PAPER,CONFIG,PROTOCOL_SHA,json_text,sha_file,verify_data

OUT=PAPER/'analysis/postprocessing'
SETTINGS=OUT/'AUTO_POSTPROCESS_CONFIG.json'
CORE=PAPER/'analysis/gpu_execution'
EXTRA=PAPER/'analysis/ref14_encoder'
DONE=OUT/'POSTPROCESS_VALIDATION.json'


def read(path):return json.loads(path.read_text())


def readiness():
    """No date assumption; all four independently written milestones are needed."""
    if (OUT/'CANCEL_AUTO_START').exists():return False,'cancelled_by_marker'
    for path,field in [(CORE/'GPU_RUN_STATUS.json','core'),(EXTRA/'RUN_STATUS.json','reference')]:
        if not path.exists():return False,field+'_status_missing'
        status=read(path)['status']
        if status!='experiments_and_analysis_completed':return False,field+':'+status
    core_path=CORE/'GPU_RESULT_VALIDATION.json';extra_path=EXTRA/'RESULT_VALIDATION.json'
    if not core_path.exists() or not extra_path.exists():return False,'validation_reports_missing'
    core,extra=read(core_path),read(extra_path)
    if (core.get('gpu_experiments_completed') is not True or core.get('all_88_gpu_jobs_complete') is not True
            or core.get('protocol_sha256')!=PROTOCOL_SHA or core.get('plan_sha256')!=sha_file(CONFIG)):
        return False,'core_validation_inconsistent'
    if (extra.get('completed') is not True or extra.get('all_12_gpu_jobs_complete') is not True
            or extra.get('configuration_sha256')!=sha_file(PAPER/'configs/ref14_encoder_v1.json')):
        return False,'reference_validation_inconsistent'
    return True,'both_experiment_chains_validated'


def verify_evidence():
    verify_data(include_final=True)
    from run_gpu_study import jobs
    from run_ref14_baselines import config,job_path
    for job in jobs():
        if read(Path(job['out'])/'COMPLETE.json').get('completed') is not True:
            raise RuntimeError('Core completion marker missing/false')
    for action in ['fit','evaluation']:
        for model in config()['models']:
            for seed in [42,43,44]:
                if read(job_path(action,model,seed)/'COMPLETE.json').get('completed') is not True:
                    raise RuntimeError('Reference completion marker missing/false')
    for report in [CORE/'GPU_RESULT_SOURCES.json',EXTRA/'RESULT_SOURCES.json']:
        for entry in read(report):
            target=(ROOT/entry['path']).resolve()
            if not target.is_relative_to(ROOT.resolve()) or sha_file(target)!=entry['sha256']:
                raise RuntimeError('Prediction source hash differs: '+entry['path'])
    protected=read(PAPER/'analysis/revision_v1/source_integrity_before.json')
    for name,digest in protected.items():
        if sha_file(ROOT/name)!=digest:raise RuntimeError('Protected original changed: '+name)


def finished():
    if not DONE.exists():return False,'completion_receipt_not_written'
    info=read(DONE)
    required=['draft_package_ready_for_author_review','manuscript_integrated','supplementary_created',
        'response_letter_created','pdf_visual_qa_passed','originals_unchanged','package_hashes_verified']
    if not all(info.get(k) is True for k in required):return False,'completion_checks_not_all_passed'
    if info.get('uploaded') is not False or info.get('author_final_review_pending') is not True:
        return False,'final_review_or_upload_state_inconsistent'
    required_paths={f'REVISION_PAPER/{name}' for name in ['manuscript.tex','manuscript.pdf',
        'supplementary_material.tex','supplementary_material.pdf','RESPONSE_TO_REVIEWERS.md',
        'response_to_reviewers.tex','response_to_reviewers.pdf']}
    artifacts=info.get('artifacts_sha256',{})
    if not required_paths.issubset(artifacts) or not any(name.endswith('.zip') for name in artifacts):
        return False,'required_deliverable_hashes_missing'
    for name,digest in artifacts.items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(PAPER.resolve()) or not target.is_file() or sha_file(target)!=digest:
            return False,'deliverable_hash_differs:'+name
    for name,digest in read(PAPER/'analysis/revision_v1/source_integrity_before.json').items():
        if sha_file(ROOT/name)!=digest:return False,'protected_original_changed:'+name
    return True,'validated_draft_package_ready_for_author_review'


def write_status(status,**details):
    path=OUT/'RUN_STATUS.json';info=read(path) if path.exists() else {}
    info.update(status=status,watcher_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**details)
    temp=path.with_suffix('.json.tmp');temp.write_text(json_text(info));temp.replace(path)
    (OUT/'AUTO_POSTPROCESS_STATUS.md').write_text('# Deney sonrası otomatik revizyon\n\n'+
        f"- Durum: **{status}**\n- Son kayıt (UTC): {info['updated_utc']}\n- İzleyici PID: {os.getpid()}\n"+
        f"- Açıklama: {info.get('reason','—')}\n- Devam süreci PID: {info.get('continuation_pid') or '—'}\n"+
        '\nBaşlama koşulu: ana ve Ref [14] deney/analiz zincirleri başarıyla tamamlanmış ve kaynak hash kontrolleri geçmiş olmalı. Tarih tek başına başlama koşulu değildir.\n\n'+
        'Görev: [AUTO_POSTPROCESS_TASK.md](AUTO_POSTPROCESS_TASK.md). Yalnız POSTPROCESS_VALIDATION.json kontrolleri ve çıktı hashleri geçtiğinde yazar incelemesine hazır sayılır. Harici gönderim yapılmaz.\n')


def command(settings,attempt):
    executable=Path(settings['codex_cli'])
    if not executable.is_file():
        candidate=shutil.which('codex')
        if not candidate:raise RuntimeError('Codex CLI is unavailable')
        executable=Path(candidate)
    child_thread=settings.get('continuation_thread_id')
    operation='fork' if settings.get('continuation_mode')=='fork' and not child_thread else 'resume'
    return [str(executable),'--no-daemon','--ask-for-approval','never','--sandbox','workspace-write',
        '--cd',str(ROOT),'exec',operation,child_thread or settings['thread_id'],'-',
        '--json','--output-last-message',str(OUT/'logs'/f'last_message_{attempt}.md')]


def started_thread(path):
    """Read only the structured session id, never infer success from CLI exit."""
    if not path.exists():return None
    with path.open() as stream:
        for i,line in enumerate(stream):
            if i>=1000:break
            try:item=json.loads(line)
            except ValueError:continue
            if item.get('type')=='thread.started':return item.get('thread_id')
    return None


def continue_revision(settings):
    if not (PAPER/'AUTO_POSTPROCESS_TASK.md').is_file():
        raise RuntimeError('The completed automatic drafting task was retired. Use the final document validation tools; no drafting continuation was started.')
    previous=read(OUT/'RUN_STATUS.json') if (OUT/'RUN_STATUS.json').exists() else {}
    attempts=int(previous.get('attempts',0))
    for attempt in range(attempts+1,settings.get('max_attempts',3)+1):
        if (OUT/'CANCEL_AUTO_START').exists():write_status('cancelled',reason='cancelled_by_marker');return
        prompt=('Kullanıcının 23 Ekim 2026 tarihli açık otomatik devam isteği için tamamlanma tetikleyicisi. '
            'Ana 88 aşama ve Ref [14] ek 12 aşama ile analizler artık başarıyla tamamlandı; izleyici kaynak hashlerini kontrol etti. '
            'Kaynak konuşmadaki daha yeni kullanıcı düzeltme veya iptal talimatlarını öncelikle uygula. '
            'REVISION_PAPER/AUTO_POSTPROCESS_TASK.md dosyasını oku ve oradaki makale, supplementary, hakem mektubu, PDF kontrolü ve paket görevlerini şimdi tamamla. '
            'ORJ_PAPER ve orijinal veri/deneylere dokunma. Yeni GPU deneyi, yayınlama veya mesaj gönderimi yapma. '
            'Bütün sayıları gerçek doğrulanmış sonuçlardan al. Yarım işi tamamlanmış sayma. '
            'Bu izleyici ana/ek koordinatör kilitlerini tutuyor; bu beklenen davranış, yeni GPU koordinatörü başlatma. '
            'Kısmi önceki düzenlemeler varsa koruyarak devam et. '
            'Görevi tam bitirdiğinde belgede tanımlanan analysis/postprocessing/POSTPROCESS_VALIDATION.json kanıtını yaz. '
            f'Bu devam denemesi {attempt}.')
        path=OUT/'logs'/f'codex_resume_{attempt}.jsonl'
        with path.open('a') as output:
            child=subprocess.Popen(command(settings,attempt),cwd=ROOT,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,text=True)
            write_status('revision_work_running',reason='context_preserving_Codex_followup',continuation_pid=child.pid,attempts=attempt,exit_code=None)
            with (PAPER/'REVISION_LOG.md').open('a') as log:
                log.write('\n## Deney sonrası otomatik devam başlatıldı\n\nUTC: '+datetime.now(timezone.utc).isoformat()+f'. Deneme {attempt}; konuşma bağlamı korunan devam görevi, PID {child.pid}. Kaynak/başlama kontrolleri geçti. Bu başlangıç kaydı belge çalışmasının tamamlandığı anlamına gelmez.\n')
            child.stdin.write(prompt);child.stdin.close()
            while child.poll() is None:
                time.sleep(15)
                session=started_thread(path)
                if session and not settings.get('continuation_thread_id'):
                    settings['continuation_thread_id']=session
                    temp=SETTINGS.with_suffix('.json.tmp');temp.write_text(json_text(settings));temp.replace(SETTINGS)
                write_status('revision_work_running',reason='context_preserving_Codex_followup',continuation_pid=child.pid,attempts=attempt,continuation_thread_id=settings.get('continuation_thread_id'))
        passed,reason=finished()
        if child.returncode==0 and passed:
            write_status('draft_package_ready_for_author_review',reason=reason,continuation_pid=None,exit_code=0)
            return
        write_status('continuation_incomplete',reason=reason,continuation_pid=None,exit_code=child.returncode)
        time.sleep(30)
    write_status('needs_attention',reason='continuation_attempts_finished_without_validated_package',continuation_pid=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    ready,reason=readiness()
    if not a.execute:print(json_text({'ready':ready,'reason':reason,'would_run_model':False}));return
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    own=(OUT/'coordinator.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if (OUT/'CANCEL_AUTO_START').exists():write_status('cancelled',reason='cancelled_by_marker');return
        try:ready,reason=readiness()
        except (ValueError,OSError,KeyError) as error:ready,reason=False,'status_read_error:'+str(error)
        if ready:break
        write_status('waiting_for_experiments',reason=reason,continuation_pid=None)
        time.sleep(30)
    core_lock=(CORE/'orchestrator.lock').open('a+')
    extra_lock=(EXTRA/'coordinator.lock').open('a+')
    while True:
        acquired=[]
        try:
            for lock in [core_lock,extra_lock]:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired.append(lock)
            break
        except BlockingIOError:
            for lock in acquired:fcntl.flock(lock,fcntl.LOCK_UN)
            if (OUT/'CANCEL_AUTO_START').exists():write_status('cancelled',reason='cancelled_by_marker');return
            write_status('waiting_for_coordinator_exit',reason='completion_written_but_lock_still_held')
            time.sleep(15)
    try:
        ready,reason=readiness()
        if not ready:raise RuntimeError('Completion state changed before launch: '+reason)
        verify_evidence()
        passed,reason=finished()
        if passed:write_status('draft_package_ready_for_author_review',reason=reason,continuation_pid=None);return
        continue_revision(read(SETTINGS))
    except BaseException as error:
        write_status('blocked',reason=str(error),continuation_pid=None);raise


if __name__=='__main__':main()

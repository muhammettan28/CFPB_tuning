#!/usr/bin/env python3
"""Validate the CPU preparation artifacts, scientific guards and protected sources."""
from __future__ import annotations
from project_paths import project_path
import ast
from collections import Counter
import io
import json
import re
import unittest
from pathlib import Path
from revision_gpu_common import (CONFIG, EXP, PAPER, PREP, ROOT, PROTOCOL_SHA,
                                json_text, sha_file, verify_data)


def main():
    verify_data(include_final=True)
    before=json.loads((project_path(PAPER, 'analysis/revision_v1/source_integrity_before.json')).read_text())
    for name,expected in before.items():
        if sha_file(project_path(ROOT, name))!=expected: raise RuntimeError(f'Original source changed: {name}')
    for p in (project_path(PAPER, 'scripts')).glob('*.py'): compile(p.read_text(),str(p),'exec')
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.discover(str(project_path(PAPER, 'scripts')),pattern='test_cpu_revision.py'),
                             loader.discover(str(project_path(PAPER, 'scripts')),pattern='test_gpu_preparation.py')])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (PREP/'CPU_PREPARATION_TESTS.txt').write_text(stream.getvalue())
    if not result.wasSuccessful(): raise RuntimeError(stream.getvalue())
    coverage=json.loads((PREP/'reviewer_comment_coverage.json').read_text())
    if sha_file(project_path(ROOT, coverage['source']))!=coverage['source_sha256']:
        raise RuntimeError('Original reviewer source changed since extraction')
    assert len(coverage['records'])==15 and coverage['subrequests']==40
    statuses=Counter(item['status'] for r in coverage['records'] for item in r['checks'])
    for record in coverage['records']:
        for item in record['checks']:
            assert item['status'] in {'completed_cpu','prepared_pending_gpu','open'}
            for evidence in item['evidence']: assert (project_path(PAPER, evidence)).exists(),evidence
    completed=list(EXP.glob('**/COMPLETE.json'))
    assert not completed,'CPU preparation must not create real GPU completion artifacts'
    pdf=json.loads((project_path(PAPER, 'analysis/revision_v1/PDF_VALIDATION.json')).read_text())
    for name,key in [('manuscript.pdf','pdf_sha256'),('manuscript.tex','tex_sha256')]:
        assert sha_file(project_path(PAPER, name))==pdf[key],name
    files=[p for p in (project_path(PAPER, 'scripts')).glob('*.py') if p.name in {
        'prepare_gpu_revision.py','revision_gpu_common.py','audit_gpu_tokenizers.py',
        'run_revision_gpu.py','prepare_reviewer_crosswalk.py','test_gpu_preparation.py','validate_gpu_preparation.py','verify_model_downloads.py'}]
    audits={}
    for model in ['qwen','llama','bert','distilbert','roberta']:
        path=PREP/f'input_audit_{model}.json'
        if path.exists():
            audit=json.loads(path.read_text())
            assert audit['protocol_sha256']==PROTOCOL_SHA and audit['plan_sha256']==sha_file(CONFIG)
            assert audit['common_code_sha256']==sha_file(project_path(PAPER, 'scripts/revision_gpu_common.py'))
            assert audit['audit_code_sha256']==sha_file(project_path(PAPER, 'scripts/audit_gpu_tokenizers.py'))
            audits[model]={'passed':audit['passed'],'rows':audit['rows'],'commit':audit['artifact']['commit']}
    report={'date':'2026-10-20','scope':'CPU preparation and recorded actual-tokenizer audits only; CUDA/model execution remains unrun.',
            'tests_run':result.testsRun,'tests_passed':True,'original_files_unchanged':len(before),
            'protocol_sha256':PROTOCOL_SHA,'plan_sha256':sha_file(CONFIG),
            'reviewer_requests':15,'reviewer_subrequests':40,'subrequest_status_counts':dict(statuses),
            'commands_in_catalogue':len((PREP/'commands_dry_run.txt').read_text().splitlines()),
            'gpu_command_defaults_cpu_tested':83,'actual_tokenizer_audits_performed':len(audits)==5,
            'actual_tokenizer_audits':audits,
            'cached_tokenizer_examples_scope':'10 fixed training examples for each archived Qwen-pilot/Llama tokenizer; prefix/mask serialization only, not final artifact provenance.',
            'cuda_smoke_test_performed':False,'real_gpu_completion_artifacts':0,'gpu_started':False,
            'existing_manuscript_pdf_source_unchanged':True,'code_sha256':{p.name:sha_file(p) for p in files}}
    (PREP/'CPU_PREPARATION_VALIDATION.json').write_text(json_text(report))
    for name in ['README.md','REVISION_LOG.md','RESPONSE_TO_REVIEWERS.md','REVIEWER_COMMENT_CROSSWALK.md','protocols/GPU_EXPERIMENT_PROTOCOL.md','protocols/REF14_BASELINE_PROTOCOL.md']:
        doc=project_path(PAPER, name)
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' not in link and not (doc.parent/link.split('#')[0]).exists():
                raise RuntimeError(f'{name}: missing local link {link}')
    print(json_text(report))


if __name__=='__main__':main()

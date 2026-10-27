#!/usr/bin/env python3
"""Build and verify a local reviewer ZIP; no upload or external sharing."""
from project_paths import project_path
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from prepare_revision_protocol import ROOT,STUDY,file_sha
from tune_revision_tfidf import EXP
from audit_revision_data import OUT

DEST=project_path(ROOT, 'reviewer_package')


def main(author_review=False, package_name=None):
    # The study is complete: every invocation now builds the current clean package.
    # Keep the old call signature for analysis scripts, but never rebuild a draft ZIP.
    author_review=True
    assert (STUDY/'FINAL_CPU_VALIDATION.json').is_file()
    assert (STUDY/'PDF_VALIDATION.json').is_file()
    DEST.mkdir(exist_ok=True)
    gpu_validation=json.loads((project_path(ROOT, 'analysis/gpu_execution/GPU_RESULT_VALIDATION.json')).read_text())
    gpu_complete=gpu_validation.get('gpu_experiments_completed',False)
    assert gpu_complete
    for name in ['MANUSCRIPT_VALIDATION.json','SUPPLEMENTARY_VALIDATION.json','RESPONSE_VALIDATION.json','CROSS_DOCUMENT_VALIDATION.json']:
        assert json.loads((project_path(ROOT, 'analysis/postprocessing', name)).read_text())['passed'],name
    readme="""# Consumer complaint classification: documents and reproducibility evidence

The three documents are under REVISION_PAPER:
- manuscript.pdf: manuscript (12 pages).
- supplementary_material.pdf: S1-S8, 33 tables (19 pages).
- response_to_reviewers.pdf: point-by-point responses to 15 comments (7 pages).
Editable sources and a reviewer-comment location index are included.

The experiment evidence is under experiments/, with statistical results in
analysis/, data selections in dataset/, recorded settings in configs/ and
execution/verification tools in scripts/revision/. The main comparison uses
500 narrative characters and a 512-token cap. Reference [14] conditions are
reported separately; full-data TF-IDF uses a larger labelled training budget.
Auxiliary analyses specify their fitting populations and dependence limitations.

## Verify and replay on CPU

From the extracted archive root:

```bash
sha256sum -c MANIFEST.sha256
python -m pip install -r scripts/revision/requirements-cpu-revision.txt
CUDA_VISIBLE_DEVICES='' python scripts/revision/replay_reviewer_results.py --output /tmp/cfpb-review-replay
python scripts/revision/verify_comparison_evidence.py --include-documents --output /tmp/cfpb-review-comparisons.json
python scripts/revision/audit_document_rows.py --output /tmp/cfpb-review-table-cells.json
python scripts/revision/verify_restored_document_content.py --output /tmp/cfpb-review-content.json
```

The replay output directory must not already exist. Replay checks 404 source
hashes and recomputes nine complete result tables, including 10,000-draw paired
intervals, seed scores, length effects and latency. It compares row identities,
row order and numeric cells at absolute tolerance 1e-12. No GPU, neural weights
or model registry are required. Document checks use the publication sources
included in this archive; public-repository replay works without them.

The archive includes fixed selections, actual search records, predictions,
timing calls, configurations and code/environment evidence. Neural weights and
optimizer states must be obtained or regenerated separately. Repeating neural
execution requires the exact recorded base-model snapshots and a separate
checkout with empty experiment output directories. Completed outputs are
protected against overwrite. Replay verifies saved results rather than repeating
training, inference or upstream data collection.

Data and model licences remain those of their sources. Upstream preprocessing
and some auxiliary-model identities are unavailable; the documents state these
limits separately from the recorded primary-model identities. Device-specific
timing boundaries and the conditional scope of the statistical intervals are
reported explicitly.

Code, data selections and experiment evidence are available at
https://github.com/muhammettan28/CFPB_tuning. See docs/REPRODUCIBILITY.md for the
full verification instructions and recorded-path mapping.
"""
    pages={stem:int(re.search(r'Pages:\s+(\d+)',subprocess.check_output(
        ['pdfinfo',str(ROOT/'REVISION_PAPER'/f'{stem}.pdf')],text=True)).group(1))
        for stem in ['manuscript','supplementary_material','response_to_reviewers']}
    supplement_tables=len(re.findall(r'\\begin\{(?:longtable|table)\}',(ROOT/'REVISION_PAPER/supplementary_material.tex').read_text()))
    # Derive publication sizes from the current PDFs instead of retaining draft counts.
    readme=readme.replace('(12 pages)',f"({pages['manuscript']} pages)")
    readme=readme.replace('33 tables (19 pages)',f"{supplement_tables} tables ({pages['supplementary_material']} pages)")
    readme=readme.replace('(7 pages)',f"({pages['response_to_reviewers']} pages)")
    (DEST/'README.md').write_text(readme)
    files=set()
    retired={'GPU_RUN_STATUS.md','REF14_BASELINE_STATUS.md','AUTO_POSTPROCESS_STATUS.md',
        'AUTO_POSTPROCESS_TASK.md','GPU_NEXT_STEPS.md','SUPPLEMENTARY_PLAN.md','REVISION_PROGRESS.md',
        'REVISION_LOG.md','TWO_PASS_REVIEW.md','reviewer_comment_coverage.json','POSTPROCESS_VALIDATION.json',
        'EXECUTION_QUEUE.json','ollama_inventory_user_reported_2026-10-20.json','commands_dry_run.txt',
        'GPU_SMOKE_VALIDATION.json','PADDING_FIX_RESTART_2026-10-24.json'}
    def add(path):
        if not path.is_file():return
        relative=path.relative_to(ROOT)
        if path.name in retired or path.name.endswith('_STATUS.json'):return
        if 'analysis' in relative.parts:
            if any(part in {'qa','qa_final','qa_pass_1','qa_pass_2','two_pass_review','before_audit','logs'} or part.startswith('before_') for part in relative.parts):return
            if path.suffix=='.md' and path.name not in {'GPU_RESULTS_REPORT.md','RESULTS_REPORT.md'}:return
            if path.name.endswith('_build_final.txt'):return
        files.add(path)
    for name in ['manuscript.tex','manuscript.pdf','supplementary_material.tex','supplementary_material.pdf','response_to_reviewers.tex','response_to_reviewers.pdf','elektr.cls','elksty.tex','elk-creative.eps','elk-creative-eps-converted-to.pdf','RESPONSE_TO_REVIEWERS.md','REVIEWER_COMMENT_CROSSWALK.md']:
        add(project_path(ROOT, 'REVISION_PAPER', name))
    for p in (project_path(ROOT, 'protocols')).glob('*.md'):add(p)
    add(project_path(ROOT, 'templates/revision_paper/manuscript_baseline.tex'))
    add(ROOT/'README.md')
    add(ROOT/'experiments/README.md')
    for p in (ROOT/'docs').glob('*.md'):add(p)
    for p in (ROOT/'analysis/repository_layout').glob('*.json'):add(p)
    for p in (ROOT/'experiments/provenance').rglob('*'):
        if p.suffix in {'.py','.json','.md','.txt','.sh'}:add(p)
    for p in (ROOT/'analysis/document_labels').glob('*.labels'):add(p)
    for directory in ['scripts/revision','scripts','CLASSIC_TEST','slurm_files']:
        for p in (project_path(ROOT, directory)).glob('*'):
            if p.suffix in {'.py','.txt','.sh','.slurm','.md'}:add(p)
    add(project_path(ROOT, 'requirements-truba.txt'))
    for p in (project_path(ROOT, 'dataset')).rglob('*'):
        if p.suffix in {'.csv','.json','.zip'}:add(p)
    for p in (project_path(ROOT, 'dataset/revision_v1')).rglob('*'):
        if p.suffix in {'.csv','.gz'}:add(p)
    for base in [OUT,STUDY,project_path(ROOT, 'analysis/literature'),project_path(ROOT, 'analysis/gpu_preparation')]:
        for p in base.glob('*'):
            if p.suffix in {'.csv','.json','.md','.sha256','.diff','.txt'}:add(p)
    for p in (project_path(ROOT, 'configs')).glob('*'):
        if p.suffix in {'.json','.sha256'} and '.local.' not in p.name:add(p)
    for p in (project_path(ROOT, 'analysis/gpu_preparation')).glob('input_keys_*.csv.gz'):
        add(p)
    execution=project_path(ROOT, 'analysis/gpu_execution')
    if execution.exists():
        for p in execution.rglob('*'):
            relative=p.relative_to(execution)
            if any(part in {'smoke','logs','before_gpu_2026-10-21','failed_attempts'} for part in relative.parts):continue
            if p.name in {'GPU_RUN_STATUS.json','GPU_AUTHORIZATION_2026-10-21.json'}:continue
            if p.suffix in {'.csv','.json','.md','.txt','.sha256'}:add(p)
        from run_gpu_study import jobs as gpu_jobs
        for job in gpu_jobs():
            if job['action']=='smoke':continue
            out=Path(job['out'])
            if not (out/'COMPLETE.json').exists():continue
            for p in out.rglob('*'):
                if 'checkpoints' in p.relative_to(out).parts and p.name!='trainer_state.json':continue
                if p.suffix in {'.csv','.gz','.json','.txt','.py','.sha256'}:add(p)
    for p in (project_path(ROOT, 'REVISION_PAPER/tables')).glob('table_revision_*.tex'):add(p)
    for p in (project_path(ROOT, 'REVISION_PAPER/tables')).glob('table_corrected_*.tex'):add(p)
    for p in (project_path(ROOT, 'REVISION_PAPER/tables/final')).glob('*.tex'):add(p)
    if author_review:
        # Compiled label evidence allows location checks without rebuilding TeX.
        for name in ['manuscript.labels','supplementary_material.labels']:
            add(ROOT/'analysis/document_labels'/name)
    reference_analysis=project_path(ROOT, 'analysis/ref14_encoder')
    for p in reference_analysis.glob('*'):
        if p.name!='RUN_STATUS.json' and p.suffix in {'.json','.md','.csv','.gz','.txt'}:add(p)
    reference_runs=project_path(ROOT, 'experiments/ref14_encoder_v1')
    for marker in reference_runs.rglob('COMPLETE.json'):
        for p in marker.parent.rglob('*'):
            if p.suffix in {'.csv','.gz','.json','.txt','.py','.sha256'}:add(p)
    for p in (project_path(ROOT, 'REVISION_PAPER/figures')).glob('*'):
        if p.name in {'corrected_confusion_matrices.pdf','corrected_confusion_matrices.png'}:add(p)
    for p in EXP.rglob('*'):
        if p.suffix in {'.csv','.json','.sha256','.gz','.joblib'}:add(p)
    for entry in json.loads((OUT/'prediction_sources.json').read_text()):
        p=project_path(ROOT, entry['path']);add(p);add(p.parent/'metrics.json')
    for p in (project_path(ROOT, 'experiment_archive/outputs')).glob('*/*selection_manifest.csv'):add(p)
    for p in (project_path(ROOT, 'experiment_archive/outputs')).glob('*/run_metadata.json'):add(p)
    for p in (project_path(ROOT, 'experiment_archive/environment')).glob('*'):add(p)
    for directory in [
        'experiment_archive/outputs/qwen3_8b_qlora_pilot_5952287/best_adapter',
        'experiment_archive/outputs/llama31_8b_fixed_protocol_seed42_5994012_0/best_adapter']:
        for p in (project_path(ROOT, directory)).glob('*'):
            if p.suffix in {'.json','.jinja','.txt'}:add(p)
    for p in (project_path(ROOT, 'experiment_archive/outputs')).glob('*.csv'):add(p)
    if author_review:
        post=project_path(ROOT, 'analysis/postprocessing')
        for name in ['corrected_class_metrics.csv','corrected_off_label_counts.csv','corrected_legacy_per_view.csv','corrected_confusion_counts.npz','long_input_per_variant.csv','DOCUMENT_NUMERICAL_SOURCES.json','RESPONSE_LOCATIONS.json','FINAL_RESPONSE_LOCK.json','MANUSCRIPT_VALIDATION.json','SUPPLEMENTARY_VALIDATION.json','RESPONSE_VALIDATION.json','CROSS_DOCUMENT_VALIDATION.json','manuscript_build_final.txt','supplement_build_final.txt','response_build_final.txt']:
            add(post/name)
        add(post/'KEYED_TABLE_VALIDATION.json')
        add(post/'REVIEWER_COVERAGE_FINAL.json')
        add(post/'RESTORED_CONTENT_VALIDATION.json')
        for name in ['qlora_encoder_contrasts.csv','computational_costs.csv','cpu_latency_summary.csv','training_runtime_evidence.json','training_search_runtime_evidence.json','protocol_comparison_evidence.json','tokenization_path_audit.json','cpu_inference_precision_check.json','COMPARISON_EXTENSION_VALIDATION.json']:
            add(post/name)
        for p in (ROOT/'experiments/cpu_costs_v1').rglob('*'):
            if p.suffix in {'.csv','.json'}:add(p)
    members={str(p.relative_to(ROOT)):p for p in sorted(files)}
    manifest={name:file_sha(path) for name,path in members.items()}
    members.pop('README.md',None)
    manifest.pop('README.md',None)
    manifest['README.md']=file_sha(DEST/'README.md')
    manifest_text='\n'.join(f'{digest}  {name}' for name,digest in sorted(manifest.items()))+'\n'
    (DEST/'MANIFEST.sha256').write_text(manifest_text)
    if author_review:
        name=package_name or 'CFPB_revision_reviewer_2026-10-26_layout.zip'
        if Path(name).name!=name or not name.endswith('.zip'):
            raise ValueError('Package name must be a plain ZIP filename')
        path=DEST/name
        if path.exists():
            raise FileExistsError('Preserve previous final package; choose a fresh filename: '+str(path))
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        z.writestr('README.md',readme)
        z.writestr('MANIFEST.sha256',manifest_text)
        for name,source in members.items():z.write(source,name)
    with zipfile.ZipFile(path) as z:
        for name,expected in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==expected,name
        assert len(z.infolist())==len(manifest)+1
    report={'path':str(path.relative_to(ROOT)),'files_checked':len(manifest),'sha256':file_sha(path),
            'bytes':path.stat().st_size,'all_member_hashes_passed':True,'uploaded':False,
            'blind_review_anonymity_certified':False,'base_or_adapter_weights_included':False}
    report['phase']='clean_reviewer_package'
    report['temporary_plans_status_files_and_internal_drafting_logs_excluded']=True
    (DEST/'PACKAGE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--author-review',action='store_true')
    parser.add_argument('--package-name',help='Fresh author-review ZIP filename; never overwrites an existing archive.')
    args=parser.parse_args()
    main(args.author_review,args.package_name)

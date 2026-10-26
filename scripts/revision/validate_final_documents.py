#!/usr/bin/env python3
"""Check frozen scientific evidence and final numerical/reference consistency, without fitting."""
from project_paths import project_path
from pathlib import Path
import json,re,subprocess,sys
import pandas as pd
from revision_gpu_common import ROOT,PAPER,sha_file
from auto_postprocess_after_experiments import verify_evidence,readiness
from prepare_reviewer_crosswalk import extract,final_response_protected
from revision_document_paths import compiled_artifact
from publication_language import internal_workflow_hits

OUT=project_path(PAPER, 'analysis/postprocessing')
def read(p):return json.loads(p.read_text())
def main():
    ready,reason=readiness();assert ready,reason
    verify_evidence()
    numerical=read(OUT/'DOCUMENT_NUMERICAL_SOURCES.json')
    for entry in numerical:assert sha_file(project_path(ROOT, entry['path']))==entry['sha256'],entry['path']
    for stem,report,texkey,pdfkey in [('manuscript','MANUSCRIPT_VALIDATION.json','manuscript_tex_sha256','manuscript_pdf_sha256'),('supplementary_material','SUPPLEMENTARY_VALIDATION.json','supplementary_tex_sha256','supplementary_pdf_sha256')]:
        d=read(OUT/report);assert d['passed']
        assert sha_file(project_path(PAPER, stem+'.tex'))==d[texkey]
        assert sha_file(project_path(PAPER, stem+'.pdf'))==d[pdfkey]
    assert read(OUT/'RESPONSE_VALIDATION.json')['passed'] and final_response_protected()
    locations=read(OUT/'RESPONSE_LOCATIONS.json')
    for doc,digest in locations['source_sha256'].items():assert sha_file(compiled_artifact(doc,'.aux'))==digest
    for entry in locations['references']:
        pattern=r'\\newlabel\{'+re.escape(entry['label'])+r'\}\{\{'+re.escape(entry['number'])+r'\}\{'+str(entry['page'])+r'\}'
        assert re.search(pattern,compiled_artifact(entry['document'],'.aux').read_text()),entry
    _,_,r1,r2=extract();response=(project_path(PAPER, 'RESPONSE_TO_REVIEWERS.md')).read_text()
    for quote in r1+r2:assert '> '+quote in response
    assert len(re.findall(r'^### R[12]\.\d+',response,re.M))==15
    # Check cells in their model/condition/metric rows; mere presence of a number
    # elsewhere in a document cannot establish a correct table association.
    keyed_report=OUT/'KEYED_TABLE_VALIDATION.json'
    subprocess.run([sys.executable,str(project_path(PAPER, 'scripts/audit_document_rows.py')),
                    '--output',str(keyed_report)],check=True)
    keyed=read(keyed_report);assert keyed['passed']
    restored_report=OUT/'RESTORED_CONTENT_VALIDATION.json'
    subprocess.run([sys.executable,str(project_path(PAPER, 'scripts/verify_restored_document_content.py')),
                    '--output',str(restored_report)],check=True)
    restored=read(restored_report);assert restored['passed']
    metrics=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_model_metrics.csv'))
    contrasts=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_paired_contrasts.csv'))
    main=(project_path(PAPER, 'manuscript.tex')).read_text();sup=(project_path(PAPER, 'supplementary_material.tex')).read_text()
    tables={role:(project_path(PAPER, f'tables/final/{role}_models.tex')).read_text() for role in ['internal','legacy']}
    point_checks=[]
    for r in metrics.loc[metrics.metric.eq('macro_f1')].itertuples(index=False):
        for value in [r.point,r.low,r.high]:assert f'{value:.4f}' in tables[r.role],(r.system,value)
        point_checks.append({'role':r.role,'system':r.system,'macro_f1':r.point})
    for r in contrasts.loc[contrasts.role.eq('internal')&contrasts.metric.eq('macro_f1')].itertuples(index=False):
        table=(project_path(PAPER, 'tables/final/internal_contrasts.tex')).read_text()
        for v in [r.paired_difference,r.low,r.high]:assert f'{v:.4f}' in table,(r.comparison,v)
    for system in ['qwen_base','qwen_ft','llama_base','llama_ft','tfidf_10000','tfidf_full']:
        point=metrics.loc[metrics.role.eq('internal')&metrics.system.eq(system)&metrics.metric.eq('macro_f1'),'point'].item()
        assert f'{point:.4f}' in main,(system,point)
        if not system.endswith('_base'):assert f'{point:.4f}' in response,(system,point)
    long=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_long_input.csv'))
    long=long.loc[long.metric.eq('macro_f1')&long.comparison.eq('500_to_1500_chars')]
    for r in long.itertuples(index=False):
        table=(project_path(PAPER, 'tables/final/long_macro.tex')).read_text()
        for v in [r.reference_point,r.comparison_point,r.paired_difference,r.low,r.high]:assert f'{v:.4f}' in table
    latency=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/controlled_latency_paired.csv'))
    for r in latency.loc[latency.condition.eq('chars500_tokens512')&latency.timing.eq('end_to_end_seconds')].itertuples(index=False):
        for v in [r.ft_seed_mean_minus_base_seconds,r.low,r.high]:assert f'{v:.4f}' in response
    # The source row identities, rounded values and separate CI/SD estimates are checked;
    # independent source-prediction metric checks are in the completed scientific validation.
    for text in [main,sup,response]:
        assert not internal_workflow_hits(text),internal_workflow_hits(text)
        for pattern in ['TODO','TBD','INSERT HERE','see pending','[final section]']:
            assert pattern not in text,pattern
    for stem in ['manuscript','supplementary_material','response_to_reviewers']:
        log=compiled_artifact(stem,'.log').read_text()
        assert not re.search(r'Overfull \\[hv]box|There were undefined|Citation .* undefined|Reference .* undefined',log),stem
        assert 'Label(s) may have changed' not in log and 'Table widths have changed' not in log,stem
        scratch=project_path(PAPER, 'build/validation_text');scratch.mkdir(parents=True,exist_ok=True)
        subprocess.run(['pdftotext',str(project_path(PAPER, stem+'.pdf')),str(scratch/(stem+'_text.txt'))],check=True)
    # TeX can turn a literal double hyphen into an en dash. A documented shell
    # command must survive PDF copy/paste, not merely look plausible on the page.
    pdf_text=(project_path(PAPER, 'build/validation_text/supplementary_material_text.txt')).read_text()
    assert 'python scripts/revision/replay_reviewer_results.py' in pdf_text
    assert '--output /tmp/cfpb-review-replay' in pdf_text
    assert '–output' not in pdf_text
    protected=read(project_path(PAPER, 'analysis/revision_v1/source_integrity_before.json'))
    report={'passed':True,'date':'2026-10-24','main_gpu_stages':88,'reference_gpu_stages':12,'source_hashes_verified':True,'numerical_source_files_checked':len(numerical),'protected_originals_checked':len(protected),'protected_ORJ_PAPER_files':sum(k.startswith('ORJ_PAPER/') for k in protected),'compiled_response_locations_checked':len(locations['references']),'exact_request_units_checked':15,'all_model_macro_f1_points_and_bounds_checked':point_checks,'main_internal_contrasts_checked':True,'long_input_table_checked':True,'latency_response_checked':True,'document_pdf_hashes_match_visual_qa':True,'no_placeholders_or_unresolved_tex_references':True,'legacy_response_generators_preserve_final':True,'irrecoverable_limitations':['Upstream CFPB-to-Kaggle preprocessing/product aggregation and original sampling implementation unavailable','Historical immutable upstream identities/final Qwen artifacts unavailable','Retrospective source-related evaluation, conservative text groups do not guarantee semantic independence','Intervals condition on fitted/selected models; no multiplicity correction','Serial batch-one latency does not establish deployment acceptability or concurrent capacity']}
    report['date']='2026-10-26'
    report['keyed_tables_checked']=len(keyed['tables'])
    report['keyed_rows_checked']=keyed['total_rows']
    report['keyed_cells_checked']=keyed['total_cells']
    report['wrong_row_negative_controls_passed']=keyed['row_swap_and_wrong_value_negative_controls_passed']
    report['pdf_copyable_ascii_replay_command_checked']=True
    report['internal_workflow_notes_absent_from_all_documents']=True
    report['historical_selection_conditions_checked']=restored['historical_selection_conditions_checked']
    report['restored_error_narrative_checked_against_raw_predictions']=True
    report['original_introduction_opening_preserved']=True
    (OUT/'CROSS_DOCUMENT_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: frozen evidence,',len(protected),'originals,',len(locations['references']),'locations, numerical/source/PDF checks')
if __name__=='__main__':main()

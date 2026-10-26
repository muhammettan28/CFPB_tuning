#!/usr/bin/env python3
"""Extract original reviewer text and maintain sub-request coverage. CPU only."""
from __future__ import annotations
from project_paths import project_path
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from revision_gpu_common import PAPER, PREP, ROOT, json_text, sha_file

D='analysis/revision_v1/'
A='analysis/cpu_revision/'
P='analysis/gpu_preparation/'


FINAL_RESPONSE_LOCK=project_path(PAPER, 'analysis/postprocessing/FINAL_RESPONSE_LOCK.json')

def final_response_protected():
    """Fail closed on unexpected edits; legacy generators cannot replace final prose."""
    if not FINAL_RESPONSE_LOCK.exists():return False
    info=json.loads(FINAL_RESPONSE_LOCK.read_text())
    for name,digest in info['artifacts_sha256'].items():
        if sha_file(project_path(PAPER, name))!=digest:
            raise RuntimeError('Final response changed; preserve and deliberately review lock: '+name)
    return True


def gpu_results_ready():
    path=project_path(PAPER, 'analysis/gpu_execution/GPU_RESULT_VALIDATION.json')
    return path.exists() and json.loads(path.read_text()).get('gpu_experiments_completed',False)


def check(title,status,evidence,remaining=''):
    return {'request':title,'status':status,'evidence':evidence,'remaining':remaining}


def extract():
    source=project_path(ROOT, 'Revision.odt')
    if source.exists():
        with ZipFile(source) as z: tree=ET.fromstring(z.read('content.xml'))
        ns='{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'
        for element in tree.iter():
            if element.tag==ns+'line-break': element.text='\n'
            if element.tag==ns+'tab': element.text='\t'
        paras=[''.join(e.itertext()) for e in tree.iter() if e.tag in {ns+'p',ns+'h'}]
        start=next(i for i,p in enumerate(paras) if p.startswith('Reviewer #1:'))
        original='\n'.join(paras[start:]).strip()+'\n'
        archived=PREP/'original_reviewer_comments.txt'
        if archived.exists() and archived.read_text()!=original:
            raise RuntimeError('Archived original comments differ from the source ODT')
    else:
        # Standalone reviewer archive retains exact original comments, not the
        # author's mixed planning notes elsewhere in the source ODT.
        source=PREP/'original_reviewer_comments.txt'
        original=source.read_text()
    r1,r2=original.split('Reviewer #2:',1)
    lines=[x.strip() for x in r1.splitlines() if x.strip() and not x.startswith('Reviewer #1:')]
    if len(lines)!=8: raise RuntimeError('Original R1 comment structure changed; inspect manually')
    starts=['The authors should strengthen','The term “external benchmark”','The manuscript would also benefit',
            'The statistical evaluation','Claims concerning constrained decoding','Most importantly, the results show',
            'Finally, improving public access']
    sentences=re.split(r'(?<=[.!?])\s+',r2.strip())
    r2_comments=[next(s for s in sentences if s.startswith(prefix)) for prefix in starts]
    return source,original,lines,r2_comments


def coverage(r1,r2):
    completed='completed_cpu'
    pending='prepared_pending_gpu'
    open_='open'
    groups=[
      [check('Quantify overlap in the 12 legacy subsets',completed,[A+'legacy_overlap_summary.json',A+'legacy_overlap_narrative_counts.csv',A+'legacy_overlap_label_narrative_counts.csv']),
       check('Use group-aware paired analysis; remove unsupported subset independence',completed,[A+'statistical_method.json',A+'paired_group_bootstrap.csv','manuscript.tex'])],
      [check('Audit the actual truncated inputs and prompt construction',completed,[A+'leakage_overlap_summary.csv',A+'tokenizer_audit_evidence.json'], 'Historical Qwen scope limited to pilot tokenizer; actual new pins must pass the prepared audit.'),
       check('Narrow the no-leakage claim',completed,['manuscript.tex']),
       check('Repair training allocation and repeat affected experiments',pending,[D+'protocol.json','scripts/audit_gpu_tokenizers.py','scripts/run_revision_gpu.py'],'Actual tokenizer audit and corrected-data GPU reruns have not been executed.')],
      [check('Explain the local 162,415-row derivation',completed,[D+'source_archive_trace.json',D+'source_placeholder_exclusions.csv']),
       check('Give the label mapping',completed,[D+'label_identity_mapping.csv'],'Pre-Kaggle raw-product aggregation code remains unavailable; inherited five labels are documented.'),
       check('Describe subset construction and preserved files',completed,[D+'legacy_source_archive_checks.json',D+'legacy_subset_class_counts.csv','manuscript.tex'],'Original sampling code is not recovered; frozen files and manifests are supplied locally.'),
       check('Provide code, manifests and predictions to reviewers',open_,['reviewer_package/README.md','reviewer_package/PACKAGE_VALIDATION.json'],'Local package exists; an actual reviewer access mechanism is still required.')],
      [check('State the Qwen thinking setting',completed,[A+'archived_implementation_details.json','manuscript.tex']),
       check('State immutable checkpoint IDs',open_,[A+'archived_implementation_details.json','configs/model_registry.template.json'],'Historical exact upstream commits/final Qwen artifacts are missing; future registry entries are unresolved.'),
       check('State full QLoRA configuration',completed,[A+'archived_implementation_details.json','manuscript.tex']),
       check('State library versions',completed,[A+'archived_implementation_details.json','manuscript.tex'],'New GPU environment must be recorded separately; historical versions do not certify desktop compatibility.')],
      [check('Tune TF-IDF using validation',completed,['experiments/tfidf_revision_v1/budget_10000/validation_search.csv','experiments/tfidf_revision_v1/budget_full/validation_search.csv']),
       check('Tune encoders using validation and repeat optimisation seeds',pending,['scripts/run_revision_gpu.py','configs/gpu_revision_v1.json'],'Six candidates/model and three refits are coded; no encoder GPU training/search is completed.'),
       check('Explain three-seed evidence fairly',completed,[D+'CPU_REVISION_COMPLETION_REPORT.md','manuscript.tex'],'TF-IDF lbfgs repetitions are deterministic checks, not stochastic seed evidence.'),
       check('One corrected-data comparison table with uncertainty; discuss small margins',pending,['tables/table_revision_tfidf.tex','manuscript.tex'],'Await matched corrected-data encoder/QLoRA predictions; do not rank old and new training pools together.')],
      [check('Paired group-level bootstrap with more replicates',completed,[A+'statistical_method.json',A+'paired_group_bootstrap.csv']),
       check('Report run/seed variation separately',completed,[A+'per_seed_metrics.csv','manuscript.tex']),
       check('Reconcile point estimates and confidence intervals in old Tables 2/3',completed,['scripts/build_revision_analysis.py','tables/table_revision_archived_internal.tex','tables/table_revision_archived_ci.tex'], 'Final table numbers are automatic LaTeX references; future GPU results need the same estimator.')],
      [check('Explicit comparison with [14]',completed,['analysis/literature/REF14_POSITIONING.md','manuscript.tex']),
       check('Explicit comparison with [15]',completed,['analysis/literature/REF14_POSITIONING.md','manuscript.tex']),
       check('Remove or test taxonomy-reuse claims',completed,['manuscript.tex']),
       check('Reconsider external benchmark terminology',completed,['manuscript.tex']),
       check('Moderate broad robustness claims',completed,['manuscript.tex'])],
      [check('Report truncation-related results',completed,[D+'paired_truncation_metrics.csv',D+'truncation_method.json']),
       check('Add a controlled longer-context experiment',pending,['configs/gpu_revision_v1.json','protocols/GPU_EXPERIMENT_PROTOCOL.md','scripts/run_revision_gpu.py'],'500/512, 500/2048, 1500/2048 same-checkpoint conditions are prepared; inference not performed.'),
       check('Control latency measurements',pending,[P+'latency_sample_manifest.csv','protocols/GPU_EXPERIMENT_PROTOCOL.md','scripts/run_revision_gpu.py'],'100 matched records, 10 warmups and three timed repetitions are prepared; no controlled latency result exists.'),
       check('Remove acceptability/regulatory-readiness claims',completed,['manuscript.tex'])]
    ]
    r2groups=[
      [check('Position against [14] and distinguish prior LLM complaint classification',completed,['analysis/literature/REF14_POSITIONING.md','manuscript.tex'])],
      [check('Reconsider external benchmark from the same CFPB source',completed,['manuscript.tex'])],
      [check('Complete QLoRA and inference configuration description',completed,[A+'archived_implementation_details.json','manuscript.tex'],'Exact historical checkpoint gaps remain open under R1.4.'),
       check('Justify 500-character truncation',completed,[D+'source_length_summary.json','manuscript.tex'],'Resource choice, not an optimality claim; controlled longer-input results remain pending.'),
       check('Justify 10,000-example training budget',completed,[D+'protocol.json','manuscript.tex'],'Fixed resource budget, not a demonstrated sample-efficiency optimum.')],
      [check('Clarify statistical evaluation and bootstrap methodology',completed,[A+'statistical_method.json','manuscript.tex'])],
      [check('Moderate constrained-decoding claim',completed,['manuscript.tex']),
       check('Moderate robustness claim',completed,['manuscript.tex']),
       check('Moderate deployment claim',completed,['manuscript.tex'],'Controlled latency work remains pending, not replaced by softened wording.')],
      [check('Frame contribution as PEFT benefits/trade-offs; acknowledge near TF-IDF performance',completed,['manuscript.tex','tables/table_revision_tfidf.tex'],'Final matched corrected-data comparison awaits GPU results.')],
      [check('Improve public/reviewer code and configuration access',open_,['reviewer_package/README.md','reviewer_package/PACKAGE_VALIDATION.json'],'Local artifacts are prepared; actual access/public release has not occurred.')]
    ]
    records=[]
    for reviewer,originals,checks in [(1,r1,groups),(2,r2,r2groups)]:
        for i,(original,items) in enumerate(zip(originals,checks),1):
            records.append({'id':f'R{reviewer}.{i}','original_comment':original,'checks':items})
    download_report=PREP/'MODEL_DOWNLOAD_VALIDATION.json'
    if download_report.exists():
        for record in records:
            if record['id']=='R1.4':
                record['checks'][1]['remaining']='Historical exact upstream commits/final Qwen artifacts remain missing. Prospective cache commits are now recorded; see MODEL_DOWNLOAD_VALIDATION.json for incomplete weights. New pins do not recover historical identity.'
            if record['id']=='R1.2':
                record['checks'][2]['remaining']='Prospective tokenizer audit status is recorded in analysis/gpu_preparation/input_audit_*.json; corrected-data GPU reruns remain unexecuted.'
    if (project_path(PAPER, 'analysis/gpu_execution/GPU_RUN_STATUS.json')).exists():
        for record in records:
            if record['id']=='R1.2':record['checks'][2]['remaining']='Prospective tokenizer audits passed; corrected-data GPU study status is in GPU_RUN_STATUS.md.'
            if record['id']=='R1.5':record['checks'][1]['remaining']='The six-candidate-per-encoder study is running under the fixed protocol; completion evidence is recorded per job.'
    if gpu_results_ready():
        for record in records:
            for item in record['checks']:
                if item['status']=='prepared_pending_gpu':
                    item['status']='completed_gpu'
                    item['remaining']='Completed corrected-data results and validation are recorded in analysis/gpu_execution/GPU_RESULTS_REPORT.md and GPU_RESULT_VALIDATION.json. Final manuscript/author review remains separate.'
                    item['evidence'].append('analysis/gpu_execution/GPU_RESULT_VALIDATION.json')
    if final_response_protected():
        for record in records:
            for item in record['checks']:
                item['evidence'] += ['RESPONSE_TO_REVIEWERS.md','supplementary_material.pdf']
                item['remaining']=''
                if item['status']=='completed_gpu':
                    item['remaining']='Experiments and document integration verified; final author approval remains.'
            if record['id']=='R1.2':
                record['checks'][0]['remaining']='New pinned-tokenizer and longer-input audits passed; historical Qwen pilot-tokenizer scope remains limited.'
            if record['id']=='R1.3':
                record['checks'][1]['remaining']='Inherited labels documented; upstream CFPB-to-Kaggle extraction/product aggregation unavailable.'
                record['checks'][2]['remaining']='Frozen files match source archive; original sampling implementation unavailable.'
                record['checks'][3]['remaining']='Hash-verified local package prepared; external reviewer/public access is not arranged.'
            if record['id']=='R1.4':
                record['checks'][1]['status']='partial_historical_gap'
                record['checks'][1]['remaining']='All new immutable commits/weights verified and reported; missing historical upstream revisions/final Qwen artifacts remain unavailable.'
            if record['id']=='R2.3':
                record['checks'][0]['remaining']='New implementation complete; historical identity gaps disclosed.'
                record['checks'][1]['remaining']='Inherited 500-character convention with measured longer-input effects; no optimality/hardware-necessity claim.'
                record['checks'][2]['remaining']='Common fixed 10k/2k comparison budget; no sample-efficiency optimum claimed.'
            if record['id']=='R2.7':record['checks'][0]['remaining']='Local package complete; actual external access and author/journal submission checks remain.'
    return records


R2_BODIES=[
('Positioning against Reference [14]', 'Text revision completed; author review pending',
 'We explicitly acknowledge that Güllü et al. already include financial complaint classification and Llama-3.1 fine-tuning. We distinguish our within-model zero-shot-to-NF4-QLoRA protocol, fixed adaptation budget and trade-offs, without claiming the first LLM application, a new corpus/taxonomy or direct cross-paper score superiority. The comparison also separates [15] zero-shot evaluation from adaptation.', 'R1.7; analysis/literature/REF14_POSITIONING.md'),
('Meaning of external benchmark', 'Terminology corrected',
 'We now describe the twelve files as preserved, source-related legacy evaluation views. They originate from the related CFPB resource and do not establish independent-source generalisation. Their repeated inputs are handled with explicit overlap analysis and group resampling.', 'R1.1/R1.7; manuscript.tex data and evaluation sections'),
('QLoRA/inference details and 500-character/10k choices', 'Available configuration and budget rationale documented; artifact and GPU checks pending',
 'The Methods specify the recorded QLoRA, thinking and generation settings, library versions and missing checkpoint provenance. The 500-character cap and 10,000-example budget are resource-controlled choices, not empirically optimal settings. We report the actual length distribution and use identical fixed 10k/2k memberships for revised model families. A prospective same-checkpoint 500/512, 500/2048 and 1500/2048 protocol is prepared; its outcomes have not been measured. The token-cap control and explicit no-duplicate-special-token serialization apply to new runs and are not presented as a reconstruction of all historical runs.', 'R1.3/R1.4/R1.8; protocols/GPU_EXPERIMENT_PROTOCOL.md; configs/gpu_revision_v1.json'),
('Statistical evaluation and bootstrap', 'CPU reanalysis completed',
 'We define the estimands, weights and label-blind dependency groups, use 10,000 paired group-bootstrap draws with shared multiplicities, and calculate reported point estimates on the observed data. Across-run sample SD is separate from resampling uncertainty. Historical seeds also changed training selections; fixed-model bootstrap intervals do not include training or model-selection uncertainty.', 'R1.1/R1.6; analysis/cpu_revision/statistical_method.json'),
('Constrained decoding, robustness and deployment', 'Claims moderated; controlled measurements pending',
 'The inference description now states greedy JSON generation followed by strict regex label parsing; it is not token-level constrained decoding. We removed unsupported general robustness, taxonomy-reuse, acceptability and regulatory-readiness claims. Controlled latency instrumentation and an outcome-independent 100-record cohort are prepared, but no new deployment measurements are claimed.', 'R1.7/R1.8; protocols/GPU_EXPERIMENT_PROTOCOL.md'),
('Practical contribution relative to TF-IDF + Logistic Regression', 'Framing and stronger CPU baseline completed; matched GPU comparison pending',
 'We frame the study around the measured benefits, costs and evaluation limits of within-model PEFT adaptation. Validation-selected TF-IDF/LR reaches 0.8068 internal macro-F1 at the common 10k budget and 0.8476 with the full corrected training pool. We explicitly acknowledge baseline competitiveness and avoid comparing these newly trained models with archived LLMs as a final fair ranking. Corrected-data encoder/QLoRA runs are still required.', 'R1.5; analysis/revision_v1/tfidf_group_bootstrap.csv'),
('Code and experimental-configuration access', 'Local package prepared; actual access open',
 'We have prepared and hash-verified a local package containing code, source/derived data, configurations, manifests and predictions. The revision log and exact-comment coverage audit are included. It has not been uploaded or supplied to reviewers; the final response and Data Availability statement must name a real access mechanism before submission. Missing upstream preprocessing/checkpoint evidence is documented rather than invented.', 'R1.3/R1.4; reviewer_package/README.md; reviewer_package/PACKAGE_VALIDATION.json')]


def refresh_response():
    if final_response_protected():
        print('Final response preserved by FINAL_RESPONSE_LOCK.json')
        return
    path=project_path(PAPER, 'RESPONSE_TO_REVIEWERS.md')
    text=path.read_text().split('## Reviewer #2')[0].rstrip()
    note='**Original-comment coverage:** [REVIEWER_COMMENT_CROSSWALK.md](REVIEWER_COMMENT_CROSSWALK.md) preserves the exact reviewer requests and tracks each sub-request. GPU preparation is documented in [protocols/GPU_EXPERIMENT_PROTOCOL.md](protocols/GPU_EXPERIMENT_PROTOCOL.md); prepared code is not a completed experiment.'
    if note not in text: text=text.replace('\n20 October 2026.', '\n'+note+'\n\n20 October 2026.',1)
    # Update the base response without claiming GPU experiments are complete.
    old='Comparable encoder tuning and three-seed corrected-data GPU runs are still pending; archived and new-training scores are kept separate.'
    new='A separate fixed-data encoder runner with six validation candidates per model and three selected-configuration refits is prepared and CPU-checked. Comparable encoder tuning and three-seed corrected-data GPU runs have not been executed; archived and new-training scores are kept separate.'
    text=text.replace(old,new)
    text=text.replace('A longer-context GPU experiment and warmup/repetition-controlled latency study remain pending.',
        'A three-condition same-checkpoint longer-input protocol and a 100-record, 10-warmup, three-repetition synchronized latency protocol are now prepared with CPU-checked execution code. The actual longer-context inference and controlled latency measurements remain pending.')
    # Update the current artifact status without rewriting historical provenance claims.
    text=re.sub(r'\n<!-- MODEL_DOWNLOAD_STATUS_START -->.*?<!-- MODEL_DOWNLOAD_STATUS_END -->\n','\n',text,flags=re.S)
    download=PREP/'MODEL_DOWNLOAD_VALIDATION.json'
    audited=[m for m in ['qwen','llama','bert','distilbert','roberta']
             if (PREP/f'input_audit_{m}.json').exists() and json.loads((PREP/f'input_audit_{m}.json').read_text())['passed']]
    if download.exists():
        state=json.loads(download.read_text())
        date=datetime.strptime(state['date'],'%Y-%m-%d').strftime('%d %B %Y')
        note=f'\n<!-- MODEL_DOWNLOAD_STATUS_START -->\n**Prospective artifact verification ({date}):** Immutable cache identities are recorded for all five planned models; {state["complete_models"]}/5 currently have complete checked weights. CPU token-input audits have passed for {len(audited)}/5 prospective tokenizers. This does not recover missing historical checkpoint identity. See [download verification](analysis/gpu_preparation/MODEL_DOWNLOAD_VALIDATION.json) and `analysis/gpu_preparation/input_audit_*.json`.\n<!-- MODEL_DOWNLOAD_STATUS_END -->\n'
        text=text.replace('\n## R1.1:',note+'\n## R1.1:',1)
    text=re.sub(r'\n<!-- GPU_EXECUTION_STATUS_START -->.*?<!-- GPU_EXECUTION_STATUS_END -->\n','\n',text,flags=re.S)
    if (project_path(PAPER, 'analysis/gpu_execution/GPU_RUN_STATUS.json')).exists():
        ready=gpu_results_ready()
        body=('The corrected-data GPU experiments, paired uncertainty, long-input and controlled-latency analyses are completed; see [results](analysis/gpu_execution/GPU_RESULTS_REPORT.md). Final manuscript integration and author review remain.' if ready else 'GPU execution was authorized on 21 October 2026. All five hardware/API smoke tests passed; the full study is in progress. See [live status](GPU_RUN_STATUS.md). Partial validation results are not a final test ranking.')
        note='\n<!-- GPU_EXECUTION_STATUS_START -->\n**Current GPU study status:** '+body+'\n<!-- GPU_EXECUTION_STATUS_END -->\n'
        text=text.replace('\n## R1.1:',note+'\n## R1.1:',1)
        text=text.replace('Comparable encoder tuning and three-seed corrected-data GPU runs have not been executed; archived and new-training scores are kept separate.',
            'The corrected-data GPU study is in progress; full results will be reported only after the fixed runs and checks complete. Archived and new-training scores are kept separate.')
        text=text.replace('New GPU runs must record these artifacts.',
            'New GPU runs record exact prospective artifact identities and their environment; these new pins do not recover missing historical identities.')
    text=re.sub(r'\n<!-- REF14_BASELINE_STATUS_START -->.*?<!-- REF14_BASELINE_STATUS_END -->\n','\n',text,flags=re.S)
    if (project_path(PAPER, 'configs/ref14_encoder_v1.json')).exists():
        validation=project_path(PAPER, 'analysis/ref14_encoder/RESULT_VALIDATION.json')
        ready=validation.exists() and json.loads(validation.read_text()).get('completed',False)
        status=('All six additional encoder refits and six evaluations are complete, with reload-inference parity and paired group-bootstrap checks; see [results](analysis/ref14_encoder/RESULTS_REPORT.md).' if ready else 'The additional six refits and six evaluations are queued separately behind the core GPU study; they are not yet completed results. See [live status](REF14_BASELINE_STATUS.md).')
        note='\n<!-- REF14_BASELINE_STATUS_START -->\n**Additional literature-parameter sensitivity:** Tables III–IV of Güllü et al. specify BERT (LR 2e-5, 3 epochs, batch 16, max 128 tokens) and RoBERTa (LR 5e-5, 3 epochs, batch 16, max 100 tokens). These learning-rate/epoch pairs already won the main validation search. We predeclared separate shorter-cap, physical-batch-16 refits with seeds 42/43/44, keeping the fixed 10k/2k selections and reporting every condition without test-based baseline selection. '+status+' This is a parameter-sensitivity extension, not exact replication: our split, 500-character inputs, PyTorch implementation and explicitly inherited execution defaults differ. [Protocol](protocols/REF14_BASELINE_PROTOCOL.md).\n<!-- REF14_BASELINE_STATUS_END -->\n'
        text=text.replace('\n## R1.1:',note+'\n## R1.1:',1)
    text=re.sub(r'\n<!-- SUPPLEMENTARY_PRESENTATION_PLAN_START -->.*?<!-- SUPPLEMENTARY_PRESENTATION_PLAN_END -->\n','\n',text,flags=re.S)
    if (project_path(PAPER, 'SUPPLEMENTARY_PLAN.md')).exists():
        note='\n<!-- SUPPLEMENTARY_PRESENTATION_PLAN_START -->\n**Presentation plan agreed on 22 October 2026:** Complete all planned main and reference-parameter experiments. Keep the central matched comparisons, key methods and limitations in the manuscript; present full searches, per-seed results, legacy dependence, input-length/latency details and reference-parameter sensitivity in a readable supplementary report. Responses will connect each reviewer request to a verified action, result and exact manuscript/supplement location, regardless of result direction. This is a presentation plan, not a claim that pending experiments or supplementary files are complete. See [SUPPLEMENTARY_PLAN.md](SUPPLEMENTARY_PLAN.md).\n<!-- SUPPLEMENTARY_PRESENTATION_PLAN_END -->\n'
        text=text.replace('\n## R1.1:',note+'\n## R1.1:',1)
    text=re.sub(r'\n<!-- AUTO_POSTPROCESS_PLAN_START -->.*?<!-- AUTO_POSTPROCESS_PLAN_END -->\n','\n',text,flags=re.S)
    if (project_path(PAPER, 'AUTO_POSTPROCESS_TASK.md')).exists():
        note='\n<!-- AUTO_POSTPROCESS_PLAN_START -->\n**Post-experiment drafting workflow:** The author authorized automatic continuation on 23 October 2026. A local completion-gated watcher waits for both experiment/analysis chains and their evidence checks, then continues in a context-preserving local Codex follow-up session to integrate manuscript results, create the supplementary report and finalize the response letter. This automation is not proof that those documents are finished; validated draft readiness is recorded separately. [Task](AUTO_POSTPROCESS_TASK.md) · [Status](AUTO_POSTPROCESS_STATUS.md).\n<!-- AUTO_POSTPROCESS_PLAN_END -->\n'
        text=text.replace('\n## R1.1:',note+'\n## R1.1:',1)
    text+='\n\n## Reviewer #2\n\nThe following responses address each request in the original comment separately. Original wording and sub-request status are preserved in the linked coverage audit.\n'
    for i,(title,status,body,evidence) in enumerate(R2_BODIES,1):
        text+=f'\n### R2.{i}: {title}\n\n**Status:** {status}.\n\n{body}\n\n**Evidence:** `{evidence}`.\n'
    text+='''\n## Submission gates still open

- Corrected-data QLoRA/encoder training and validation search with the fixed selections and three optimisation seeds; CPU-prepared code is not GPU-validated.
- Actual tokenizer audits for immutable snapshots, controlled input beyond 500 characters and controlled latency measurements.
- Missing historical immutable checkpoint/tokenizer evidence; a new pin does not recover historical provenance.
- Matched final corrected-data comparison and resulting Abstract/Conclusion/response updates.
- Actual reviewer/public access arrangement and final author review.
'''
    if len(audited)==5:
        text=text.replace('- Actual tokenizer audits for immutable snapshots, controlled input beyond 500 characters and controlled latency measurements.',
                          '- Complete missing model weight downloads; controlled input beyond 500 characters and controlled latency measurements. Actual prospective tokenizer CPU audits have passed.')
        if download.exists() and json.loads(download.read_text())['all_models_complete']:
            text=text.replace('- Complete missing model weight downloads; controlled input beyond 500 characters and controlled latency measurements. Actual prospective tokenizer CPU audits have passed.',
                              '- Controlled input beyond 500 characters and controlled latency measurements. All five weight sets and prospective tokenizer CPU audits are verified.')
    if gpu_results_ready():
        text=text.replace('**Status:** Audit and revised allocation completed; GPU reruns pending.', '**Status:** Audit, revised allocation and corrected-data GPU reruns completed.')
        text=text.replace('New model results are not claimed until corrected-data GPU training is repeated; Qwen tokenizer provenance remains explicitly limited.',
                          'Corrected-data GPU predictions and paired results are now available separately from the archived runs. New artifact identities are pinned; historical Qwen provenance limitations remain explicit.')
        text=text.replace('**Status:** TF-IDF completed; corrected-data encoder/LLM comparison pending GPU.', '**Status:** Validation-selected TF-IDF/encoders, three-seed GPU refits and corrected-data comparison completed.')
        text=text.replace('The corrected-data GPU study is in progress; full results will be reported only after the fixed runs and checks complete.',
                          'The fixed GPU study and its paired analysis are completed: six candidates per encoder, three selected-configuration refits per encoder, and three QLoRA seeds per LLM. The consolidated results include uncertainty.')
        text=text.replace('**Status:** Retrospective CPU analyses completed; new inference/latency pending GPU.', '**Status:** Retrospective CPU and controlled GPU analyses completed.')
        text=text.replace('The actual longer-context inference and controlled latency measurements remain pending.',
                          'The same-checkpoint longer-input and controlled latency measurements are now completed and recorded in analysis/gpu_execution/.')
        text=text.replace('**Status:** Available configuration and budget rationale documented; artifact and GPU checks pending.', '**Status:** Configuration, budget rationale and prospective GPU checks documented; historical artifact limits explicit.')
        text=text.replace('its outcomes have not been measured.', 'its measured outcomes are recorded in analysis/gpu_execution/corrected_long_input.csv.')
        text=text.replace('**Status:** Claims moderated; controlled measurements pending.', '**Status:** Claims moderated and controlled measurements completed.')
        text=text.replace('but no new deployment measurements are claimed.', 'and the synchronized measurements are now recorded; they do not establish deployment suitability.')
        text=text.replace('**Status:** Framing and stronger CPU baseline completed; matched GPU comparison pending.', '**Status:** Framing, validation-selected baselines and matched corrected-data GPU comparison completed.')
        text=text.replace('Corrected-data encoder/QLoRA runs are still required.', 'Corrected-data encoder/QLoRA results and paired contrasts are now available in analysis/gpu_execution/.')
        text=text.split('## Submission gates still open')[0]+'''## Submission gates still open

- Final manuscript integration/rendering and author review of the completed experiment results.
- Actual reviewer/public access arrangement; local packages have not been uploaded or supplied.
- Historical upstream preprocessing/checkpoint limitations remain disclosed; prospective exact pins do not recover missing historical identities.
'''
    path.write_text(text)


def write_final_crosswalk():
    """Keep the reviewer index aligned with the locked final responses, not old plans."""
    source,original,r1,r2=extract()
    response=(project_path(PAPER, 'RESPONSE_TO_REVIEWERS.md')).read_text()
    blocks=re.split(r'(?=^### R[12]\.\d+)',response,flags=re.M)[1:]
    records=[]
    for block,quote in zip(blocks,r1+r2):
        ident=re.match(r'### (R[12]\.\d+)',block).group(1)
        assert '> '+quote in block
        locations=block.split('**Changes and locations.** ',1)[1].strip().split('\n\n## Reviewer',1)[0]
        records.append({'id':ident,'original_comment':quote,'locations':locations,
            'status':'public_repository_link_provided' if ident in {'R1.3','R2.7'} else 'addressed_with_stated_limits'})
    assert len(records)==15
    report={'source':str(source.relative_to(ROOT)),'source_sha256':sha_file(source),
            'response_sha256':sha_file(project_path(PAPER, 'RESPONSE_TO_REVIEWERS.md')),'requests':records,
            'request_units':15,'access_method':'Public GitHub repository',
            'public_repository_url':'https://github.com/muhammettan28/CFPB_tuning'}
    final=project_path(PAPER, 'analysis/postprocessing/REVIEWER_COVERAGE_FINAL.json')
    final.write_text(json_text(report))
    md=['# Reviewer requests and final response locations','',
        'This index refers to the completed revised documents. R1/R2 IDs are editorial tracking IDs, not original reviewer numbering. All original wording is retained below; the complete original text, including introductory comments, is in [the original extract](../analysis/gpu_preparation/original_reviewer_comments.txt).','',
        '[Full response letter](RESPONSE_TO_REVIEWERS.md) · [Machine-readable final coverage](../analysis/postprocessing/REVIEWER_COVERAGE_FINAL.json)','',
        'Reproducibility repository: https://github.com/muhammettan28/CFPB_tuning. The responses describe the completed analyses and their scientific limitations; each entry below links the reviewer comment to the corresponding document locations.']
    for r in records:
        md+=['', '## '+r['id'],'','> '+r['original_comment'],'',
             '**Response:** see '+r['id']+' in [the response letter](RESPONSE_TO_REVIEWERS.md).',
             '', '**Changes and locations:** '+r['locations']]
    (project_path(PAPER, 'REVIEWER_COMMENT_CROSSWALK.md')).write_text('\n'.join(md)+'\n')


def main():
    if final_response_protected():
        write_final_crosswalk()
        return
    source,original,r1,r2=extract()
    records=coverage(r1,r2)
    for record in records:
        for item in record['checks']:
            for evidence in item['evidence']:
                if not (project_path(PAPER, evidence)).exists(): raise RuntimeError(f'Evidence missing: {evidence}')
    updated=datetime.now(ZoneInfo('Europe/Istanbul'))
    report={'source':str(source.relative_to(ROOT)),'source_sha256':sha_file(source),
            'date':updated.date().isoformat(),'id_policy':'R1.1-R1.8 and R2.1-R2.7 are editorial tracking IDs, not original numbering.',
            'r1_comments':len(r1),'r2_request_units':len(r2),
            'subrequests':sum(len(r['checks']) for r in records),'records':records,
            'gpu_started':(project_path(PAPER, 'analysis/gpu_execution/GPU_RUN_STATUS.json')).exists(),
            'gpu_results_validated':gpu_results_ready()}
    PREP.mkdir(parents=True,exist_ok=True)
    (PREP/'original_reviewer_comments.txt').write_text(original)
    (PREP/'reviewer_comment_coverage.json').write_text(json_text(report))
    md=['# Original reviewer comments — sub-request coverage','',
        updated.strftime('%d %B %Y')+'. Source: project-root `Revision.odt`, original comments at the end. Earlier GPT-generated notes are not treated as reviewer wording.',
        '',f'ODT SHA-256: `{report["source_sha256"]}`.', '',
        'R1/R2 IDs are our tracking IDs; the original comments were not numbered this way. `completed_cpu` closes only the stated sub-request; `prepared_pending_gpu` means results are not complete; `completed_gpu` requires the final validated GPU result report; `open` remains unresolved; `partial_historical_gap` means new identities are complete but historical identities remain missing. The final response lock protects completed prose from legacy regeneration.',
        '',f'Coverage: {len(r1)} R1 comments, {len(r2)} R2 request units, {report["subrequests"]} sub-requests. Introductory praise and the major-revision recommendation are preserved in the extracted source but are not action items.',
        '', '[Full extracted original text](../analysis/gpu_preparation/original_reviewer_comments.txt) · [Machine-readable audit](analysis/gpu_preparation/reviewer_comment_coverage.json) · [Response draft](RESPONSE_TO_REVIEWERS.md)']
    for record in records:
        md+=['',f'## {record["id"]}','', '> '+record['original_comment'],'',
             '| Sub-request | Status | Evidence | Remaining work / limitation |','| --- | --- | --- | --- |']
        for item in record['checks']:
            links='; '.join(f'[{Path(x).name}]({x})' for x in item['evidence'])
            md.append(f'| {item["request"]} | {item["status"]} | {links} | {item["remaining"] or "—"} |')
    (project_path(PAPER, 'REVIEWER_COMMENT_CROSSWALK.md')).write_text('\n'.join(md)+'\n')
    refresh_response()
    print(f'Covered {len(records)} original request units and {report["subrequests"]} sub-requests; '+
          ('GPU results validated.' if gpu_results_ready() else 'full GPU results remain pending.'))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent dataset, selection and numerical checks for the CPU revision."""
from project_paths import project_path
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

from prepare_revision_protocol import ROOT,DATA,STUDY,file_sha,protected_snapshot
from audit_revision_data import NORM,PROMPT
from tune_revision_tfidf import EXP


def main():
    assert protected_snapshot()==json.loads((STUDY/'source_integrity_before.json').read_text())
    assert file_sha(STUDY/'protocol.json')==(STUDY/'protocol.sha256').read_text().split()[0]
    protocol=json.loads((STUDY/'protocol.json').read_text())
    for name,digest in protocol['files_sha256'].items():assert file_sha(project_path(ROOT, name))==digest,name
    frames={n:pd.read_csv(DATA/f'{n}.csv') for n in ['train','validation','internal_test']}
    frames['legacy']=pd.read_csv(DATA/'legacy_test_manifest.csv')
    independent=[]
    for a,b in [('train','validation'),('train','internal_test'),('validation','internal_test'),('train','legacy'),('validation','legacy')]:
        for kind,limit in [('full',None),('500',500)]:
            left=frames[a].narrative.astype(str);right=frames[b].narrative.astype(str)
            if limit:left=left.str[:limit];right=right.str[:limit]
            common=set(left.map(NORM))&set(right.map(NORM));assert not common,(a,b,kind)
            independent.append({'left':a,'right':b,'representation':kind,'shared':0})
    legacy_prefix=set(frames['legacy'].narrative.astype(str).str[:250].map(NORM))
    for name in ['train','validation']:assert not set(frames[name].narrative.astype(str).str[:250].map(NORM))&legacy_prefix
    bootstrap=pd.read_csv(STUDY/'tfidf_group_bootstrap.csv',dtype={'budget':str})
    metric_checks=[]
    for budget in ['10000','full']:
        dest=EXP/f'budget_{budget}'
        chosen=json.loads((dest/'selected_config.json').read_text())
        assert chosen['protocol_sha256']==file_sha(STUDY/'protocol.json')
        assert file_sha(dest/'selected_config.json')==(dest/'selected_config.sha256').read_text().split()[0]
        search=pd.read_csv(dest/'validation_search.csv')
        assert len(search)==48 and set(search.candidate_id)==set(range(48))
        max_value=search.macro_f1.max()
        winner=search.loc[search.macro_f1.ge(max_value-1e-12)].sort_values('candidate_id').iloc[0]
        assert int(winner.candidate_id)==chosen['candidate_id']
        vec=joblib.load(dest/'selected_vectorizer.joblib');model=joblib.load(dest/'seed_42/classifier.joblib')
        valid=pd.read_csv(DATA/'validation_2000.csv');pred=model.predict(vec.transform(valid.narrative.str[:500]))
        assert np.isclose(f1_score(valid['product'],pred,labels=PROMPT['LABELS'],average='macro'),chosen['validation_metrics']['macro_f1'],atol=1e-12,rtol=0)
        for role in ['internal_test','legacy']:
            pred=pd.read_csv(dest/f'seed_42/{role}_predictions.csv.gz')
            weight=None
            compare_role='internal_test' if role=='internal_test' else 'legacy_unique'
            if role=='legacy':
                assert pred[['source_file','source_row']].equals(frames['legacy'][['source_file','source_row']])
                counts=frames['legacy'].groupby(['true_label',frames['legacy'].narrative.map(NORM)])['source_row'].transform('size')
                weight=1./counts.to_numpy()
            measured={'accuracy':accuracy_score(pred.true_label,pred.prediction,sample_weight=weight),
                'macro_f1':f1_score(pred.true_label,pred.prediction,labels=PROMPT['LABELS'],average='macro',sample_weight=weight,zero_division=0),
                'weighted_f1':f1_score(pred.true_label,pred.prediction,labels=PROMPT['LABELS'],average='weighted',sample_weight=weight,zero_division=0)}
            for metric,value in measured.items():
                row=bootstrap.loc[bootstrap.budget.eq(budget)&bootstrap.role.eq(compare_role)&bootstrap.metric.eq(metric)].iloc[0]
                assert np.isclose(row['mean'],value,atol=1e-12,rtol=0),(budget,role,metric)
                assert row['low']<=row['mean']<=row['high']
                metric_checks.append({'budget':budget,'role':role,'metric':metric,'matches_sklearn':True})
        assert json.loads((dest/'repetition_checks.json').read_text())['identical_predictions_all_three_seeds']
    text=(project_path(ROOT, 'REVISION_PAPER/manuscript.tex')).read_text()
    keys=set(re.findall(r'\\bibitem\{([^}]+)\}',text));cited=set()
    for group in re.findall(r'\\cite\{([^}]+)\}',text):cited.update(group.split(','))
    assert cited<=keys,cited-keys
    for item in re.findall(r'\\input\{([^}]+)\}',text):assert (project_path(ROOT, 'REVISION_PAPER', item)).is_file(),item
    assert 'Revision working draft.' in text
    assert '300 bootstrap' not in text and 'one-sided Wilcoxon' not in text
    assert 'likely acceptable' not in text
    report={'source_integrity':'all original files unchanged','protocol_hash':'matches frozen signature',
        'independent_text_separation_checks':independent,'legacy_prefix_guards':'passed',
        'validation_selection':'48+48 candidates and validation-only winners verified',
        'bootstrap_point_checks_against_sklearn':metric_checks,'lbfgs_determinism_checks':'passed both budgets',
        'citation_keys':len(cited),'latex_input_files':'all exist','gpu_used':False}
    (STUDY/'FINAL_CPU_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':main()

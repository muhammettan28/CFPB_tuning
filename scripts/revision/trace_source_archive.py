#!/usr/bin/env python3
"""Recover source derivation from the existing Dataset.zip without mutation."""
from project_paths import project_path
import hashlib
import io
import json
import zipfile
import pandas as pd

from prepare_revision_protocol import ROOT, STUDY, file_sha
from audit_revision_data import PROMPT


def main():
    archive=project_path(ROOT, 'dataset/Dataset.zip')
    with zipfile.ZipFile(archive) as z:
        content=z.read('complaints.csv')
        raw=pd.read_csv(io.BytesIO(content))
        metadata=z.getinfo('complaints.csv')
        legacy_checks=[]
        for path in sorted((project_path(ROOT, 'dataset/legacy_subsets')).glob('*.csv')):
            digest=hashlib.sha256(z.read(path.name)).hexdigest()
            assert digest==file_sha(path)
            legacy_checks.append({'file':path.name,'sha256':digest,'archive_member_matches_byte_for_byte':True})
    (STUDY/'legacy_source_archive_checks.json').write_text(json.dumps(legacy_checks,indent=2)+'\n')
    legacy=pd.read_csv(project_path(ROOT, 'dataset/splits/legacy_test_manifest.csv'))
    pd.crosstab(legacy.source_file,legacy.true_label).to_csv(STUDY/'legacy_subset_class_counts.csv')
    local=pd.read_csv(project_path(ROOT, 'dataset/complaints.csv'))
    removed=raw.loc[~raw['Unnamed: 0'].isin(local['Unnamed: 0'])]
    pd.testing.assert_frame_equal(raw.iloc[:len(local)].reset_index(drop=True),local.reset_index(drop=True))
    assert len(raw)-len(local)==len(removed)==6
    assert removed.narrative.eq('name').all()
    assert removed['Unnamed: 0'].tolist()==list(range(162415,162421))
    labels=sorted(raw['product'].dropna().unique())
    assert labels==sorted(PROMPT['LABELS'])
    missing=int(local[['product','narrative']].isna().any(axis=1).sum())
    removed.to_csv(STUDY/'source_placeholder_exclusions.csv',index=False)
    report={'attribution':'Project prior zero-shot manuscript, Section 3.1 and reference [30], attributes the processed resource to Shashwat Tiwari on Kaggle, derived from CFPB.',
        'project_prior_source':'literature/ref_paper/Can Open Source LLMs Replace Paid API Services for Zero Shot Consumer Complaint Classification.pdf',
        'dataset_url':'https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp',
        'archive_path':'dataset/Dataset.zip','archive_sha256':file_sha(archive),
        'member':'complaints.csv','member_sha256':hashlib.sha256(content).hexdigest(),
        'member_zip_timestamp':list(metadata.date_time),'archive_rows':len(raw),'working_rows':len(local),
        'excluded_placeholder_rows':6,'excluded_ids':removed['Unnamed: 0'].tolist(),
        'excluded_narrative_literal':'name','all_retained_rows_identical_by_id_label_and_narrative':True,
        'working_missing_rows':missing,'usable_rows':len(local)-missing,
        'labels_already_present_in_source_archive':labels,
        'mapping_statement':'The five canonical product strings are inherited unchanged from the archived processed dataset. The current project performs identity retention, not a new aggregation of original CFPB product categories.',
        'limits':['The files prove the six-row difference; they do not prove when or why it was removed.',
                  'The archive is an already preprocessed artifact, not the full raw CFPB database. Earlier raw-product aggregation and preprocessing code are not in the project.',
                  'Source attribution follows the project prior manuscript. No byte-identity claim is made against a newly downloaded Kaggle version.'],
        'flow':[{'stage':'archived processed source','rows':162421},{'stage':'remove six trailing name-only placeholders','rows':162415},
                {'stage':'drop ten missing required rows','rows':162405},{'stage':'original legacy exclusion (8123 rows)','rows':154282},
                {'stage':'quarantine model-visible overlaps (3232 rows)','rows':151050}]}
    (STUDY/'source_archive_trace.json').write_text(json.dumps(report,indent=2)+'\n')
    names={'retail_banking':'Retail banking','credit_reporting':'Credit reporting','credit_card':'Credit cards',
           'mortgages_and_loans':'Mortgages and loans','debt_collection':'Debt collection'}
    rows=[]
    for label in PROMPT['LABELS']:
        rows.append({'archive_product':label,'analysis_product':label,'meaning':names[label],
                     'transformation':'identity; already assigned in archived processed source'})
    pd.DataFrame(rows).to_csv(STUDY/'label_identity_mapping.csv',index=False)
    tex='\n'.join(r'\texttt{'+r['archive_product'].replace('_',r'\_')+'} & '+r['meaning']+r' \\' for r in rows)+'\n'+r'\hline'+'\n'
    (project_path(ROOT, 'REVISION_PAPER/tables/table_revision_label_mapping.tex')).write_text(tex)
    print(json.dumps({k:report[k] for k in ['archive_rows','working_rows','excluded_ids','all_retained_rows_identical_by_id_label_and_narrative']},indent=2))


if __name__=='__main__':main()

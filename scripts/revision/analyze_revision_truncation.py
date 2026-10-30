#!/usr/bin/env python3
"""Retrospective, text-matched 250/500 legacy views; no new inference."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prepare_revision_protocol import ROOT, DATA, STUDY
from audit_revision_data import OUT, NORM, sha, dependency_groups
from analyze_archived_predictions import load_predictions, SPECS, encode, group_confusions, paired_cluster_bootstrap, METRICS


def match_views(legacy):
    pairs=[];counts=[]
    for (seed,condition),subset in legacy.groupby(['seed','condition'],sort=True):
        a=subset.loc[subset.max_chars.eq(250)].copy();b=subset.loc[subset.max_chars.eq(500)].copy()
        a['key']=a.narrative.astype(str).map(NORM)
        b['key']=b.narrative.astype(str).str[:250].map(NORM)
        keycols=['true_label','key']
        ca=a.groupby(keycols)['source_row'].transform('size');cb=b.groupby(keycols)['source_row'].transform('size')
        # Exclude ambiguous within-file prefix matches instead of choosing a
        # convenient long narrative or assuming missing complaint identities.
        au=a.loc[ca.eq(1)].copy();bu=b.loc[cb.eq(1)].copy()
        au['short_index']=au.index;bu['long_index']=bu.index
        joined=au.merge(bu,on=keycols,suffixes=('_250','_500'),validate='one_to_one')
        counts.append({'seed':int(seed),'condition':condition,'short_rows':len(a),'long_rows':len(b),
                       'unambiguous_pairs':len(joined),'ambiguous_short_rows':int(ca.gt(1).sum()),
                       'ambiguous_long_rows':int(cb.gt(1).sum())})
        for r in joined.itertuples():
            pairs.append({'seed':int(seed),'condition':condition,'short_index':r.short_index,'long_index':r.long_index,
                          'true_label':r.true_label,'short_key':sha(r.key),'long_key':sha(NORM(r.narrative_500)),
                          'short_file':r.source_file_250,'short_row':r.source_row_250,
                          'long_file':r.source_file_500,'long_row':r.source_row_500,
                          'changed_input':NORM(r.narrative_250)!=NORM(r.narrative_500)})
    return pd.DataFrame(pairs),pd.DataFrame(counts)


def main():
    legacy=pd.read_csv(DATA/'legacy_test_manifest.csv')
    pairs,counts=match_views(legacy)
    pairs['group']=dependency_groups(pairs,['short_key','long_key'])
    pairs.to_csv(STUDY/'truncation_pair_manifest.csv',index=False)
    counts.to_csv(STUDY/'truncation_pair_counts.csv',index=False)
    results=[]
    for model in SPECS:
        predictions,_=load_predictions(model,'legacy',legacy)
        for cohort,mask in [('all_matched',np.ones(len(pairs),dtype=bool)),('changed_input_only',pairs.changed_input.to_numpy())]:
            f=pairs.loc[mask].copy()
            repeated=f.groupby(['true_label','short_key','long_key'])['seed'].transform('size')
            weight=1./repeated.to_numpy()
            pred=np.empty((len(f),8),dtype=object)
            pred[:,::2]=predictions[f.short_index.to_numpy()]
            pred[:,1::2]=predictions[f.long_index.to_numpy()]
            cm=group_confusions(encode(f.true_label.to_numpy()),encode(pred),f.group.to_numpy(),weight)
            point,samples=paired_cluster_bootstrap(cm,10000,20261020)
            for variant,indices in [('base_zero_shot',[0]),('fine_tuned_seed_mean',[1,2,3])]:
                short_indices=np.array(indices)*2;long_indices=short_indices+1
                short=point[short_indices].mean(axis=0);long=point[long_indices].mean(axis=0)
                delta=samples[:,long_indices].mean(axis=1)-samples[:,short_indices].mean(axis=1)
                low,high=np.quantile(delta,[.025,.975],axis=0)
                for k,metric in enumerate(METRICS):
                    results.append({'model':model,'cohort':cohort,'variant':variant,'metric':metric,
                         'pair_appearances':len(f),'unique_pairs':len(f[['true_label','short_key','long_key']].drop_duplicates()),
                         'groups':len(cm),'short_point':short[k],'long_point':long[k],'gain':long[k]-short[k],
                         'gain_low':low[k],'gain_high':high[k]})
    f=pd.DataFrame(results);f.to_csv(STUDY/'paired_truncation_metrics.csv',index=False)
    (STUDY/'truncation_method.json').write_text(json.dumps({
        'data':'Archived predictions only. No newly evaluated 1000/1500-character or token-budget model.',
        'pairing':'Within same sampling seed and balanced/imbalanced condition; same true label and normalized raw250 prefix; exactly one row per key on both sides. Ambiguous matches excluded before examining predictions.',
        'identity_limit':'A unique prefix match is not a verified CFPB complaint ID. Results describe the matched-view subset, not the entire corpus or a prospective randomized ablation.',
        'weighting':'Equal total weight per distinct (label, short normalized view, long normalized view) pair across repeated appearances. Connected short/long-view groups resampled together, labels ignored for clustering.',
        'bootstrap_replicates':10000,'rng_seed':20261020,
        'internal_length_strata_limit':'Different complaints/class distributions form these strata; observed differences are not a causal estimate of truncation harm.',
        'matches':counts.to_dict(orient='records')},indent=2)+'\n')
    print(f.loc[f.cohort.eq('changed_input_only') & f.metric.eq('macro_f1')].to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

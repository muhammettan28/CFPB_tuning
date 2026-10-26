#!/usr/bin/env python3
"""Generate manuscript-ready tables from archived audits and new CPU fits."""
from __future__ import annotations
from project_paths import project_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prepare_revision_protocol import ROOT, DATA, STUDY, file_sha
from audit_revision_data import OUT, NORM
from analyze_archived_predictions import encode, group_confusions, paired_cluster_bootstrap, METRICS
from tune_revision_tfidf import EXP

TABLES = project_path(ROOT, "REVISION_PAPER/tables")
FIGURES = project_path(ROOT, "REVISION_PAPER/figures")


def tex_rows(name, rows):
    (TABLES / name).write_text("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n" + r"\hline" + "\n")


def interval(point, low, high):
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def archived_tables():
    stats = pd.read_csv(OUT / "paired_group_bootstrap.csv")
    rows = []
    for model, subset in stats.loc[stats.analysis.eq("internal_original")].groupby("model", sort=False):
        m = subset.set_index("metric")
        rows.append([model, f"{m.loc['macro_f1','base_point']:.4f}",
            f"{m.loc['macro_f1','ft_seed_mean_point']:.4f}$\\pm${m.loc['macro_f1','ft_seed_sd_ddof1']:.4f}",
            f"{m.loc['macro_f1','paired_gain_point']:.4f}", f"{m.loc['weighted_f1','base_point']:.4f}",
            f"{m.loc['weighted_f1','ft_seed_mean_point']:.4f}$\\pm${m.loc['weighted_f1','ft_seed_sd_ddof1']:.4f}"])
    tex_rows("table_revision_archived_internal.tex", rows)
    rows = []
    for r in stats.loc[stats.analysis.eq("internal_original")].itertuples():
        rows.append([r.model, r.metric.replace("_", " "), interval(r.base_point,r.base_low,r.base_high),
                     interval(r.ft_seed_mean_point,r.ft_low,r.ft_high), interval(r.paired_gain_point,r.gain_low,r.gain_high)])
    tex_rows("table_revision_archived_ci.tex", rows)
    rows = []
    for r in stats.loc[stats.analysis.eq("legacy_unique_input_weighted") & stats.metric.eq("macro_f1")].itertuples():
        rows.append([r.model, f"{r.base_point:.4f}", f"{r.ft_seed_mean_point:.4f}$\\pm${r.ft_seed_sd_ddof1:.4f}",
                     interval(r.paired_gain_point,r.gain_low,r.gain_high)])
    tex_rows("table_revision_archived_legacy.tex", rows)
    rows = []
    for r in stats.loc[stats.analysis.isin(["internal_conservative_clean", "legacy_conservative_prefix_clean"]) & stats.metric.eq("macro_f1")].itertuples():
        name = "Internal sensitivity" if r.analysis.startswith("internal") else "Legacy sensitivity"
        rows.append([r.model, name, r.rows, f"{r.ft_seed_mean_point:.4f}", interval(r.paired_gain_point,r.gain_low,r.gain_high)])
    tex_rows("table_revision_sensitivity.tex", rows)


def baseline_tables():
    internal = pd.read_csv(DATA / "internal_test.csv").rename(columns={"product":"true_label"})
    legacy = pd.read_csv(DATA / "legacy_test_manifest.csv")
    assignments = pd.read_csv(STUDY / "role_assignment_manifest.csv", low_memory=False)
    legacy_groups = pd.read_csv(OUT / "legacy_groups.csv")
    internal["group"] = internal.record_id.map(assignments.loc[assignments.original_role.eq("internal_test")].set_index("record_id").group)
    legacy["group"] = legacy_groups.prefix250_group
    assert legacy[["source_file","source_row","true_label"]].equals(legacy_groups[["source_file","source_row","true_label"]])
    weights = 1. / legacy.groupby(["true_label", legacy.narrative.map(NORM)])["source_row"].transform("size").to_numpy()
    result = []
    for budget in ["10000", "full"]:
        for role, data, weight in [("internal_test", internal, None), ("legacy_unique", legacy, weights)]:
            predictions = []
            for seed in [42,43,44]:
                p = EXP / f"budget_{budget}/seed_{seed}/{('internal_test' if role=='internal_test' else 'legacy')}_predictions.csv.gz"
                f = pd.read_csv(p)
                if role == "internal_test":
                    assert f.record_id.is_unique and set(f.record_id)==set(data.record_id)
                    f = f.set_index("record_id").loc[data.record_id].reset_index()
                else:
                    assert f[["source_file","source_row"]].equals(data[["source_file","source_row"]])
                assert f.true_label.tolist()==data.true_label.tolist()
                predictions.append(f.prediction.to_numpy())
            cm = group_confusions(encode(data.true_label.to_numpy()), encode(np.stack(predictions,axis=1)),data.group.to_numpy(),weight)
            point,samples = paired_cluster_bootstrap(cm,10000,20261020)
            average_samples=samples.mean(axis=1);mean=point.mean(axis=0)
            low,high=np.quantile(average_samples,[.025,.975],axis=0)
            for i,metric in enumerate(METRICS):
                result.append({"budget":budget,"role":role,"metric":metric,"rows":len(data),"groups":len(cm),
                    "mean":mean[i],"seed_sd_ddof1":point[:,i].std(ddof=1),"low":low[i],"high":high[i],"replicates":10000})
    f=pd.DataFrame(result); f.to_csv(STUDY / "tfidf_group_bootstrap.csv",index=False)
    rows=[]
    for budget in ["10000","full"]:
        x=f.loc[f.budget.eq(budget) & f.role.eq("internal_test")].set_index("metric")
        leg=f.loc[f.budget.eq(budget) & f.role.eq("legacy_unique") & f.metric.eq("macro_f1")].iloc[0]
        selection=json.loads((EXP/f"budget_{budget}/selected_config.json").read_text())
        rows.append(["TF-IDF + LR",selection['train_rows'],f"{x.loc['accuracy','mean']:.4f}",
            interval(x.loc['macro_f1','mean'],x.loc['macro_f1','low'],x.loc['macro_f1','high']),
            f"{x.loc['weighted_f1','mean']:.4f}",f"{leg['mean']:.4f}"])
    tex_rows("table_revision_tfidf.tex",rows)
    rows=[]
    for budget in ["10000","full"]:
        d=json.loads((EXP/f"budget_{budget}/selected_config.json").read_text());c=d['config']
        rows.append([d['train_rows'],f"(1,{c['ngram_max']})",c['max_features'],c['min_df'],c['C'],"none" if c['class_weight'] is None else "balanced",f"{d['validation_metrics']['macro_f1']:.4f}"])
    tex_rows("table_revision_tfidf_selection.tex",rows)


def dataset_and_overlap():
    protocol=json.loads((STUDY/"protocol.json").read_text())
    rows=[[r['partition'].replace('_',' '),r['original_rows'],r['quarantined_rows'],r['retained_rows'],r['groups']] for r in protocol['split_sizes']]
    tex_rows("table_revision_data_flow.tex",rows)
    all_counts=[]
    for role in ['train','validation','internal_test']:
        f=pd.read_csv(DATA/f'{role}.csv')
        for label,n in f['product'].value_counts().items():all_counts.append({'role':role,'label':label,'rows':n})
    pd.DataFrame(all_counts).to_csv(STUDY/'revised_class_counts.csv',index=False)
    source=pd.read_csv(project_path(ROOT, 'dataset/complaints.csv')).dropna(subset=['product','narrative'])
    length=source.narrative.str.len()
    (STUDY/'source_length_summary.json').write_text(json.dumps({'rows':len(source),'median_characters':float(length.median()),
       'p90_characters':float(length.quantile(.9)),'percent_at_most_250':float(100*length.le(250).mean()),
       'percent_at_most_500':float(100*length.le(500).mean()),'percent_at_most_1500':float(100*length.le(1500).mean())},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=pd.read_csv(OUT/'legacy_overlap_narrative_counts.csv',index_col=0)
    names=[n.replace('_subset_balanced_','-B-').replace('_subset_imbalanced_','-I-').replace('.csv','') for n in matrix.index]
    fig,ax=plt.subplots(figsize=(8.4,7.1),layout='constrained')
    image=ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,vmax=1000)
    ax.set_xticks(range(12),names,rotation=60,ha='right',fontsize=9);ax.set_yticks(range(12),names,fontsize=9)
    for i in range(12):
        for j in range(12):
            v=int(matrix.iloc[i,j]);ax.text(j,i,str(v),ha='center',va='center',fontsize=8,color='white' if v>600 else 'black')
    ax.set_title('Shared unique normalized narratives across legacy subsets',fontsize=11)
    ax.set_xlabel('Seed-condition-character limit (B: balanced; I: imbalanced)',fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.75,label='Shared unique narratives')
    for extension in ['pdf','png']:fig.savefig(FIGURES/f'revision_legacy_overlap.{extension}',dpi=180)
    plt.close(fig)


def main():
    TABLES.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
    archived_tables();baseline_tables();dataset_and_overlap()
    print('Generated unified archived/new-baseline tables and legacy overlap figure',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

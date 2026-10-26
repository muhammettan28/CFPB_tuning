#!/usr/bin/env python3
"""Check table identities and cells against saved evidence (CPU, no authoring).

Unlike a whole-document number search, a number in the wrong model/condition row
cannot satisfy this check. Replay the source CSVs first with replay_reviewer_results.
"""
from project_paths import project_path
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from revision_gpu_common import PAPER, DATA, EXP, LABELS, sha_file, load_frame
from analyze_corrected_gpu_results import targets
from analyze_ref14_baselines import predictions
from restored_content_evidence import historical_protocol_rows

NAMES = dict(qwen_base='Qwen3 base', qwen_ft='Qwen3 QLoRA', llama_base='Llama 3.1 base',
             llama_ft='Llama 3.1 QLoRA', bert='BERT', distilbert='DistilBERT', roberta='RoBERTa',
             tfidf_10000='TF-IDF/LR 10k', tfidf_full='TF-IDF/LR full')
CLASSES = ['Retail banking', 'Credit reporting', 'Credit card', 'Mortgages/loans', 'Debt collection']
COND = dict(chars500_tokens512='A', chars500_tokens2048='B', chars1500_tokens2048='C')
RESULT = project_path(PAPER, 'analysis/gpu_execution')
POST = project_path(PAPER, 'analysis/postprocessing')

def f(v): return f'{float(v):.4f}'
def ci(r): return f'[{f(r.low)}, {f(r.high)}]'
def sd(v): return '---' if pd.isna(v) else f(v)
def normalize(v):
    value=re.sub(r'\s+', ' ', str(v).replace(r'\_', '_')).strip()
    return 'benchmark' if value.lower() in {'benchmark','legacy'} else value
def parse_rows(text):
    return [[normalize(c) for c in line.strip().removesuffix(r'\\').split(' & ')]
            for line in text.splitlines() if ' & ' in line and line.strip().endswith(r'\\')]
def equal_rows(actual, expected, label):
    expected = [[normalize(c) for c in row] for row in expected]
    if actual != expected:
        if len(actual) != len(expected): raise AssertionError((label, 'row count', len(actual), len(expected)))
        i = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise AssertionError((label, 'row', i+1, 'actual', actual[i], 'expected', expected[i]))

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    tex = (project_path(PAPER, 'supplementary_material.tex')).read_text()
    blocks = re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}', tex, re.S)
    tables = {re.search(r'\\label\{([^}]+)\}', b).group(1): parse_rows(b.split(r'\endhead', 1)[1]) for b in blocks}
    for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S):
        label=re.search(r'\\label\{([^}]+)\}',b).group(1)
        files=re.findall(r'\\input\{([^}]+)\}',b)
        tables[label]=parse_rows('\n'.join((project_path(PAPER, path)).read_text() for path in files))
    report = []
    def check(label, rows):
        actual = tables[label] if label.startswith('sup:') else parse_rows((project_path(PAPER, 'tables/final', label)).read_text())
        equal_rows(actual, rows, label)
        report.append(dict(table=label, rows=len(actual), cells=sum(map(len, actual))))
    metrics = pd.read_csv(RESULT/'corrected_model_metrics.csv')
    for role in ['internal', 'legacy']:
        m = metrics.loc[metrics.role.eq(role)]
        check('sup:metrics-'+role, [[NAMES[r.system], r.metric, f(r.point), ci(r), sd(r.seed_sd_ddof1)] for r in m.itertuples()])
        rows = []
        for system, g in m.groupby('system', sort=False):
            g = g.set_index('metric'); r = g.loc['macro_f1']
            rows.append([NAMES[system], f'{int(r.train_rows):,}', f(r.point)+' '+ci(r), sd(r.seed_sd_ddof1),
                         f(g.loc['accuracy', 'point']), f(g.loc['weighted_f1', 'point'])])
        check(role+'_models.tex', rows)
        seeds = pd.read_csv(RESULT/'corrected_per_seed_metrics.csv')
        seeds = seeds.loc[seeds.role.eq(role)].pivot(index='variant', columns='metric', values='value')
        check('sup:seeds-'+role, [[v, f(r.accuracy), f(r.macro_f1), f(r.weighted_f1)] for v, r in seeds.iterrows()])
    c = pd.read_csv(RESULT/'corrected_paired_contrasts.csv'); c = c.loc[c.metric.eq('macro_f1')]
    check('sup:contrasts', [[r.role, NAMES[r.reference], NAMES[r.comparison], f(r.paired_difference), ci(r)] for r in c.itertuples()])
    check('internal_contrasts.tex', [[NAMES[r.reference], NAMES[r.comparison], f(r.paired_difference), ci(r)] for r in c.loc[c.role.eq('internal')].itertuples()])
    classes = pd.read_csv(POST/'corrected_class_metrics.csv')
    check('sup:class-metrics', [[NAMES[r.system], CLASSES[LABELS.index(r.label)], f(r.precision), f(r.recall), f(r.f1), f'{r.support:,}'] for r in classes.itertuples()])
    off = pd.read_csv(POST/'corrected_off_label_counts.csv', dtype={'variant':str})
    check('sup:off-label', [[NAMES[r.system], r.variant, r.off_label_count] for r in off.itertuples()])
    protocol = json.loads((project_path(PAPER, 'analysis/revision_v1/protocol.json')).read_text())
    check('sup:flow', [[r['partition'].replace('internal_test','internal').title(), *[f'{r[k]:,}' for k in ['original_rows','quarantined_rows','retained_rows','groups']]] for r in protocol['split_sizes']])
    tr = load_frame(DATA/'train_10000.csv')['product'].value_counts(); va = load_frame(DATA/'validation_2000.csv')['product'].value_counts()
    check('sup:classes', [[title, *[f'{r["class_counts"][label]:,}' for r in protocol['split_sizes']], f'{tr[label]:,}', f'{va[label]:,}'] for label, title in zip(LABELS, CLASSES)])
    check('sup:audits', [[model, (a:=json.loads((project_path(PAPER, f'analysis/gpu_preparation/input_audit_{model}.json')).read_text()))['rows'], len(a['failures']), a['passed']] for model in ['qwen','llama','bert','distilbert','roberta']])
    rows=[]
    for model in ['bert','distilbert','roberta']:
        d=EXP/'encoder'/model/'search'; win=json.loads((d/'selected_config.json').read_text())['candidate']['candidate_id']
        for r in json.loads((d/'validation_search.json').read_text()):
            rows.append([model,r['candidate_id'],f'{r["learning_rate"]:.0e}',r['epochs'],f(r['validation_macro_f1']),'Selected' if r['candidate_id']==win else ''])
    check('sup:encoder-search', rows)
    rows=[]
    for budget in ['10000','full']:
        d=project_path(PAPER, 'experiments/tfidf_revision_v1', 'budget_'+budget)
        winner=json.loads((d/'selected_config.json').read_text())['candidate_id']
        for r in pd.read_csv(d/'validation_search.csv').fillna({'class_weight':'none'}).itertuples():
            rows.append([budget,r.candidate_id,f'(1,{r.ngram_max})',f'{r.max_features:,}',r.min_df,f'{r.C:g}',r.class_weight,f(r.macro_f1),'*' if r.candidate_id==winner else ''])
    check('sup:tfidf-search', rows)
    files=sorted((DATA/'legacy_subsets').glob('*.csv'))
    check('sup:legacy-key', [[i+1, path.stem.split('_')[0], path.stem.split('_')[2], path.stem.split('_')[3],
          '200/class' if path.stem.split('_')[2]=='balanced' else '511/182/107/100/100'] for i,path in enumerate(files)])
    pins=json.loads((project_path(PAPER, 'analysis/gpu_preparation/MODEL_DOWNLOAD_VALIDATION.json')).read_text())['models']
    check('sup:pins', [[model,r'\nolinkurl{'+pins[model]['repo_id']+'}',r'\nolinkurl{'+pins[model]['commit']+'}'] for model in ['qwen','llama','bert','distilbert','roberta']])
    check('sup:protocol-hashes', [[name,r'\nolinkurl{'+sha_file(project_path(PAPER, path))+'}'] for name,path in [
          ('Data protocol','analysis/revision_v1/protocol.json'),('Main GPU plan','configs/gpu_revision_v1.json'),('Reference plan','configs/ref14_encoder_v1.json')]])
    for stem in ['legacy_overlap_narrative_counts','legacy_overlap_label_narrative_counts']:
        frame=pd.read_csv(project_path(PAPER, f'analysis/cpu_revision/{stem}.csv'),index_col=0)
        check('sup:'+stem, [[i+1,*[int(v) for v in frame.loc[file.name,[p.name for p in files]]]] for i,file in enumerate(files)])
    views=pd.read_csv(POST/'corrected_legacy_per_view.csv')
    for suffix,systems in [('llm',['qwen_base','qwen_ft','llama_base','llama_ft']),('baseline',['bert','distilbert','roberta','tfidf_10000','tfidf_full'])]:
        check('sup:legacy-'+suffix, [[i+1,*[f(views.loc[views.source_file.eq(path.name)&views.system.eq(s),'macro_f1'].item()) for s in systems]] for i,path in enumerate(files)])
    lengths=pd.read_csv(POST/'long_input_per_variant.csv',dtype={'variant':str})
    check('sup:length-seeds', [[r.model,r.variant,COND[r.condition],f(r.accuracy),f(r.macro_f1),f(r.weighted_f1)] for r in lengths.itertuples()])
    effects=pd.read_csv(RESULT/'corrected_long_input.csv'); effects=effects.loc[effects.comparison.eq('500_to_1500_chars')]
    check('sup:length-effects', [[r.model,'base' if r.variant=='base' else 'FT mean','all' if r.cohort=='all' else r'$>500$',r.metric,f(r.paired_difference),ci(r)] for r in effects.itertuples()])
    check('long_macro.tex', [['Qwen3' if r.model=='qwen' else 'Llama 3.1','base' if r.variant=='base' else 'QLoRA mean','All' if r.cohort=='all' else r'$>500$', f(r.reference_point),f(r.comparison_point),f(r.paired_difference),ci(r)] for r in effects.loc[effects.metric.eq('macro_f1')].itertuples()])
    latency=pd.read_csv(RESULT/'controlled_latency_summary.csv',dtype={'variant':str}); rows=[]
    for (model,variant,condition),g in latency.groupby(['model','variant','condition'],sort=False):
        g=g.set_index('timing');e=g.loc['end_to_end_seconds']
        rows.append([model,variant,COND[condition],f(g.loc['generation_seconds','mean']),f(e['mean']),f(e['median']),f(e['p95']),f'{e.mean_input_tokens:.2f}',f'{e.mean_new_tokens:.2f}',f'{e.serial_requests_per_second:.3f}'])
    check('sup:latency-all',rows)
    paired=pd.read_csv(RESULT/'controlled_latency_paired.csv')
    check('sup:latency-paired', [[r.model,COND[r.condition],'E2E' if r.timing=='end_to_end_seconds' else 'Generation',f(r.ft_seed_mean_minus_base_seconds),ci(r)] for r in paired.itertuples()])
    rows=[]
    for model in ['qwen','llama']:
        g=latency.loc[latency.model.eq(model)&latency.condition.eq('chars500_tokens512')&latency.timing.eq('end_to_end_seconds')]
        r=paired.loc[paired.model.eq(model)&paired.condition.eq('chars500_tokens512')&paired.timing.eq('end_to_end_seconds')].iloc[0]
        rows.append(['Qwen3' if model=='qwen' else 'Llama 3.1',f(g.loc[g.variant.eq('base'),'mean'].item()),f(g.loc[~g.variant.eq('base'),'mean'].mean()),f(r.ft_seed_mean_minus_base_seconds),ci(r)])
    check('latency_main.tex',rows)
    ref=pd.read_csv(project_path(PAPER, 'analysis/ref14_encoder/model_metrics.csv'))
    check('sup:ref14-means', [[r.role,r.model,r.setting,f(r.point),ci(r),f(r.seed_sd_ddof1)] for r in ref.loc[ref.metric.eq('macro_f1')].itertuples()])
    ref=pd.read_csv(project_path(PAPER, 'analysis/ref14_encoder/paired_contrasts.csv'))
    check('sup:ref14-deltas', [[r.role,r.model,r.metric,f(r.ref14_minus_main),ci(r)] for r in ref.itertuples()])
    internal,legacy=targets();rows=[]
    for model in ['bert','roberta']:
        for seed in [42,43,44]:
            for role,target in [('internal',internal),('legacy',legacy)]:
                pred=predictions(model,seed,target,role);weights=target.unit_weight if role=='legacy' else None
                values=[accuracy_score(target.true_label,pred,sample_weight=weights), *[f1_score(target.true_label,pred,labels=LABELS,average=avg,zero_division=0,sample_weight=weights) for avg in ['macro','weighted']]]
                rows.append([model,seed,role,*map(f,values)])
    check('sup:ref14-seeds',rows)
    old=pd.read_csv(project_path(PAPER, 'analysis/cpu_revision/paired_group_bootstrap.csv'))
    old=old.loc[old.analysis.eq('internal_original')]
    rows=[]
    for model,g in old.groupby('model',sort=False):
        g=g.set_index('metric'); m=g.loc['macro_f1']; w=g.loc['weighted_f1']
        rows.append([model,f(m.base_point),f(m.ft_seed_mean_point)+r'$\pm$'+f(m.ft_seed_sd_ddof1),f(m.paired_gain_point),f(w.base_point),f(w.ft_seed_mean_point)+r'$\pm$'+f(w.ft_seed_sd_ddof1)])
    check('sup:historical-internal',rows)
    def interval(point,low,high):return f(point)+' ['+f(low)+', '+f(high)+']'
    check('sup:historical-ci', [[r.model,r.metric.replace('_',' '),interval(r.base_point,r.base_low,r.base_high),interval(r.ft_seed_mean_point,r.ft_low,r.ft_high),interval(r.paired_gain_point,r.gain_low,r.gain_high)] for r in old.itertuples()])
    old=pd.read_csv(project_path(PAPER, 'analysis/cpu_revision/descriptive_truncation_strata.csv'));rows=[]
    for (model,stratum),g in old.loc[old.role.eq('internal')].groupby(['model','stratum'],sort=False):
        b=g.loc[g.variant.eq('base_zero_shot')].iloc[0]
        rows.append([model,{'<=250_chars':r'$\leq250$','251-500_chars':'251--500','>500_chars':r'$>500$'}[stratum],int(b.rows),f(b.macro_f1),f(g.loc[~g.variant.eq('base_zero_shot'),'macro_f1'].mean())])
    check('sup:historical-strata',rows)
    old=pd.read_csv(project_path(PAPER, 'analysis/revision_v1/paired_truncation_metrics.csv'))
    check('sup:historical-pairs', [[r.model,'Base' if r.variant=='base_zero_shot' else 'FT mean',r.unique_pairs,f(r.short_point),f(r.long_point),interval(r.gain,r.gain_low,r.gain_high)] for r in old.loc[old.cohort.eq('changed_input_only')&old.metric.eq('macro_f1')].itertuples()])
    check('sup:historical-selection', [[r['condition'],r['sampling'],f"{r['learning_rate']:.0e}",r['rank'],r['epochs'],f(r['macro_f1']),f(r['debt_f1'])] for r in historical_protocol_rows()])
    # Guard the checker itself: row swaps must fail even if all numbers are present.
    probe=[['A','0.1'],['B','0.2']]
    for bad in [probe[::-1],[['A','0.2'],['B','0.1']]]:
        try: equal_rows(bad,probe,'intentional corruption')
        except AssertionError: pass
        else: raise AssertionError('Corrupted row escaped validation')
    payload={'passed':True,'tables':report,'total_rows':sum(r['rows'] for r in report),'total_cells':sum(r['cells'] for r in report),'row_swap_and_wrong_value_negative_controls_passed':True,'supplement_table_labels':re.findall(r'\\label\{(sup:[^}]+)\}',tex),'documents_sha256':{name:sha_file(project_path(PAPER, name)) for name in ['manuscript.tex','supplementary_material.tex']},'scope':'Result/selection/count matrices checked cell by cell; narrative and static implementation tables require the separate source/reading audit.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'PASS: {len(report)} keyed tables, {payload["total_rows"]} rows, {payload["total_cells"]} cells; two corruption controls.')

if __name__=='__main__':main()

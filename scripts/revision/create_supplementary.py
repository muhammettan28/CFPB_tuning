#!/usr/bin/env python3
"""Create a readable supplement from completed CPU/GPU artifacts, without fitting."""
from project_paths import project_path
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from finalize_revision_documents import OUT,NAMES,CLASS_NAMES,esc,f,ci,row,metric_frame
from revision_gpu_common import PAPER,ROOT,DATA,EXP,LABELS,load_frame,sha_file
from restored_content_evidence import historical_protocol_rows
from publication_language import publication_terms


def read(path):return json.loads(Path(path).read_text())
def section(title,label):return '\n'+r'\section{'+title+'}'+r'\label{'+label+'}\n'
def table(caption,label,headers,rows,fmt=None,size='small'):
    fmt=fmt or ('l'*len(headers))
    return ('\n\\Needspace{8\\baselineskip}\n{\\'+size+'\n\\setlength{\\tabcolsep}{4pt}\n'+r'\begin{longtable}{@{}'+fmt+r'@{}}'+'\n'+
        r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+row(headers)+'\n'+
        r'\midrule\endfirsthead'+'\n'+r'\multicolumn{'+str(len(headers))+r'}{l}{\small Table \thetable\ continued}\\'+'\n'+
        r'\toprule'+'\n'+row(headers)+'\n'+r'\midrule\endhead'+'\n'+r'\bottomrule\endfoot'+'\n'+
        '\n'.join(row(values) for values in rows)+'\n'+r'\end{longtable}'+'\n}\n')


def main():
    lines=[r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=22mm]{geometry}
\usepackage{lmodern,booktabs,longtable,array,tabularx,graphicx,pdflscape,amsmath,fvextra,needspace,float}
\usepackage{xurl,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,pdftitle={Supplementary material: QLoRA complaint classification}}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small QLoRA complaint classification -- Supplementary material}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{3em}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\begin{center}{\Large\bfseries Supplementary material}\\[5pt]
{\large QLoRA Fine-Tuning for Consumer Complaint Classification}\\[5pt]
Implementation, full results and reproducibility information
\end{center}
This supplement describes data allocation, model implementation, validation searches, complete results, input-length and latency conditions, and reference-parameter sensitivity. Auxiliary analyses specify their fitting populations and dependence limitations. The reproducibility package supplies the corresponding machine-readable data, predictions and verification code.
\tableofcontents
\clearpage
''']
    lines += [section('Data provenance, fixed selections and visible-input checks','sup:data'),r'''
The local source ZIP has 162,421 rows; the working file retains its first 162,415 rows identically. The six absent trailing records have the placeholder narrative \texttt{name}. Ten records with missing required fields are removed, leaving 162,405 usable rows. Excluding 8,123 benchmark-related source rows produces a 154,282-row partitioning pool. Model-visible overlap screening excludes 2,925 candidate training and 307 candidate validation rows from fitting while retaining the evaluation records. These operations document the local archive, not the unavailable original CFPB-to-Kaggle preprocessing or product aggregation.

The benchmark-related exclusion records comprise 4,990 exact normalised-text matches, 3,096 250-character prefix matches, 23 additional narrative-group members and 14 cross-label prefix matches. These counts describe source-pool filtering; model-visible screening additionally excludes fitting conflicts under the actual input representations.

Labels are inherited by identity. Their code order is retail\_banking, credit\_reporting, credit\_card, mortgages\_and\_loans, debt\_collection. Connected components use full normalisation, 500-character and reconstructed tokenizer representations, with a conservative 250-character legacy prefix guard. They represent text dependencies, not verified complaint IDs. No semantic/paraphrase independence guarantee follows.
''']
    counts=pd.read_csv(project_path(PAPER, 'analysis/revision_v1/revised_class_counts.csv'))
    train=load_frame(DATA/'train_10000.csv')['product'].value_counts();val=load_frame(DATA/'validation_2000.csv')['product'].value_counts()
    rows=[]
    for label,title in zip(LABELS,CLASS_NAMES):
        values=[int(counts.loc[counts.role.eq(role)&counts.label.eq(label),'rows'].iloc[0]) for role in ['train','validation','internal_test']]
        rows.append([title,*[f'{x:,}' for x in values],f'{train[label]:,}',f'{val[label]:,}'])
    lines += [table('Retained and actually selected class counts. The full training pool is used only by the full-data sparse baseline.','sup:classes',
        ['Class','Train pool','Val. pool','Internal','Train 10k','Val. 2k'],rows,'lrrrrr')]
    flow=[['Train','123,427','2,925','120,502','93,609'],['Validation','15,427','307','15,120','11,925'],['Internal','15,428','0','15,428','12,099']]
    lines += [table('Candidate allocation, excluded fitting conflicts, retained rows and conservative component counts.','sup:flow',
        ['Role','Candidate rows','Excluded','Retained','Groups'],flow,'lrrrr')]
    rows=[]
    for model in ['qwen','llama','bert','distilbert','roberta']:
        audit=read(project_path(PAPER, f'analysis/gpu_preparation/input_audit_{model}.json'))
        rows.append([model,str(audit['rows']),'0' if not audit['failures'] else str(len(audit['failures'])),str(audit['passed'])])
    lines += [table('Actual pinned-tokenizer audits. Every model has 10k train + 2k validation + 15,428 internal + 12k legacy appearances. Zero refers only to the tested train/validation split intersections.','sup:audits',
        ['Model','Inputs','Separation failures','Passed'],rows,'lrrl'),r'''
The 25 retained-pool representation checks validate the allocation, while the five pinned-tokenizer audits verify the exact selected inputs supplied to the models. Longer-input keys and both reference-parameter encoder tokenizers are checked separately. Equal token inputs remain inside the frozen bootstrap groups. All data hashes and row selections are in \path{analysis/revision_v1/protocol.json} and per-run manifests.

The internal cohort is fixed and inspected, so evaluation is retrospective. Benchmark files are byte-identical to source-ZIP members. Their sampling implementation is unavailable; seed identifiers and balanced/imbalanced/length conditions are documented from the frozen files. The local source derivation does not recover unknown upstream transformations.
''']
    lines += [section('Implementation, immutable artifacts and validation searches','sup:implementation'),r'''
\subsection{Pinned models and execution environment}
All neural runs use locally cached, immutable Hugging Face snapshots. Weight files were checked against the pinned Hub metadata by full hashes, sizes and safetensors headers/offsets. The identities in Table~\ref{sup:pins} define the primary-comparison models.
''']
    downloads=read(project_path(PAPER, 'analysis/gpu_preparation/MODEL_DOWNLOAD_VALIDATION.json'))['models']
    lines += [table('Immutable model identities used for the primary comparison.','sup:pins',['Model','Repository','Commit'],
        [[m,r'\nolinkurl{'+downloads[m]['repo_id']+'}',r'\nolinkurl{'+downloads[m]['commit']+'}'] for m in ['qwen','llama','bert','distilbert','roberta']],
        'p{16mm}p{55mm}p{86mm}')]
    env=read(project_path(PAPER, 'analysis/gpu_execution/GPU_ENVIRONMENT.json'))
    versions=[['GPU','NVIDIA RTX 5060 Ti 16 GB; driver 595.99.02'],['Python','3.12.14'],['PyTorch/CUDA','2.7.1+cu128 / 12.8']]
    versions += [[esc(k),esc(v)] for k,v in env['versions'].items()]
    lines += [table('Actual GPU environment. Complete pip freeze and code snapshots are retained per job. CPU statistics use the separately recorded analysis environment (pandas 3.0.3; scikit-learn 1.9.0).','sup:environment',['Component','Value'],versions,'p{42mm}p{115mm}')]
    lines += [r'''
\clearpage
\subsection{QLoRA and inference}
''']
    settings=[['Training / validation','10,000 / 2,000 fixed rows; selection seed 42'],['Optimisation seeds','42, 43, 44; fresh same-data refits'],
        ['Quantisation','4-bit NF4, double quantisation, FP16 compute'],['Adapter','Rank 16; alpha 32; dropout 0.05; all-linear targets'],
        ['Objective','Completion-only cross-entropy over target JSON and EOS'],['Epochs / LR','2 / 0.0002'],['Physical / effective batch','1 / 16 (accumulation 16)'],
        ['Optimiser / scheduler','Paged AdamW 8-bit / cosine'],['Warmup','0.03'],['Gradient checkpointing','Non-reentrant'],
        ['Model selection','Best saved validation loss, every 100 optimiser steps'],['Main visible narrative','First 500 characters; input cap 512 tokens'],
        ['Generation','Greedy; one beam; maximum 32 new tokens; cache enabled'],['Thinking / serialization',r'\texttt{enable\_thinking=False}; render once; no added duplicate BOS'],
        ['Output handling','Final product-field regex; invalid or absent labels remain errors; no retries']]
    lines += [table('QLoRA and generation settings.','sup:qlora',['Setting','Value'],settings,'p{43mm}p{114mm}')]
    from revision_gpu_common import SYSTEM_PROMPT
    lines += [r'''The exact system prompt below is taken from the retained evaluation code; the user message is \texttt{Complaint:} followed by a newline and the truncated narrative. The target is \texttt{\{"product": "label"\}} followed by EOS. The template boundary and the full target fitting the 512-token SFT limit are verified for every selected training/validation record. JSON-format prompting is not token-level constrained decoding.
\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\small]
'''+SYSTEM_PROMPT+r'''
\end{Verbatim}
Displayed prompt line wraps are typographical; the source string and rendered templates in the retained code determine actual serialization. Exact-input memoisation is used only in deterministic LLM prediction generation. Identity includes token IDs/masks, immutable base and fitted-weight hashes, generation settings, runner/cache code hashes, library versions, hardware and driver. All appearances, labels and row metadata remain in prediction tables. Labels and row identities are not cache keys. Latency and warmups always make fresh GPU calls. The two 500-character token-cap control settings have identical encoded inputs and may reuse replies; they are not independent inference replicates.

\subsection{Encoder selection and fresh refits}
Main BERT, DistilBERT and RoBERTa use 500 characters and a 512-token cap, physical batch 4/accumulation 4, evaluation batch 16, FP16, AdamW, weight decay 0.01, linear scheduling and warmup 0.06. Search seed is 42. Each candidate is evaluated at its final epoch; the globally highest validation macro-F1 wins, with the smallest candidate ID for ties within $10^{-12}$. Three fresh refits use the selected configuration. Test inference is blocked unless all 2,000 reloaded validation predictions exactly match their Trainer predictions.
''']
    rows=[]
    for model in ['bert','distilbert','roberta']:
        directory=EXP/'encoder'/model/'search';winner=read(directory/'selected_config.json')['candidate']['candidate_id']
        for c in read(directory/'validation_search.json'):
            rows.append([model,c['candidate_id'],f"{c['learning_rate']:.0e}",c['epochs'],f(c['validation_macro_f1']),'Selected' if c['candidate_id']==winner else ''])
    lines += [table('All 18 main encoder validation candidates. Only validation enters selection.','sup:encoder-search',
        ['Model','ID','LR','Epochs','Val. macro-F1','Decision'],rows,'llrrrr')]
    lines += [r'''\subsection{Sparse baseline selection}
Each budget has 48 candidates. Fixed settings are lowercasing, Unicode accent stripping, sublinear term frequency, L2 normalisation, lbfgs, tolerance $10^{-5}$ and maximum 2,000 iterations. Vocabulary/IDF fit training only. Validation and test transformations never refit the feature map. The highest validation macro-F1 wins with the same predefined smallest-ID tie rule. The seed-labelled lbfgs fits have identical predictions at each budget; their SD zero is a determinism check, not stochastic seed evidence.
''']
    rows=[]
    for budget in ['10000','full']:
        directory=project_path(PAPER, f'experiments/tfidf_revision_v1/budget_{budget}')
        winner=read(directory/'selected_config.json')['candidate_id']
        frame=pd.read_csv(directory/'validation_search.csv').fillna({'class_weight':'none'})
        for r in frame.itertuples(index=False):rows.append([budget,r.candidate_id,f'(1,{r.ngram_max})',f'{r.max_features:,}',r.min_df,f'{r.C:g}',r.class_weight,f(r.macro_f1),'*' if r.candidate_id==winner else ''])
    lines += [table('All 96 sparse validation candidates (48 per budget). An asterisk marks the selected candidate; full means 120,502 training rows.','sup:tfidf-search',
        ['Budget','ID','N-grams','Max feat.','Min df','C','Weight','Val. mF1',''],rows,'llrrrrlrl','footnotesize')]
    lines += [section('Full model, seed, class and uncertainty results','sup:performance'),r'''
All primary estimates use the same fixed training and validation selections. For adapted/encoder systems, points are means of three individually scored refits, not ensembles. SD is sample SD over those refits. Base systems have one checkpoint. Sparse repeated fits are deterministic. Intervals use 10,000 paired whole-group bootstrap draws (seed 20261020) and condition on fitted models/selection; they exclude training and hyperparameter-selection uncertainty. No family-wise adjustment is applied. Legacy estimands equally weight unique visible label--narrative units rather than the 12,000 repeated appearances. All 19 main variants share each main-study draw.

For class $k$, F1 is $2TP_k/(2TP_k+FP_k+FN_k)$ with zero-denominator scores set to zero; macro-F1 averages the five class scores and weighted F1 uses true-class support. The internal group-confusion tensor has five true classes and six prediction outcomes, including off-label. Off-label outputs therefore remain false negatives. Point estimates are computed from observed group counts before resampling. For computational efficiency, groups with identical joint confusion profiles across the analysed variants are collapsed into multinomial categories, with category probabilities equal to their frequencies divided by the group count. Sampling these categories preserves exactly the ordinary whole-group bootstrap distribution. The separate 12-variant reference analysis has a different joint-profile compression from the 19-variant main analysis; the same RNG seed does not imply identical Monte Carlo draws between those analyses. Within each analysis, every contrast still uses common paired draws.

More explicitly, let $C_{gs}$ be the weighted confusion counts of group $g$ for fitted variant $s$, and $h$ the metric. A three-refit system has observed point $\widehat\theta=\frac{1}{3}\sum_{s=1}^3 h(\sum_g C_{gs})$. In draw $b$, $(N_{1b},\ldots,N_{Gb})\sim\mathrm{Multinomial}(G;1/G,\ldots,1/G)$ replaces each group count by $N_{gb}C_{gs}$. Paired differences use the same $N_{gb}$ for both systems; the 2.5th and 97.5th percentiles define the interval. Base and deterministic sparse systems have one scored variant. Internal row weights are one. A legacy appearance has weight $1/m_u$, where $m_u$ is the number of appearances of its normalised label--narrative unit $u$; those weights remain fixed during resampling. Metrics are averaged after scoring each fitted variant, rather than scoring an averaged confusion matrix. The fitted checkpoints and selected configurations are held fixed throughout.
''']
    frame=metric_frame()
    for role,title in [('internal','Internal: 15,428 rows / 12,099 groups'),('legacy','Legacy: 5,909 weighted units / 3,919 groups')]:
        rows=[]
        for r in frame.loc[frame.role.eq(role)].itertuples(index=False):rows.append([NAMES[r.system],esc(r.metric),f(r.point),ci(r.low,r.high),'---' if pd.isna(r.seed_sd_ddof1) else f(r.seed_sd_ddof1)])
        lines += [table(title+'. All three metrics use observed points and common paired-group draws.','sup:metrics-'+role,
            ['System','Metric','Point',r'95\% CI','Seed SD'],rows,'llrrr')]
    seed=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_per_seed_metrics.csv'))
    for role in ['internal','legacy']:
        pivot=seed.loc[seed.role.eq(role)].pivot(index='variant',columns='metric',values='value')
        lines += [table('Individual '+role+' checkpoint scores. Each TF-IDF row represents three identical deterministic fits.','sup:seeds-'+role,
            ['Variant','Accuracy','Macro-F1','Weighted F1'],[[esc(i),f(r.accuracy),f(r.macro_f1),f(r.weighted_f1)] for i,r in pivot.iterrows()],'lrrr')]
    contrast=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_paired_contrasts.csv'))
    lines += [table(r'All main macro-F1 paired contrasts, comparison minus reference. Full-data TF-IDF comparisons have unequal training budgets. The internal Llama-versus-full-data upper bound is $+6.8\times10^{-6}$ before rounding, so that interval includes zero.','sup:contrasts',
        ['View','Reference','Comparison','Difference',r'95\% CI'],
        [[r.role,NAMES[r.reference],NAMES[r.comparison],f(r.paired_difference),ci(r.low,r.high)] for r in contrast.loc[contrast.metric.eq('macro_f1')].itertuples(index=False)],'lllrr')]
    classes=pd.read_csv(OUT/'corrected_class_metrics.csv')
    lines += [table('Internal class metrics. Multi-seed rows average the individually computed class statistics; supports are fixed.','sup:class-metrics',
        ['System','Class','Precision','Recall','F1','Support'],[[NAMES[r.system],CLASS_NAMES[LABELS.index(r.label)],f(r.precision),f(r.recall),f(r.f1),f'{r.support:,}'] for r in classes.itertuples(index=False)],'llrrrr','footnotesize')]
    off=pd.read_csv(OUT/'corrected_off_label_counts.csv')
    lines += [table('Observed off-label outputs among 15,428 internal predictions. Compliance is not constrained decoding.','sup:off-label',
        ['System','Variant','Off-label count'],[[NAMES[r.system],r.variant,r.off_label_count] for r in off.itertuples(index=False)],'llr')]
    lines += [section('Legacy overlaps and per-view descriptions','sup:legacy'),r'''
The twelve files contain 12,000 appearances but only 5,909 unique (label, normalised visible narrative) units. Full-text/prefix linkage forms 3,919 conservative dependency groups. A group draw retains all linked short/long views, with total weight one for repeated identical units. This estimand differs from an unweighted mean of twelve file scores. The files are dependent, shared-source evaluation views. Statistical inference resamples their linked groups rather than treating file scores as independent observations.
''']
    files=sorted((DATA/'legacy_subsets').glob('*.csv'));ids={p.name:i+1 for i,p in enumerate(files)}
    rows=[]
    for path in files:
        seed_no,_,balance,chars=path.stem.split('_')
        rows.append([ids[path.name],seed_no,balance,chars,'200/class' if balance=='balanced' else '511/182/107/100/100'])
    lines += [table('Legacy file key. ID order matches both overlap matrices. Imbalanced counts are credit reporting/debt/loans/card/banking; sampling code is unavailable.','sup:legacy-key',
        ['ID','Seed ID','Sampling','Chars','Class counts'],rows,'rrlrl')]
    for stem,title in [('legacy_overlap_narrative_counts','Normalised narrative overlap'),('legacy_overlap_label_narrative_counts','Normalised label--narrative-unit overlap')]:
        overlap=pd.read_csv(project_path(PAPER, f'analysis/cpu_revision/{stem}.csv'),index_col=0)
        rows=[[ids[name],*[int(x) for x in overlap.loc[name,[p.name for p in files]]]] for name in [p.name for p in files]]
        lines += [table(title+'. Counts are shared unique inputs; diagonals count within-file unique inputs. Row fractions and Jaccard matrices are in the archive.','sup:'+stem,
            ['ID',*[str(i) for i in range(1,13)]],rows,'r'*13,'scriptsize')]
    views=pd.read_csv(OUT/'corrected_legacy_per_view.csv')
    for suffix,systems in [('llm',['qwen_base','qwen_ft','llama_base','llama_ft']),('baseline',['bert','distilbert','roberta','tfidf_10000','tfidf_full'])]:
        rows=[]
        for path in files:
            group=views.loc[views.source_file.eq(path.name)].set_index('system')
            rows.append([ids[path.name],*[f(group.loc[s,'macro_f1']) for s in systems]])
        lines += [table('Per-file macro-F1: '+suffix+'. Multi-seed columns are means. Descriptive scores are not independent replicates or the unique-unit aggregate.','sup:legacy-'+suffix,
            ['ID',*[NAMES[s] for s in systems]],rows,'r'*(len(systems)+1),'footnotesize')]
    lines += [section('Input-length conditions, cohorts and sensitivity analyses','sup:length'),r'''
Same-checkpoint conditions are 500 characters/512 tokens (A), 500/2,048 (B), and 1,500/2,048 (C). Models are not retrained for C. Every condition uses the same 15,428 internal IDs; the predefined full-narrative $>500$ cohort has 5,770 rows in 5,131 groups. A and B token inputs/predictions are identical, with no extra truncation in either. Their metric differences and bootstrap bounds are exactly zero; exact-response reuse is disclosed in Section S2.

The benchmark files supply only their recorded 250/500-character text; they are not used to construct 1,500-character inputs. Short/long maxima before caps are Qwen 207/403 and Llama 224/421 tokens; all fit the stated caps. Every row records requested/retained characters, tokens and hashes. One common paired group draw is used across the length conditions and checkpoint variants. Longer-text effects include negative zero-shot findings, and do not establish robustness outside this task.
''']
    long=pd.read_csv(OUT/'long_input_per_variant.csv')
    condition={'chars500_tokens512':'A','chars500_tokens2048':'B','chars1500_tokens2048':'C'}
    lines += [table('Every actual length/checkpoint condition on the complete internal test.','sup:length-seeds',
        ['Model','Variant','Condition','Accuracy','Macro-F1','Weighted F1'],
        [[r.model,r.variant,condition[r.condition],f(r.accuracy),f(r.macro_f1),f(r.weighted_f1)] for r in long.itertuples(index=False)],'lllrrr')]
    effects=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_long_input.csv'))
    rows=[]
    for r in effects.loc[effects.comparison.eq('500_to_1500_chars')].itertuples(index=False):
        rows.append([r.model,'base' if r.variant=='base' else 'FT mean','all' if r.cohort=='all' else r'$>500$',esc(r.metric),f(r.paired_difference),ci(r.low,r.high)])
    lines += [table('Paired C minus B changes for all metrics and both predefined cohorts.','sup:length-effects',
        ['Model','Variant','Cohort','Metric','Difference',r'95\% CI'],rows,'llllrr','footnotesize')]
    lines += [r'''
\subsection{Auxiliary length-stratum and matched-view analyses}
The following analyses use auxiliary models fitted with full-text-grouped allocations and seed-specific fitting selections. These models have model-visible training/evaluation overlap and are excluded from the primary model comparison. Length strata differ in class/content and do not isolate a truncation effect. For 250/500 matching, only one-to-one label/normalised-250-prefix matches are retained before looking at predictions; ambiguous matches are excluded. Repeated short/long pairs share total weight one and linked views are resampled together. This is not verified underlying CFPB record identity. Filtering evaluation groups does not remove exposure incurred during fitting or selection.
\begin{table}[H]\centering\small
\caption{Auxiliary-model internal macro-F1 by full narrative length. Inputs are capped at 500 characters; model-visible overlap limits interpretation.}\label{sup:historical-strata}
\begin{tabular}{llrrr}\toprule Model & Length & Rows & Base & FT mean\\\midrule
\input{tables/table_revision_length_strata.tex}
\end{tabular}\end{table}
\begin{table}[H]\centering\footnotesize
\caption{Auxiliary-model 250/500-character matched-view results, unique-pair weighted. The source cohort and fitted models differ from the controlled B/C comparison.}\label{sup:historical-pairs}
\begin{tabular}{llrrrr}\toprule Model & Variant & Unique pairs & 250 & 500 & Difference [CI]\\\midrule
\input{tables/table_revision_paired_truncation.tex}
\end{tabular}\end{table}
''']
    lines += [section('Controlled latency and measurement limitations','sup:latency'),r'''
One hundred internal IDs (20 per class) are selected by fixed hashes; warmup uses the first ten fixed validation IDs. Each model/variant/condition has three repetitions of the same 100 IDs in a hash-defined order. Every timed call is a real generate invocation; no memoisation is used. CUDA is synchronised around generation. End-to-end additionally includes tokenisation, transfer, decoding and regex parsing; loading, warmup and disk I/O are excluded. Base and adapters use the same NF4/double/FP16, SDPA and batch 1; adapters are unmerged. Other CUDA compute causes an abort rather than termination of unrelated processes.

Times are conditional on this RTX 5060 Ti and separate variant job blocks. GPU state is retained before/after each job. The jobs are not a randomised crossover and no thermal equivalence is claimed. The balanced sample need not represent the natural workload. Output lengths differ between base/FT, so differences are not isolated adapter overhead. Repetition means are formed within each ID before a 10,000-draw paired group bootstrap (100 distinct groups). p95 below is the descriptive quantile of the 300 timed calls, not a confidence bound. Requests/second is serial batch-one throughput, not concurrent service capacity.
''']
    latency=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/controlled_latency_summary.csv'))
    rows=[]
    for (model,variant,cond),group in latency.groupby(['model','variant','condition'],sort=False):
        g=group.set_index('timing');e=g.loc['end_to_end_seconds']
        rows.append([model,variant,condition[cond],f(g.loc['generation_seconds','mean']),f(e['mean']),f(e['median']),f(e['p95']),
            f'{e.mean_input_tokens:.2f}',f'{e.mean_new_tokens:.2f}',f'{e.serial_requests_per_second:.3f}'])
    landscape_table = [r'\begin{landscape}',table('All 24 fresh-call latency conditions (300 timed calls each; total 7,200). Gen. is generation seconds; E2E includes preprocessing/postprocessing.','sup:latency-all',
        ['Model','Variant','Cond.','Gen. mean','E2E mean','E2E median','E2E p95','Input tok.','New tok.','Req./s'],rows,'lllrrrrrrr','small'),r'\end{landscape}']
    paired=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/controlled_latency_paired.csv'))
    lines += [table('All paired FT-three-seed-mean minus base time differences (seconds), using ID means over three repetitions.','sup:latency-paired',
        ['Model','Condition','Timing','Difference',r'95\% CI'],[[r.model,condition[r.condition],'E2E' if r.timing=='end_to_end_seconds' else 'Generation',f(r.ft_seed_mean_minus_base_seconds),ci(r.low,r.high)] for r in paired.itertuples(index=False)],'lllrr')]
    lines += landscape_table
    lines += [section('Reference [14] encoder-parameter sensitivity','sup:ref14'),r'''
G\"ull\"u et al. (ACDSA 2025, DOI \nolinkurl{10.1109/ACDSA65407.2025.11166338}), Tables III--IV, report BERT LR $2\times10^{-5}$, three epochs, batch 16, max 128 tokens, and RoBERTa LR $5\times10^{-5}$, three epochs, batch 16, max 100 tokens. The reference learning-rate/epoch pairs coincide with the winners of the primary validation search. Separate conditions therefore change physical batch from 4/accumulation 4 to 16/accumulation 1 and caps from 512 to 128/100. Their joint effect is measured; it is not attributed to one parameter. The \textbf{main RoBERTa condition uses 500 characters/512 tokens}; only this supplementary condition uses 100 tokens.

We retain the fixed 10k/2k membership, 500-character input, same pinned models, PyTorch/FP16 and main weight decay/warmup/scheduler. The source reports TensorFlow for RoBERTa and different split/preprocessing, so this is not an exact replication or a cross-paper score comparison. All six reference-parameter final-epoch refits (seeds 42/43/44) are reported, without choosing a replacement on test scores.

Both reference-parameter tokenizer audits cover 39,428 inputs, with zero tested split collisions and no violations of the frozen dependency groups. The token caps clip 1 BERT and 18 RoBERTa appearances. Tokenisation truncates the narrative inputs before batch collation; padding to a multiple of eight gives a width of at most 104 for the RoBERTa condition while retaining at most 100 real tokens. Training and evaluation use the same truncation and padding procedure. Each of the six reload parity checks exactly matches all 2,000 saved validation predictions.
''']
    ref=pd.read_csv(project_path(PAPER, 'analysis/ref14_encoder/model_metrics.csv'))
    lines += [table('Reference-parameter conditions, macro-F1. Point estimates refer to the same primary-model predictions; intervals belong to the separate supplementary paired analysis.','sup:ref14-means',
        ['View','Model','Setting','Macro-F1',r'95\% CI','Seed SD'],[[r.role,r.model,r.setting,f(r.point),ci(r.low,r.high),f(r.seed_sd_ddof1)] for r in ref.loc[ref.metric.eq('macro_f1')].itertuples(index=False)],'lllrrr')]
    contrasts=pd.read_csv(project_path(PAPER, 'analysis/ref14_encoder/paired_contrasts.csv'))
    lines += [table('All supplementary paired reference-setting minus main-setting differences. The seed order is paired and all 12 prediction variants share each supplementary draw.','sup:ref14-deltas',
        ['View','Model','Metric','Difference',r'95\% CI'],[[r.role,r.model,esc(r.metric),f(r.ref14_minus_main),ci(r.low,r.high)] for r in contrasts.itertuples(index=False)],'lllrr')]
    from analyze_corrected_gpu_results import targets
    from analyze_ref14_baselines import predictions
    internal,legacy=targets();rows=[]
    for model in ['bert','roberta']:
        for seed in [42,43,44]:
            scores=[]
            for role,target in [('internal',internal),('legacy',legacy)]:
                pred=predictions(model,seed,target,role)
                weight=target.unit_weight.to_numpy() if role=='legacy' else None
                scores.append([accuracy_score(target.true_label,pred,sample_weight=weight),
                    f1_score(target.true_label,pred,labels=LABELS,average='macro',zero_division=0,sample_weight=weight),
                    f1_score(target.true_label,pred,labels=LABELS,average='weighted',zero_division=0,sample_weight=weight)])
            for role,values in zip(['internal','legacy'],scores):rows.append([model,seed,role,*map(f,values)])
    lines += [table('Every supplementary refit, with the same legacy unique-input weighting.','sup:ref14-seeds',
        ['Model','Seed','View','Accuracy','Macro-F1','Weighted F1'],rows,'lllrrr')]
    lines += [r'''The internal macro-F1 changes are $-0.0017$ for BERT and $-0.0010$ for RoBERTa, with intervals spanning zero. Legacy changes likewise have intervals spanning zero. These findings do not justify test-based replacement of the main baselines. The slightly different displayed main intervals in this separate 12-variant supplementary analysis reflect its resampling run; primary comparisons use the joint 19-variant main analysis.
''']
    lines += [section('Reproducibility and auxiliary protocol analyses','sup:reproducibility'),r'''
\subsection{Artifact map and verification}
The reproducibility repository is available at \url{https://github.com/muhammettan28/CFPB_tuning}. It contains source/derived datasets, fixed role/selection manifests, analysis code, validation and final prediction files, fitted sparse objects, configurations and per-run code/environment evidence. Recorded SHA-256 values identify the inputs and results. Neural weights, optimizer states and third-party literature are excluded. Exact base-model snapshots must be downloaded separately. Repeating neural fitting requires a separate project checkout with an empty experiments directory and a registry identifying the downloaded snapshots. Execution scripts use project-relative output paths and reject attempts to overwrite completed runs. Saved predictions support result verification without the omitted fitted weights.

Saved-prediction analysis can instead be repeated without neural weights, a GPU or a local model registry. From the repository root, install \path{scripts/revision/requirements-cpu-revision.txt}, then run:
\begin{Verbatim}[fontsize=\footnotesize]
python scripts/revision/replay_reviewer_results.py \
  --output /tmp/cfpb-review-replay
\end{Verbatim}
The output directory must not exist. This checks frozen data and source hashes, recomputes all main/reference metrics, seed scores, paired intervals, length sensitivity and latency summaries, and compares eight full result tables to reference CSVs at absolute tolerance $10^{-12}$. It writes replay outputs to a separate directory. It does not rerun neural inference or recover unavailable provenance. The replay verifies the supplied result inputs and does not require the manuscript preparation files.

Source hashes identify the analysed files; upstream CFPB-to-Kaggle preprocessing and product aggregation are unavailable. Auxiliary fits lack some immutable upstream identities and complete fitted Qwen artifacts, limiting their reconstruction. Primary-comparison identities are listed in Table~\ref{sup:pins}. Execution metadata can contain absolute machine paths.
''']
    rows=[['Frozen data protocol',r'\path{analysis/revision_v1/protocol.json}'],['GPU protocol',r'\path{configs/gpu_revision_v1.json}'],
        ['Reference sensitivity',r'\path{configs/ref14_encoder_v1.json}'],['Actual tokenizer checks',r'\path{analysis/gpu_preparation/input_audit_*.json}'],
        ['Main results and sources',r'\path{analysis/gpu_execution/}'],['Supplementary sensitivity',r'\path{analysis/ref14_encoder/}'],
        ['Descriptive document tables',r'\path{analysis/postprocessing/}'],['CPU sparse searches',r'\path{experiments/tfidf_revision_v1/}'],
        ['Neural predictions/code',r'\path{experiments/gpu_revision_v1/}'],['Reference-condition runs',r'\path{experiments/ref14_encoder_v1/}'],
        ]
    lines += [table(r'Key paths relative to the repository root. Complete filename/hash lists are machine-readable.','sup:artifact-map',['Evidence','Location'],rows,'p{52mm}p{105mm}')]
    hashes=[['Data protocol',sha_file(project_path(PAPER, 'analysis/revision_v1/protocol.json'))],['Main GPU plan',sha_file(project_path(PAPER, 'configs/gpu_revision_v1.json'))],['Reference plan',sha_file(project_path(PAPER, 'configs/ref14_encoder_v1.json'))]]
    lines += [table('Frozen protocol SHA-256 values.','sup:protocol-hashes',['Artifact','SHA-256'],[[name,r'\nolinkurl{'+digest+'}'] for name,digest in hashes],'p{35mm}p{122mm}')]
    lines += [r'''
\subsection{Auxiliary data-overlap analysis}
Auxiliary models fitted under full-normalised-narrative grouping support a diagnostic analysis of dependence under truncated inputs. Their fitting selections vary with seed, so sample SD combines selection and optimisation variation. At 500 characters, 192 shared unique training inputs are associated with 584 internal-test rows; the three Llama fitting selections expose 383/375/390 test rows. These models therefore have model-visible training/evaluation overlap. Their scores are excluded from the primary comparison. Complete fitted Qwen artifacts and immutable upstream identities for these auxiliary models are unavailable. Differences from the primary comparison also involve membership, serialization, model identities and hardware, so they do not isolate the effect of overlap handling.
\begin{table}[H]\centering\small
\caption{Internal results for auxiliary models with model-visible overlap. FT is mean plus/minus sample SD across fits with varying selections; these scores are excluded from the primary comparison.}\label{sup:historical-internal}
\begin{tabular}{lrrrrr}\toprule Model & Base mF1 & FT mF1 & Gain & Base wF1 & FT wF1\\\midrule
\input{tables/table_revision_archived_internal.tex}\end{tabular}\end{table}
\begin{table}[H]\centering\footnotesize
\caption{Observed points and paired group-bootstrap intervals for the auxiliary models. Conditioning on fitted models does not eliminate training exposure.}\label{sup:historical-ci}
\begin{tabular}{llrrr}\toprule Model & Metric & Base & FT mean & Paired gain\\\midrule
\input{tables/table_revision_archived_ci.tex}\end{tabular}\end{table}

The auxiliary internal sensitivity excludes whole components touching the fitting or validation records, leaving 14,826 of 15,428 rows. The conservative legacy sensitivity leaves 11,986 of 12,000 appearances. These evaluation filters do not remove fitting/selection exposure and do not validate the primary comparison. The final full-data sparse macro-F1 of 0.8476 is independently recalculated from the retained predictions; implementation-audit details remain in the machine-readable evidence.
''']
    lines += [r'''
\subsection{Configuration screening for the shared QLoRA settings}
Qwen3-8B configuration screening compares six conditions on the full-text-grouped allocation: natural/balanced sampling, two learning rates, ranks 8/16 and one/two epochs. This limited comparison is not an exhaustive hyperparameter search. Natural sampling with two epochs, learning rate $2\times10^{-4}$ and rank 16 has the highest recorded validation macro-F1. The primary model comparison uses these fixed settings for both Qwen and Llama, without separate family-specific optimisation.

The screening fits use the full-text-grouped candidate allocation, whose model-visible overlap is quantified in Section S8.2. Their scores describe configuration screening rather than independent test performance or an optimal configuration. They are excluded from the primary model ranking. Run identities and source hashes identify all six conditions in the search-summary CSVs.
''']
    historical=historical_protocol_rows()
    lines += [table('Qwen configuration-screening conditions on the full-text-grouped allocation. Validation scores are auxiliary selection evidence, not primary-comparison test estimates.',
        'sup:historical-selection',['Condition','Sampling','LR','Rank','Epochs','Macro-F1','Debt F1'],
        [[r['condition'],r['sampling'],f"{r['learning_rate']:.0e}",r['rank'],r['epochs'],f(r['macro_f1']),f(r['debt_f1'])] for r in historical],
        'llrrrrr')]
    lines += [r'''
The 154,282-row partitioning pool contains 85,537 credit-reporting records (55.44\%), 22,503 debt-collection records (14.59\%), 18,367 mortgage/loan records (11.90\%), 14,943 card records (9.69\%) and 12,932 banking records (8.38\%). Its median narrative length was 365 characters and its 90th percentile was 1,261; 63.01\% fit within 500 characters. These statistics describe the candidate partitioning pool. The 63.98\% coverage in the main text instead describes all 162,405 usable source records. Retained and selected class counts are in Table~\ref{sup:classes}; the populations must not be interchanged.

\subsection{Validation scope}
Execution verification checks all 88 main-study stages, actual fitting memberships, the three independently rescored encoder searches, 19 main variants in both views, all same-checkpoint length and controlled-latency files, and source-integrity manifests. The reference audit checks all 12 extra stages, six actual refits, six reload parity tests and independent point metrics. Small hardware/API smoke fits (32 train/8 validation; two optimiser steps) are excluded from scientific results. Frozen prediction checks and tests support the reported computations, but do not establish prospective generalisation, semantic independence, an optimal prompt/budget, operational acceptability or regulatory readiness. The reproducibility package provides the source hashes, selections and recorded outputs for the stated verification scope.
\end{document}
''']
    (project_path(PAPER, 'supplementary_material.tex')).write_text(publication_terms('\n'.join(lines)))
    print('Supplementary source written with all completed main and reference-condition evidence.')


if __name__=='__main__':main()

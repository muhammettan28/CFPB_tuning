#!/usr/bin/env python3
"""CPU-only document integration from immutable, validated prediction artifacts."""
from __future__ import annotations
from project_paths import project_path
import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from revision_gpu_common import ROOT,PAPER,DATA,EXP,LABELS,load_frame,json_text,sha_file
from publication_language import publication_terms

OUT=project_path(PAPER, 'analysis/postprocessing')
TABLES=project_path(PAPER, 'tables/final')
NAMES={'qwen_base':'Qwen3 base','qwen_ft':'Qwen3 QLoRA','llama_base':'Llama 3.1 base',
       'llama_ft':'Llama 3.1 QLoRA','bert':'BERT','distilbert':'DistilBERT','roberta':'RoBERTa',
       'tfidf_10000':'TF-IDF/LR 10k','tfidf_full':'TF-IDF/LR full'}
CLASS_NAMES=['Retail banking','Credit reporting','Credit card','Mortgages/loans','Debt collection']


def read(path):return json.loads(Path(path).read_text())
def esc(text):
    result=str(text)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:result=result.replace(a,b)
    return result
def f(value):return f'{float(value):.4f}'
def ci(low,high):return f'[{f(low)}, {f(high)}]'
def row(cells):return ' & '.join(str(x) for x in cells)+r' \\'
def write_table(name,rows):
    TABLES.mkdir(parents=True,exist_ok=True)
    (TABLES/name).write_text('\n'.join(rows)+'\n'+r'\hline'+'\n')
def metric_frame():return pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_model_metrics.csv'))


def macro_contrasts():
    old=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_paired_contrasts.csv'))
    extra=pd.read_csv(OUT/'qlora_encoder_contrasts.csv')
    return pd.concat([old.loc[old.metric.eq('macro_f1')],extra.loc[extra.metric.eq('macro_f1')]],ignore_index=True)


def comparison_tables():
    contrasts=macro_contrasts();selected=contrasts.loc[contrasts.role.eq('internal')]
    write_table('internal_contrasts.tex', [row([NAMES[r.reference],NAMES[r.comparison],f(r.paired_difference),ci(r.low,r.high)]) for r in selected.itertuples(index=False)])
    costs=pd.read_csv(OUT/'computational_costs.csv')
    write_table('computational_costs.tex',[row([NAMES[r.system],f'{r.mean_fit_minutes:.2f}','CPU' if r.training_device=='CPU' else 'GPU',f'{r.mean_e2e_ms:.2f}','CPU' if r.inference_device=='CPU' else 'GPU']) for r in costs.itertuples(index=False)])


def facts():
    """Additional descriptive summaries only: no fitting, selection or inference."""
    from auto_postprocess_after_experiments import verify_evidence
    verify_evidence();TABLES.mkdir(parents=True,exist_ok=True)
    metrics=metric_frame()
    for role in ['internal','legacy']:
        selected=metrics.loc[metrics.role.eq(role)];lines=[]
        for system,group in selected.groupby('system',sort=False):
            g=group.set_index('metric');m=g.loc['macro_f1']
            sd='---' if pd.isna(m.seed_sd_ddof1) else f(m.seed_sd_ddof1)
            lines.append(row([NAMES[system],f'{int(m.train_rows):,}',f(m.point)+' '+ci(m.low,m.high),sd,
                              f(g.loc['accuracy','point']),f(g.loc['weighted_f1','point'])]))
        write_table(role+'_models.tex',lines)
    contrasts=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_paired_contrasts.csv'))
    selected=contrasts.loc[contrasts.role.eq('internal') & contrasts.metric.eq('macro_f1')]
    write_table('internal_contrasts.tex',[row([NAMES[r.reference],NAMES[r.comparison],f(r.paired_difference),ci(r.low,r.high)]) for r in selected.itertuples(index=False)])
    length=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/corrected_long_input.csv'))
    selected=length.loc[length.metric.eq('macro_f1') & length.comparison.eq('500_to_1500_chars')]
    write_table('long_macro.tex',[row(['Qwen3' if r.model=='qwen' else 'Llama 3.1',
        'base' if r.variant=='base' else 'QLoRA mean','All' if r.cohort=='all' else r'$>500$',
        f(r.reference_point),f(r.comparison_point),f(r.paired_difference),ci(r.low,r.high)]) for r in selected.itertuples(index=False)])
    latency=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/controlled_latency_summary.csv'))
    paired=pd.read_csv(project_path(PAPER, 'analysis/gpu_execution/controlled_latency_paired.csv'))
    lines=[]
    for model in ['qwen','llama']:
        group=latency.loc[latency.model.eq(model) & latency.condition.eq('chars500_tokens512') & latency.timing.eq('end_to_end_seconds')]
        base=group.loc[group.variant.eq('base'),'mean'].iloc[0];ft=group.loc[~group.variant.eq('base'),'mean'].mean()
        delta=paired.loc[paired.model.eq(model) & paired.condition.eq('chars500_tokens512') & paired.timing.eq('end_to_end_seconds')].iloc[0]
        lines.append(row(['Qwen3' if model=='qwen' else 'Llama 3.1',f(base),f(ft),f(delta.ft_seed_mean_minus_base_seconds),ci(delta.low,delta.high)]))
    write_table('latency_main.tex',lines)
    test=load_frame(DATA/'internal_test.csv');truth=test['product'].tolist();class_rows=[];off_rows=[];matrices={};view_rows=[]
    for system in NAMES:
        variants=['base'] if system.endswith('_base') else ['42','43','44'] if system not in {'tfidf_10000','tfidf_full'} else ['42']
        model=system.split('_')[0] if system.startswith(('qwen','llama')) else system
        stats=[];cms=[]
        for variant in variants:
            if system.startswith('tfidf'):
                budget='10000' if system=='tfidf_10000' else 'full'
                source=project_path(PAPER, f'experiments/tfidf_revision_v1/budget_{budget}/seed_42')
            else:source=EXP/'evaluation/evaluate'/model/variant/'suite'
            path=source/'internal_test_predictions.csv.gz'
            pred=pd.read_csv(path,keep_default_na=False,dtype={'record_id':str})
            if pred.record_id.tolist()!=test.record_id.astype(str).tolist() or pred.true_label.tolist()!=truth:raise RuntimeError('Descriptive internal alignment changed')
            stat=np.array(precision_recall_fscore_support(truth,pred.prediction,labels=LABELS,zero_division=0),dtype=float)
            stats.append(stat);cms.append(confusion_matrix(truth,pred.prediction,labels=LABELS+['__off_label__']))
            off_rows.append({'system':system,'variant':variant,'n':len(pred),'off_label_count':int(pred.prediction.eq('__off_label__').sum())})
        average=np.mean(stats,axis=0);matrices[system]=np.mean(cms,axis=0)
        for i,label in enumerate(LABELS):class_rows.append({'system':system,'label':label,'precision':average[0,i],
            'recall':average[1,i],'f1':average[2,i],'support':int(average[3,i])})
        # Per-file scores are descriptive; they are never independent replicates.
        for path in sorted((DATA/'legacy_subsets').glob('*.csv')):
            values=[]
            for variant in variants:
                if system.startswith('tfidf'):
                    full=pd.read_csv(source/'legacy_predictions.csv.gz',keep_default_na=False)
                    pred=full.loc[full.source_file.eq(path.name)]
                else:pred=pd.read_csv(EXP/'evaluation/evaluate'/model/variant/'suite'/f'{path.stem}_predictions.csv.gz',keep_default_na=False)
                from revision_gpu_common import metrics as score
                values.append(score(pred.true_label.tolist(),pred.prediction.tolist()))
            view_rows.append({'source_file':path.name,'system':system,**{key:float(np.mean([v[key] for v in values])) for key in ['accuracy','macro_f1','weighted_f1']}})
    pd.DataFrame(class_rows).to_csv(OUT/'corrected_class_metrics.csv',index=False)
    pd.DataFrame(off_rows).to_csv(OUT/'corrected_off_label_counts.csv',index=False)
    pd.DataFrame(view_rows).to_csv(OUT/'corrected_legacy_per_view.csv',index=False)
    np.savez_compressed(OUT/'corrected_confusion_counts.npz',**matrices)
    from render_publication_confusion import render
    render()
    # Full condition-level observed metrics, including actual seed refits.
    long_rows=[]
    for model in ['qwen','llama']:
        for variant in ['base','42','43','44']:
            for condition in ['chars500_tokens512','chars500_tokens2048','chars1500_tokens2048']:
                q=EXP/'evaluation/long-input'/model/variant/condition/'metrics.json'
                long_rows.append({'model':model,'variant':variant,'condition':condition,**read(q)['internal_test']})
    pd.DataFrame(long_rows).to_csv(OUT/'long_input_per_variant.csv',index=False)
    sources=[]
    for path in sorted((project_path(PAPER, 'analysis/gpu_execution')).glob('*.csv')):sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha_file(path)})
    for path in sorted((project_path(PAPER, 'analysis/ref14_encoder')).glob('*.csv')):sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha_file(path)})
    (OUT/'DOCUMENT_NUMERICAL_SOURCES.json').write_text(json_text(sources))
    print('Validated source metrics and generated descriptive-only class/confusion/view tables.')


def manuscript():
    comparison_tables()
    original=(project_path(PAPER, 'templates/manuscript_baseline.tex')).read_text()
    preamble=original.split(r'\begin{document}')[0]+r'\setlength{\emergencystretch}{2em}'+'\n'
    preamble += r'\interfootnotelinepenalty=10000'+'\n'
    preamble=preamble.replace('QLoRA Fine-Tuning for Consumer Complaint Classification', 'QLoRA fine-tuning for consumer complaint classification')
    # Preserve existing bibliography and related-work attribution, including [14]/[15].
    bibliography=original[original.index(r'\begin{thebibliography}'):original.index(r'\end{document}')]
    # Move online-only entries to footnotes; retain the order of all remaining works.
    for key in ['cfpbdata', 'mangrulkar2022peft', 'tiwari2021complaints']:
        bibliography=re.sub(r'\\bibitem\{'+key+r'\}[^\n]*\n', '', bibliography)
    bibliography=bibliography.replace('11 (3): 3606.', '11 (3): 3606-3615.')
    bibliography=bibliography.replace('Gewaly M, Saeed MS, Ammar MA.', 'Gewaly MM, Saeed MS, Ammar AM.')
    bibliography=bibliography.replace('Liu Y, Ouyang X, Bai R, Zhu X, Zhou Y et al. A framework', 'Yuan L, Ouyang X, Bai R, Zhu X, Zhou Y et al. A framework')
    bibliography=bibliography.replace('Aranguiz I', 'Aránguiz I')
    bibliography=bibliography.replace('Systems, and Applications; 2024. doi:', 'Systems, and Applications; 2024. pp. 12-17. doi:')
    bibliography=bibliography.replace('Algorithms and Control Engineering; 2024. doi:', 'Algorithms and Control Engineering; 2024. pp. 519-523. doi:')
    bibliography=bibliography.replace('Antalya, Türkiye; 2025. doi:', 'Antalya, Türkiye; 2025. pp. 1-6. doi:')
    related=original[original.index(r'\section{Related work}'):original.index(r'\section{Materials and methods}')]
    related=related.replace('Archived full-text grouping is audited after truncation; revised training/validation conflicts are quarantined while evaluation records are preserved.',
        'Model-visible conflicts are quarantined; fixed training/validation selections and evaluation records are retained across all model runs.')
    body=MAIN_BODY.replace('@@RELATED@@',related).replace('@@BIBLIOGRAPHY@@',bibliography)
    body=re.sub(r'(\\input\{tables/final/[^}]+\})\\hline',r'\1',body)
    for caption in ['Internal test', 'Internal macro-F1 paired differences']:
        body=body.replace(r'\begin{table}[htbp]'+'\n'+r'\caption{'+caption, r'\begin{table}[H]'+'\n'+r'\caption{'+caption)
    (project_path(PAPER, 'manuscript.tex')).write_text(publication_terms(preamble+r'\begin{document}'+'\n'+body))


MAIN_BODY=r'''
\maketitle
\begin{abstract}
We evaluate how QLoRA adaptation changes consumer financial complaint classification under a fixed labelled-data budget. Qwen3-8B and Llama 3.1-8B-Instruct are trained on the same 10,000 examples, with 2,000 validation examples and audited separation from evaluation inputs. On the 15,428-row internal test, three-seed mean macro-F1 rises from 0.6295 to 0.8331 for Qwen and from 0.6247 to 0.8402 for Llama. Paired group-bootstrap gains are 0.2035 [0.1934, 0.2141] and 0.2155 [0.2054, 0.2257]. Both adapted models exceed validation-tuned TF-IDF/Logistic Regression at the matched 10,000-example budget and all three matched-budget encoder baselines. Full-data TF-IDF reaches 0.8476 using 120,502 training examples, exceeding Qwen QLoRA; its paired difference from Llama QLoRA has a 95\% confidence interval containing zero. Adaptation substantially improves debt-collection recall, while extending inference text from 500 to 1,500 characters adds 0.0072 and 0.0055 macro-F1 for the adapted models. Controlled batch-one inference adds 0.1224 and 0.0802 seconds per request relative to the corresponding base models. QLoRA refits take about seven GPU-hours, versus 1.5--2.7 GPU-minutes for encoders and under one CPU-minute for sparse baselines. The results support choosing classifiers according to labelled-data and compute budgets.
\keywords{Consumer complaint classification, large language models, QLoRA, parameter-efficient fine-tuning, model-visible overlap, paired group bootstrap}
\end{abstract}

\section{Introduction}
The Consumer Financial Protection Bureau (CFPB) complaint database is a large public corpus of consumer narratives describing financial disputes, product failures, debt collection issues, credit reporting problems, and banking complaints. Accurate classification of such narratives is important for regulatory monitoring, institutional triage, and the prioritisation of consumer-service workflows. The task is challenging because complaint narratives often mention multiple financial products, contain noisy informal descriptions, and must nevertheless be mapped to a single constrained product category.

Recent work has explored large language models (LLMs) for zero-shot complaint classification. In zero-shot settings, models receive the candidate categories in the prompt and must return a valid label without task-specific training. This is attractive for complaint triage because new product taxonomies or institutional routing schemes may be introduced before large labelled datasets are available. However, zero-shot classification also raises practical issues: models can confuse overlapping financial products, output labels outside the allowed taxonomy, and show sensitivity to prompt and deployment details. Once a local zero-shot baseline is established for a given open-weight model, the next methodological question is how much performance can be gained through parameter-efficient fine-tuning under a fixed labelled-data budget.

This paper addresses that question. The central novelty is not a new model architecture or a new fine-tuning algorithm. G\"ull\"u et al. already fine-tuned GPT-2 and Llama 3.1-8B for complaints, including the same five financial categories \cite{gullu2025complaintllm}; Roumeliotis et al. studied zero-shot financial complaint classification \cite{roumeliotis2025think}. Our contribution is a reproducible applied evaluation of the paired zero-shot-to-QLoRA transition, including validation-tuned lighter classifiers and the benefits and costs of adaptation. The primary comparison uses fixed data selections and matched inference settings. Supplementary Section S8 details configuration screening and auxiliary dependence analyses.

The contributions are as follows:
\begin{enumerate}
    \item We compare base zero-shot and QLoRA-adapted models within the same Hugging Face quantised inference pipeline, holding prompt, labels and main input conventions fixed within each model family.
    \item We audit model-visible inputs, quarantine conflicting training/validation records, and preserve evaluation records and the source-related legacy views with their dependence explicitly accounted for.
    \item We evaluate Qwen3-8B and Llama 3.1-8B across optimisation seeds 42, 43 and 44 on fixed 10,000-example training and 2,000-example validation selections, alongside tuned sparse and encoder baselines.
    \item We report internal and legacy performance, paired group-bootstrap intervals, separate seed variation, off-label behaviour and class-level error patterns.
    \item We measure same-checkpoint longer-input effects and controlled fresh-call latency to quantify conditional adaptation trade-offs.
\end{enumerate}

@@RELATED@@

\section{Materials and methods}
\subsection{Data provenance and preserved evaluation views}
\label{sec:source}
The processed CFPB-related resource is attributed to Tiwari on Kaggle \footnote{\label{fn:data-sources}Consumer Financial Protection Bureau. Consumer complaint database [online]. Website \url{https://www.consumerfinance.gov/data-research/consumer-complaints/} [accessed 29 October 2026]. Tiwari S. Consumer complaints dataset for NLP [online]. Kaggle, 2021. Website \url{https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp} [accessed 29 October 2026].}. Its local archive contains 162,421 rows. The working file is identical by index, label and narrative for the first 162,415 rows; six trailing records contain only the placeholder narrative \texttt{name} and are absent. The archive-to-working-file correspondence identifies the analysed source records; the available records do not establish the undocumented preprocessing decisions. Removing ten rows with missing fields leaves 162,405 usable records. Excluding 8,123 benchmark-related source rows leaves a partitioning pool of 154,282 records. The upstream CFPB-to-Kaggle extraction and product-aggregation code is unavailable; source hashes identify the exact analysed input.

The five inherited labels are retail banking, credit reporting, credit card, mortgages and loans, and debt collection. No new upstream product aggregation is performed. Twelve preserved legacy files contain 1,000 appearances each: sampling-seed identifiers 1/2/3, balanced/imbalanced sampling, and 250/500-character views. Balanced files contain 200 examples per class. Each imbalanced file contains 511 credit-reporting, 182 debt-collection, 107 mortgage/loan, 100 card and 100 banking examples. These files are byte-identical to their source-archive members. The sampling implementation is unavailable; files and construction manifests are supplied. They are source-related legacy views, not independently collected external data.

\subsection{Model-visible separation and fixed selections}
\label{sec:revised-data}
Data allocation starts from an 80/10/10 split with seed 42 and full-normalised-narrative grouping. Because different full narratives can share a truncated input, label-blind connected components also link normalised 500-character prefixes and reconstructed 512-token inputs. A conservative 250-character prefix guard covers the benchmark views. Evaluation components take priority over fitting records; retained validation components take priority over conflicting training records. Table~\ref{tab:data-flow} reports the resulting exclusions and retained rows.
\begin{table}[htbp]
\caption{Data allocation with model-visible overlap screening. Evaluation records are excluded from fitting.}
\label{tab:data-flow}\centering\small
\begin{tabular}{lrrrr}\hline
Partition & Candidate rows & Excluded & Retained & Groups\\\hline
\input{tables/table_revision_data_flow.tex}
\end{tabular}
\end{table}

Exact-input audits find no shared training/validation-to-evaluation token sequences for the five pinned model tokenizers. Supplementary Section S1 gives the representation checks, audit sizes and longer-input checks; Section S4 describes dependencies among the benchmark views.

One naturally sampled 10,000-row train selection and one 2,000-row validation selection, with selection seed 42, are fixed for every model and optimisation seed. The shared 10k/2k allocation supports matched-data comparisons across model families. Optimisation seeds 42/43/44 change training randomness, not membership. Full-data TF-IDF uses all 120,502 retained training rows. The internal test contains 15,428 rows in 12,099 groups. Evaluation uses a fixed, inspected CFPB-related cohort and is retrospective rather than prospective. Data details and the overlap matrices are in Supplementary Sections S1 and S4.

\subsection{Pinned LLMs, prompting and QLoRA}
\label{sec:llm}
We use Qwen/Qwen3-8B \cite{yang2025qwen3} and meta-llama/Llama-3.1-8B-Instruct \cite{dubey2024llama}. Exact 40-character commits, tokenizer/config files and weight hashes are recorded; Supplementary Section S2 gives the identities. The recorded identities apply to the models in the primary comparison.

Base and adapted models share the same five-label prompt, 500 narrative characters, a 512-token input cap, NF4 4-bit double quantisation and FP16 compute. The 500-character limit holds the paired input convention fixed; 512 is the tokenizer ceiling, not another character limit. Within 500 characters, coverage is 63.98\% among the 162,405 usable source records and 63.01\% in the 154,282-row candidate partitioning pool; these percentages use different populations. Sections~\ref{sec:length-latency} and~\ref{sec:long-results} assess the effects of longer input. The chat template is rendered once and tokenised with \texttt{add\_special\_tokens=False}; Qwen thinking is disabled with \texttt{enable\_thinking=False}. SFT uses the same prompt prefix and completion-only target JSON/EOS; full serialization and target-length checks are detailed in Supplementary Section S2. Greedy generation uses at most 32 new tokens, one beam and no sampling. A regular expression extracts the final \texttt{product} field; absent or invalid labels remain off-label errors. This is JSON prompting with strict parsing, not token-level constrained decoding. No retry, synonym mapping or post-hoc label repair is used.

QLoRA uses rank 16, alpha 32, dropout 0.05 and all-linear targets, with frozen base weights and completion-only cross-entropy. Each fixed-data run has two epochs, learning rate $2\times10^{-4}$, physical batch 1, accumulation 16, paged AdamW 8-bit, cosine scheduling, warmup 0.03 and non-reentrant gradient checkpointing. Validation loss is measured every 100 optimiser steps; the best saved validation-loss checkpoint is retained. These settings are fixed for both LLM families, without test-based configuration selection. Each model has three independently initialised optimisation refits.

The shared configuration is supported by an exploratory Qwen validation comparison of sampling strategy, learning rate, rank and epoch count. Natural sampling, learning rate $2\times10^{-4}$, rank 16 and two epochs have the highest recorded validation macro-F1, 0.8235, among the six screened conditions. Supplementary Section S8 specifies the screening allocation and its limitations. These configuration-screening fits are auxiliary evidence; their validation scores are excluded from the primary test-set comparison. Llama uses the same settings without family-specific optimisation. In contrast, sparse and encoder baselines receive validation searches on the common fitting allocation. The matched-budget comparison therefore includes explicitly tuned baselines rather than relying on extra model-specific QLoRA tuning.

The primary model comparison uses an RTX 5060 Ti 16 GB, driver 595.99.02, Python 3.12.14, PyTorch 2.7.1+cu128 (CUDA 12.8) \cite{paszke2019pytorch}, Transformers 4.57.6 \cite{wolf2020transformers}, PEFT 0.17.1 \footnote{\label{fn:peft-software}Mangrulkar S, Gugger S, Debut L, Belkada Y, Paul S et al. PEFT: state-of-the-art parameter-efficient fine-tuning methods [online]. GitHub, 2022. Website \url{https://github.com/huggingface/peft} [accessed 29 October 2026].}, TRL 0.23.1, bitsandbytes 0.46.1 and Accelerate 1.10.1. Configuration-screening and auxiliary dependence analyses use the separately documented TRUBA/V100 environment; their scores are excluded from the primary comparison. Each run retains code, versions, manifests and fitted-weight hashes.

\subsection{Validation-tuned supervised baselines}
\label{sec:baselines}
TF-IDF plus Logistic Regression has separate 48-candidate validation searches at the 10k and full-data budgets. Like the neural systems, both sparse budgets use the first 500 narrative characters; the full-data condition increases training rows, not visible text per row. The grid crosses unigram/bigram features, maximum features 50,000/200,000, minimum document frequency 2/5, $C=0.25/1/4$, and no/balanced class weights. Vocabulary/IDF fit training text only. Lowercasing, Unicode accent stripping, sublinear TF and L2 normalisation are fixed. The lbfgs solver uses tolerance $10^{-5}$ and a 2,000-iteration cap; convergence warnings fail the run. Both winners use bigrams, minimum document frequency 5, $C=4$ and no class weighting; feature caps are 50,000 and 200,000. Validation macro-F1 is 0.7989 and 0.8363. Three identical seed-labelled refits at each budget are determinism checks, not stochastic optimisation replicates.

BERT-base-uncased, DistilBERT-base-uncased and RoBERTa-base use all 10k/2k selected rows. Each has six validation candidates: learning rate $2/3/5\times10^{-5}$ crossed with two/three epochs, search seed 42. The maximum final-epoch validation macro-F1 wins; ties within $10^{-12}$ use the smallest predefined candidate ID. Winners are BERT $2\times10^{-5}$/three epochs, DistilBERT and RoBERTa $5\times10^{-5}$/three epochs. Each is freshly refit with seeds 42/43/44 and evaluated at its final epoch. The DistilBERT and RoBERTa selections lie at the upper learning-rate and epoch boundaries of the tested grid; this search does not identify a global optimum. Main encoder inputs are 500 characters with a \textbf{512-token cap}, physical batch 4, accumulation 4, evaluation batch 16, AdamW, weight decay 0.01, linear scheduling, warmup 0.06 and FP16 training/inference. Reloaded predictions must exactly match all 2,000 saved validation predictions before test inference.

A reference-parameter sensitivity uses the encoder token caps and physical batch reported in Tables III--IV of G\"ull\"u et al., with the same selected learning rates and three epochs. Parameter provenance and all six extra refits are in Supplementary Section S7. The primary encoder baselines use 512-token inputs. Our data, 500-character text convention, PyTorch implementation and execution defaults differ from that source, so this is not an exact replication. No baseline is replaced on test performance.

\subsection{Metrics, dependence and uncertainty}
\label{sec:statistics}
The natural CFPB distribution is class-imbalanced, so macro-F1 is emphasised alongside accuracy and weighted F1. Accuracy, macro-F1 and weighted F1 use the same confusion-count estimator. Off-label outputs contribute errors rather than being dropped. Points are observed-data estimates; an adapted system is the mean of three separately scored refits, not an ensemble or a pooled confusion matrix. Run variation is the sample SD ($\mathrm{ddof}=1$). The main and supplementary analyses report stochastic seed SD separately from conditional sampling intervals.

We use 10,000 paired percentile group-bootstrap draws (RNG seed 20261020), sampling whole groups uniformly with replacement. All 19 main prediction variants share the same group multiplicities. Internal metrics retain row weights. Legacy analysis gives total weight one to each unique (true label, normalised visible narrative) across appearances: 5,909 units in 3,919 conservative full-text/prefix groups. Linked 250/500-character views are sampled together. The twelve overlapping file scores are dependent observations; statistical inference therefore operates on their linked groups. Paired differences are computed within each common resample. Intervals condition on fitted checkpoints and fixed selected configurations; they exclude training/model-selection uncertainty. They are descriptive 95\% intervals without a family-wise multiple-comparison adjustment.

\subsection{Longer inputs and controlled latency}
\label{sec:length-latency}
Each base and adapted checkpoint is evaluated on the same internal IDs under 500 characters/512 tokens, 500/2,048 (a token-cap control), and 1,500/2,048. Training uses 500 characters throughout. The predefined full-narrative $>500$ cohort contains 5,770 rows in 5,131 groups. Input retention and deterministic-response reuse are detailed in Supplementary Sections S2 and S5.

Controlled LLM latency uses fresh batch-one calls on 100 fixed internal IDs (20 per class), ten validation warmups and three repetitions. End-to-end time includes tokenisation, transfer, generation, decoding and parsing; model loading and disk I/O are excluded. CUDA synchronisation and an exclusive GPU-compute check define the timing boundary. Paired intervals use within-ID repetition means. Supplementary Section S6 gives the call-order, precision, output-length and job-block details.

For sparse and encoder baselines, CPU inference uses the same 100 IDs and repetition orders, four CPU threads and fresh calls. Encoders use FP32 and their seed-42 fitted checkpoints; sparse models retain their selected feature maps. Recorded neural training runtimes and fresh fixed-configuration sparse fits provide training wall times. Device and timing boundaries are reported separately so that these implementation costs are not interpreted as a controlled architecture-only speed comparison.

\section{Results}
\subsection{Model comparison}
\label{sec:main-results}
Table~\ref{tab:models} combines the fixed-data LLMs, tuned encoders and sparse baselines. Qwen macro-F1 rises from 0.6295 to 0.8331 and Llama from 0.6247 to 0.8402. Seed SD is 0.0048/0.0020 for the adapted models. Their paired macro-F1 gains are 0.2035 [0.1934, 0.2141] and 0.2155 [0.2054, 0.2257], conditional on these fitted checkpoints.
\begin{table}[htbp]
\caption{Internal test (15,428 rows). Macro-F1 brackets are 95\% paired-group intervals; SD is across three actual optimisation refits. Base systems have no adaptation rows. TF-IDF SD zero denotes deterministic checks. Full-data TF-IDF has a larger training budget.}
\label{tab:models}\centering\small\renewcommand{\arraystretch}{1.08}\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}\hline
System & Train rows & Macro-F1 [CI] & Seed SD & Accuracy & Weighted F1\\\hline
\input{tables/final/internal_models.tex}\hline
\end{tabular}
\end{table}

At the matched 10k budget, TF-IDF/LR reaches macro-F1 0.8068; QLoRA improves on it by 0.0263 and 0.0334, with paired intervals above zero (Table~\ref{tab:contrasts}). Encoder differences from the sparse baseline are smaller, with macro-F1 intervals spanning zero. Full-data TF-IDF reaches 0.8476; Qwen's difference from it is $-0.0145$, whereas Llama's $-0.0074$ interval includes zero (unrounded upper bound $6.8\times10^{-6}$). Direct matched-budget encoder contrasts are also positive: Qwen QLoRA exceeds the three encoders by 0.0212--0.0227 macro-F1 and Llama by 0.0283--0.0297; all six paired intervals exclude zero (Table~\ref{tab:contrasts}). The consolidated points and intervals share one estimator; bootstrap means are not substituted for observed points.
\begin{table}[htbp]
\caption{Internal macro-F1 paired differences, comparison minus reference. All intervals condition on fitted models. Rows using full-data TF-IDF provide unequal-training-budget context.}
\label{tab:contrasts}\centering\small\renewcommand{\arraystretch}{1.08}
\begin{tabular}{llrr}\hline
Reference & Comparison & Difference & 95\% CI\\\hline
\input{tables/final/internal_contrasts.tex}\hline
\end{tabular}
\end{table}

\subsection{Legacy views, classes and output compliance}
\label{sec:legacy-errors}
Unique-input-weighted legacy macro-F1 is 0.6324/0.8210 for base/adapted Qwen and 0.6271/0.8272 for Llama. Paired gains are 0.1886 [0.1723, 0.2051] and 0.2000 [0.1838, 0.2166]. Matched-budget TF-IDF gives 0.7871; full-data TF-IDF gives 0.8347. The overlapping views support a dependence-aware source-related analysis, not twelve independent replications or external generalisation. Supplementary Sections S3--S4 give all metrics, seeds and per-view descriptions.

Figure~\ref{fig:corrected-confusion} and the supplementary class tables describe the primary-comparison predictions. Debt-collection recall rises from 0.3262 to 0.7347 for Qwen and from 0.2754 to 0.7540 for Llama. Improvements are not uniform: Qwen credit-card recall changes from 0.7637 to 0.7550 despite its higher class F1. Of 15,428 internal predictions, Qwen base has one off-label output and each adapted refit has zero; Llama base and each refit have one. These are observed compliance counts, not enforced constrained decoding or evidence of prompt robustness. Residual confusions and different class supports remain important.

The dominant zero-shot error is the confusion of \texttt{debt\_collection} with \texttt{credit\_reporting}. Qwen makes this error 1,232 times before adaptation and 430.33 times after adaptation on average; the corresponding Llama counts are 1,167 and 403.33. These are reductions of 65.1\% and 65.4\%. Fine-tuning also reduces \texttt{retail\_banking} $\rightarrow$ \texttt{credit\_card} errors from 562 to 81 for Qwen and from 263 to 89 for Llama. Adapted counts are means of three refit count matrices; fractional values do not denote fractional complaints in a single run.

One residual issue remains visible in both models: \texttt{credit\_reporting} $\rightarrow$ \texttt{debt\_collection} errors increase from 142 to 344 for Qwen and from 87 to 336 for Llama. Higher debt-collection recall therefore coexists with more false assignments from the majority class. The main change is a redistribution of errors across neighbouring financial categories, rather than a uniform improvement in every decision boundary. These descriptive counts locate the remaining problem; they do not establish its causal mechanism or the operational cost of particular mistakes.
\begin{figure}[htbp]\centering
\includegraphics[width=0.90\linewidth]{figures/corrected_confusion_matrices.pdf}
\caption{Internal confusion fractions, normalised within true class. QLoRA matrices average the three refit count matrices before normalisation; performance metrics are means of separately scored runs. Class order and off-label counts are supplied in Supplementary Section S3.}
\label{fig:corrected-confusion}
\end{figure}

\subsection{Same-checkpoint longer-input results}
\label{sec:long-results}
The 500/512 and 500/2,048 control predictions are identical. Increasing visible text to 1,500 characters improves adapted Qwen/Llama macro-F1 by 0.0072/0.0055 overall and by 0.0156/0.0117 in the longer-narrative cohort (Table~\ref{tab:long}). Zero-shot Qwen instead declines; the zero-shot Llama interval includes zero. The result supports a conditional benefit of longer input for these adapted checkpoints, not a universal improvement or an evaluation of longer-context training. Supplementary Section S5 also reports auxiliary length-stratum descriptions and matched 250/500-character benchmark views; their cohort and fitting conditions differ from this controlled inference comparison.
\begin{table}[htbp]
\caption{Macro-F1 change for 1,500 versus 500 characters, both with a 2,048-token cap. ``All'' includes 15,428 rows; $>500$ is the predefined 5,770-row cohort. QLoRA is the three-seed mean.}
\label{tab:long}\centering\small\renewcommand{\arraystretch}{1.08}\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllrrrr}\hline
Model & Variant & Cohort & 500 & 1,500 & Difference & 95\% CI\\\hline
\input{tables/final/long_macro.tex}\hline
\end{tabular}
\end{table}

\subsection{Controlled inference cost and supplementary sensitivity}
\label{sec:cost-results}
For 500/512, mean end-to-end seconds are 0.2896/0.4120 for base/adapted Qwen and 0.2831/0.3633 for Llama (Table~\ref{tab:latency}). The additional mean times are 0.1224 [0.1166, 0.1284] and 0.0802 [0.0758, 0.0847]. These differences include actual generated lengths and adapter execution, not an isolated adapter-overhead experiment. The balanced 100-record cohort and serial batch-one calls are not an operational workload benchmark. Full per-seed median/p95, token counts and all conditions are in Supplementary Section S6.
\begin{table}[htbp]
\caption{Fresh-call end-to-end latency (seconds), 500 characters/512 tokens, RTX 5060 Ti. QLoRA mean averages the three variant means; paired intervals use within-ID repetition means.}
\label{tab:latency}\centering\small
\begin{tabular}{lrrrr}\hline
Model & Base mean & QLoRA mean & Added time & 95\% CI\\\hline
\input{tables/final/latency_main.tex}\hline
\end{tabular}
\end{table}

Table~\ref{tab:compute-cost} places these LLM timings alongside the measured CPU baselines and training wall times. The sparse models provide the lowest compute-cost option in these implementations. Encoder inference occupies an intermediate range on the CPU; adapted LLM inference uses the GPU. Supplementary Section S8 gives per-method fitting/search times, CPU median/p95 latency and measurement scopes.
\begin{table}[htbp]
\caption{Implementation-specific compute costs. Neural fitting uses the RTX 5060 Ti; sparse fitting and baseline inference use a four-thread CPU. Fit times are per refit; LLM inference is NF4/FP16 on GPU and encoder inference is FP32 on CPU. Device differences preclude architecture-only speed claims.}
\label{tab:compute-cost}\centering\small
\begin{tabular}{lrrrr}\hline
System & Fit min & Fit device & E2E ms & Inference device\\\hline
\input{tables/final/computational_costs.tex}
\end{tabular}
\end{table}

The separately declared Ref [14] conditions give internal macro-F1 0.8102 for BERT and 0.8095 for RoBERTa versus main values 0.8119/0.8105. Paired difference intervals span zero. The main RoBERTa condition uses 512 tokens. All source-specific parameter differences and three refits are reported in Supplementary Section S7 without selecting a replacement on test performance.

\section{Discussion}
\label{sec:discussion}
At the matched 10,000-example budget, QLoRA provides clear gains over each model's zero-shot baseline, validation-tuned sparse classification and all three encoder baselines. Llama QLoRA reaches macro-F1 0.8402 with 10,000 labelled training examples, compared with 0.8476 for TF-IDF trained on 120,502 examples: about one twelfth as many training labels. Its paired difference is $-0.0074$, with an interval spanning zero (unrounded upper bound $+6.8\times10^{-6}$). This is compatible with a small performance difference, rather than evidence of statistical equivalence or non-inferiority. The label-budget result must be weighed against the substantially greater fitting and inference cost of the LLMs. Where labels and compute are both available, full-data sparse classification remains a strong practical choice.

Model-visible separation is central to the comparison because full-text duplicate detection alone does not identify all collisions under truncation. Grouping the visible representations and excluding fitting conflicts provides an explicit basis for the paired evaluation. The auxiliary dependence analyses in Supplementary Sections S5 and S8 quantify the limitations of full-text grouping; their different memberships, model identities and environments preclude attributing cross-condition score differences solely to overlap handling.

The class-level pattern is particularly relevant for complaint triage. Debt collection narratives may mention disputed accounts, credit bureaus and credit reports, so the five-label task has closely related categories. The reduction in debt-to-reporting errors is accompanied by increased reporting-to-debt errors, which helps explain why a gain in aggregate macro-F1 does not resolve every routing concern. The balanced legacy views expose minority-class behaviour, while the imbalanced views retain different class prevalences; their overlap requires the shared-group analysis rather than treating them as independent confirmations. Human review, cost-sensitive training and class-specific calibration are possible directions for studying these residual errors, not evaluated safeguards in the present experiments.

Related work already establishes complaint LLM fine-tuning. Our distinction from G\"ull\"u et al. is the explicit NF4/LoRA and completion-only protocol, paired same-model baseline, fixed adaptation membership and separated uncertainty; from Roumeliotis et al., it is the adaptation comparison beyond zero-shot evaluation. Different splits, metrics and budgets preclude cross-paper performance ranking. Longer input helps the adapted checkpoints here, while zero-shot results show that more text alone is not a general solution. Latency intervals quantify conditional measured costs; their operational acceptability requires workload-specific evidence.

\section{Limitations}
\label{sec:limitations}
Evaluation is retrospective on previously inspected CFPB-related records; there is no independent-source, temporal or prospective generalisation test. Exact-input audits and conservative groups do not eliminate semantic duplication. Bootstrap intervals condition on fixed refits and the chosen protocol, excluding training/model-selection uncertainty. Three seeds give limited evidence about training variation, and multiple intervals are not family-wise adjusted. The 10k budget, fixed prompt and two 8B model families do not establish an optimal sample size, prompt robustness or universal classifier ranking.

The longer-input comparison changes inference text only; the models were trained with at most 500 characters. Latency uses serial unmerged-adapter jobs, a balanced 100-record sample and separate variant blocks, leaving thermal/background drift and output-length effects. CPU baseline and GPU LLM timings describe different implementations and precision settings; thermal/background drift can remain. Training wall times have method-specific validation/preprocessing boundaries, and encoder search totals cover training phases rather than full wall time. Energy, concurrent service capacity and total hardware lifecycle cost are not measured. Upstream CFPB-to-Kaggle preprocessing is unavailable. Some auxiliary fits also lack complete fitted weights and immutable upstream identities, limiting their reconstruction; Supplementary Section S8 states this scope separately from the fully identified primary models.

\section{Conclusion}
\label{sec:conclusion}
QLoRA raises internal macro-F1 by about 0.20 for two 8B model families under a fixed 10,000-example training budget. Both adapted models outperform the matched-data sparse and encoder baselines, and substantially improve debt-collection recall. The adapters also benefit from longer inference text despite being trained with 500-character inputs. Llama QLoRA approaches the observed full-data sparse score using about one twelfth of its training labels, at a higher computational cost. Together, the matched-data comparisons, class-level results and measured costs support a practical choice: sparse classifiers offer an efficient default, while QLoRA can provide stronger fixed-budget classification when an adapted generative model fits the intended workflow.
\section*{Acknowledgments}
The authors acknowledge the TRUBA ARF infrastructure operated by T\"UB\.{I}TAK ULAKB\.{I}M for computational resources supporting configuration screening and auxiliary analyses. The primary model comparison uses a local RTX 5060 Ti 16 GB.

\section*{Declaration of Generative AI and AI-assisted technologies}
During the preparation of this manuscript, the authors used AI-assisted tools for English grammar correction, wording refinement, LaTeX formatting support, analysis-code preparation, and literature comparison. The research idea, experimental design, dataset preparation, model fine-tuning, inference runs, statistical analyses, interpretation of results, and final scientific decisions were carried out by the authors. The authors reviewed and edited all AI-assisted text and take full responsibility for the content of the manuscript.

\section*{Data availability statement}
The source data derive from the CFPB complaint database and the processed Kaggle resource attributed to Tiwari \textsuperscript{\ref{fn:data-sources}}. Source/derived data, fixed selections, code, configurations, model identities, validation and final predictions, and statistical evidence are provided in the reproducibility repository at \url{https://github.com/muhammettan28/CFPB_tuning}. Supplementary Section S8 describes its contents and the CPU result-verification procedure. The repository excludes third-party literature and neural-model weights. Limitations concerning upstream preprocessing and auxiliary-model reconstruction are documented in Supplementary Section S8.

\section*{Funding}
The authors received no specific funding for this study.
\section*{Conflict of interest}
The authors declare no conflict of interest.

@@BIBLIOGRAPHY@@
\end{document}
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['facts','manuscript']);a=p.parse_args()
    {'facts':facts,'manuscript':manuscript}[a.phase]()


if __name__=='__main__':main()

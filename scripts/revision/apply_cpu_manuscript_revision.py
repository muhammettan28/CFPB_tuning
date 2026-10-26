#!/usr/bin/env python3
"""Integrate verified CPU results, preserving archived/GPU-pending distinctions."""
from __future__ import annotations
from project_paths import project_path

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from prepare_revision_protocol import ROOT, STUDY


def table(caption,label,columns,header,body,size=r'\small'):
    return '\n'.join([r'\begin{table}[htbp]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{center}',size,
        r'\begin{tabular}{'+columns+'}',r'\hline',header+r'\\',r'\hline',r'\input{tables/'+body+'}',r'\end{tabular}',r'\end{center}',r'\end{table}'])+'\n'


def main():
    paper=project_path(ROOT, 'REVISION_PAPER/manuscript.tex');text=paper.read_text()
    backup=STUDY/'before_cpu_revision';backup.mkdir(exist_ok=True)
    if not (backup/'manuscript.tex').exists():
        shutil.copy2(paper,backup/'manuscript.tex');shutil.copy2(paper.with_suffix('.pdf'),backup/'manuscript.pdf')
    baseline=pd.read_csv(STUDY/'tfidf_group_bootstrap.csv',dtype={'budget':str})
    def bf(budget):return baseline.loc[baseline.budget.eq(budget)&baseline.role.eq('internal_test')&baseline.metric.eq('macro_f1')].iloc[0]
    ten,full=bf('10000'),bf('full')
    lengths=json.loads((STUDY/'source_length_summary.json').read_text())
    abstract=r'''Consumer financial complaint classification requires balancing predictive performance, adaptation cost, and reproducible evaluation. This study examines the within-model transition from zero-shot inference to QLoRA adaptation for Qwen3-8B and Llama 3.1-8B, rather than introducing LLM fine-tuning to this task or claiming superiority over lighter classifiers. Reanalysis of archived three-seed experiments gives internal-test macro-F1 changes from 0.6294 to 0.8350 and from 0.6189 to 0.8378, respectively. However, a model-visible input audit identifies residual overlap after 500-character truncation in the original partitions. We therefore distinguish these historical results from a corrected-data protocol that preserves evaluation records and quarantines conflicting training and validation examples. We replace subset-level independence assumptions with paired group bootstrap analyses and report across-run variation separately from sampling uncertainty. A validation-tuned TF-IDF plus Logistic Regression baseline on the corrected training pool reaches @TF10@ macro-F1 with 10,000 training examples and @TFFULL@ with the full retained training pool. Corrected-data QLoRA and encoder reruns, a controlled longer-context experiment, and controlled latency measurements remain pending in this revision draft. The current evidence characterises adaptation gains and their evaluation limitations; it does not establish deployment acceptability or a final corrected-data ranking.
\keywords{Consumer complaint classification, large language models, QLoRA, parameter-efficient fine-tuning, model-visible overlap, group bootstrap, CFPB}'''
    abstract=abstract.replace('@TF10@',f'{ten["mean"]:.4f}').replace('@TFFULL@',f'{full["mean"]:.4f}')
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',text,count=1,flags=re.S)
    banner=r'\noindent{\small\textbf{Revision working draft.} CPU reanalysis and baseline tuning completed; corrected-data GPU experiments and immutable checkpoint confirmations are pending.}\par\medskip'
    if banner not in text:
        text=text.replace(r'\maketitle',r'\maketitle'+'\n\n'+banner,1)
    text=text.replace('under a fair and leakage-controlled protocol.','under an explicit adaptation and evaluation protocol.')
    if 'The archived model results and new corrected-data CPU baselines are reported separately;' not in text:
        text=text.replace('An internal test partition and preserved balanced and imbalanced legacy subsets provide complementary evaluation views.',
            'An internal test partition and preserved balanced and imbalanced legacy subsets provide complementary evaluation views. The archived model results and new corrected-data CPU baselines are reported separately; an updated model ranking awaits corrected-data GPU reruns.')
    # float must precede hyperref so caption targets are not duplicated.
    hyper=r'\usepackage{hyperref}'+'\n'+r'\hypersetup{colorlinks=true,urlcolor=black,citecolor=blue}'+'\n'
    text=text.replace(hyper,'')
    text=text.replace(r'\renewcommand{\arraystretch}{0.94}',r'\renewcommand{\arraystretch}{0.94}'+'\n'+hyper,1)
    text=text.replace('Train/validation/internal-test partitions and preserved legacy subsets; full-text grouping does not establish separation after truncation.',
        'Archived full-text grouping is audited after truncation; revised training/validation conflicts are quarantined while evaluation records are preserved.')

    methods=r'''\section{Materials and methods}
\subsection{Available source data and label provenance}
The complaint data are the processed CFPB-related resource attributed to Tiwari on Kaggle \cite{tiwari2021complaints,cfpbdata}. The local source archive contains 162,421 rows and three columns: an index, product, and narrative. The working file contains the first 162,415 rows identically by index, label and narrative; the six trailing source rows, IDs 162415--162420, contain only the placeholder narrative \texttt{name} and are absent. This establishes the exact six-row difference without claiming when or why the earlier copy was changed. Removing ten working-file rows with missing narrative or product fields leaves 162,405 usable rows. The earlier legacy exclusion removes 8,123 rows, leaving the original 154,282-row partitioning pool. Its archived allocation is 123,427 train, 15,427 validation, and 15,428 internal-test rows. Record IDs, labels and complete narratives in these partitions have been checked against the working file.

The five canonical labels are already present in the archived processed dataset and are retained unchanged (Table~\ref{tab:label-mapping}). No new original-product aggregation is performed by the current pipeline. The original CFPB-to-Kaggle preprocessing and aggregation code is not in this project; the archived source hash fixes the actual input artifact used here, without asserting byte identity with a newly downloaded Kaggle version. The project prior zero-shot manuscript independently records the same Tiwari source attribution.
'''
    methods+=table(r'Labels inherited unchanged from the archived processed resource. This is an identity mapping for the present pipeline, not a reconstruction of upstream CFPB product aggregation.',
        'tab:label-mapping','ll','Archived/current label & Task category','table_revision_label_mapping.tex')
    methods+=r'''

The twelve legacy files are byte-identical to the corresponding members of the same source ZIP and each contains 1,000 records. Balanced files contain 200 records per class; every imbalanced file contains 511 credit-reporting, 182 debt-collection, 107 mortgage/loan, 100 credit-card and 100 retail-banking records. Names encode sampling seeds 1, 2 and 3, balanced/imbalanced condition, and maximum narrative characters (250 or 500). The earlier design applies a 100-record minimum per class in the imbalanced condition. The revision uses the frozen files and source-row manifests rather than claiming to recover unavailable original sampling code. These are preserved legacy evaluation views from a related complaint source, not an independently collected external benchmark. They contain 5,909 unique normalised label--narrative pairs in 12,000 appearances.

In the available usable source, median narrative length is @MEDIAN@ characters and the 90th percentile is @P90@; @COVER500@\% of narratives are at most 500 characters. Coverage describes this corpus but does not establish that truncation preserves all classification information. The 500-character limit was originally a resource-budget choice; a controlled longer-context experiment remains necessary.

\subsection{Model-visible audit and revised partitions}
\label{sec:revised-data}
The original grouping uses SHA-256 of complete narratives after NFKC normalisation, lowercasing and whitespace normalisation. This prevents exact full-normalised-text overlap but does not guarantee separation after truncation. At 500 characters, the original training pool shares 192 unique inputs with 584 internal-test rows. Recorded Llama training selections confirm 383, 375 and 390 affected internal-test rows for seeds 42, 43 and 44. These are historical exposure counts, not the number of distinct original complaints.

We form label-blind connected components from shared full-normalised narratives, normalised first 500 raw characters, and the reconstructed 512-token inference inputs of the two archived tokenizers. Any row sharing a normalised first-250-character prefix with a legacy example is additionally linked through a conservative legacy guard. Such prefix components can join different complaints with common text and are not asserted to be verified complaint identities.

Allocation gives priority to the existing internal test and legacy records, which are retained byte-for-byte. Training and validation rows whose component touches evaluation are quarantined; remaining validation components are retained, and conflicting training rows are also quarantined. No evaluation record is moved into training. Table~\ref{tab:revision-data-flow} shows the resulting counts. Quarantine removes 3,232 additional rows, leaving 151,050 rows across the revised train/validation/internal-test partitions. The original 80/10/10 ratios are not reasserted after quarantine.
'''
    methods+=table(r'Original and revised partition counts. Evaluation records are preserved; conflicting training/validation rows are quarantined. Groups refer to the audited connected components.',
        'tab:revision-data-flow','lrrrr','Partition & Original & Quarantine & Retained & Groups','table_revision_data_flow.tex')
    methods+=r'''
Twenty-five representation/partition checks find zero shared audited inputs between retained training, validation and evaluation. The revised internal and legacy sets also have no shared components under this construction; legacy subsets still overlap each other. The cached tokenizer audit covers 123,551 unique raw 500-character inputs; none exceeds 512 tokens including its prompt. Llama keys use an archived final adapter tokenizer; Qwen keys use a pilot tokenizer because final tokenizer artifacts are absent. Base/final checkpoint identity and arbitrary semantic duplication are not certified by these checks. A new tokenizer or input limit requires another audit.

We freeze one naturally sampled 10,000-row training selection and one 2,000-row validation selection with selection seed 42, and retain them for all revised model families and optimisation seeds. Their IDs and hashes are recorded before baseline fitting. The full-data classical comparison uses all 120,502 retained training rows. Evaluation examples were already inspected in the earlier project and this revision; preservation does not make them a newly unseen prospective test.

\subsection{Models, prompting and archived adaptation settings}
The archived decoder models are Qwen3-8B \cite{yang2025qwen3} and Llama 3.1-8B Instruct \cite{dubey2024llama}. Base and adapted inference use the same five-label JSON prompt, 500 narrative characters, 512 input tokens, 4-bit NF4 quantisation with double quantisation and FP16 compute, greedy generation, and at most 32 new tokens. Qwen3 thinking is disabled through \texttt{enable\_thinking=False} in prompt construction. Outputs are extracted with a regular expression over the final \texttt{product} field; absent or invalid labels are counted as off-label errors. This is JSON-formatted prompting with strict parsing, not token-level constrained decoding. No retry or synonym-normalisation step is used.

The training script specifies LoRA rank 16, alpha 32, dropout 0.05, and \texttt{all-linear} targets; the stored Llama adapter configurations enumerate query, key, value, output, gate, up and down projections. Training uses completion-only loss on the target JSON label, two epochs, learning rate $2\times10^{-4}$, per-device batch size 1, gradient accumulation 16, paged AdamW 8-bit optimisation, cosine scheduling, warmup ratio 0.03, and non-reentrant gradient checkpointing. Validation loss selects the saved training checkpoint. The original Qwen validation-stage search selected this fixed protocol before its transfer to Llama; that historical search is not described as validation under the revised data allocation.

Historical metadata identify Python 3.10.20, PyTorch 2.7.1 with CUDA 11.8 \cite{paszke2019pytorch}, Transformers 4.57.6 \cite{wolf2020transformers}, PEFT 0.17.1 \cite{mangrulkar2022peft}, TRL 0.23.1, bitsandbytes 0.46.1 and Accelerate 1.10.1. Recorded Llama jobs use a Tesla V100-SXM2-16GB on TRUBA ARF. Immutable upstream checkpoint commit IDs were not retained; local checkpoint paths and null revision fields do not recover them. Final Qwen training metadata/adapters and complete selection manifests are also absent locally, so script settings and pilot metadata do not independently certify every final Qwen run.

\subsection{Validation-tuned classical baseline and pending encoder comparisons}
For TF-IDF plus Logistic Regression, separate 48-candidate searches are run for the fixed 10,000-row budget and full retained training pool. The grid crosses n-grams $(1,1)/(1,2)$, 50,000/200,000 maximum features, minimum document frequency 2/5, $C\in\{0.25,1,4\}$, and no/balanced class weights. Vocabulary and IDF are fitted only on training text; validation and evaluation data are transformed without fitting. Lowercasing, Unicode accent stripping, sublinear term frequency and L2 normalisation are fixed. The solver is lbfgs, with tolerance $10^{-5}$ and at most 2000 iterations; convergence warnings are treated as failures.

The same frozen 2,000 validation rows select the highest macro-F1 candidate for each budget, with a pre-specified candidate-order tie break. Both selected configurations are saved and hashed before opening test or legacy files for final prediction. Three seed-labelled refits (42, 43, 44) use the same rows. lbfgs does not use these random seeds for shuffling; identical predictions and zero across-refit SD are determinism checks, not three independent stochastic training outcomes. Sampling uncertainty is estimated separately by group bootstrap.

The archived BERT-base, DistilBERT and RoBERTa-base experiments remain available as historical records. Comparable validation search and three-seed reruns on the revised selections require GPU work and have not been performed. Consequently, no corrected-data QLoRA-versus-encoder-versus-TF-IDF ranking is claimed yet.

\subsection{Metrics, dependence and uncertainty}
Accuracy, macro-F1 and weighted F1 are computed from a single confusion-count implementation; off-label outputs contribute false negatives rather than being dropped. Observed-data point estimates are used in every summary and confidence table. The old discrepancy caused by reporting bootstrap sample means as point estimates is removed.

Paired percentile bootstrap intervals use 10,000 replicates and RNG seed 20261020. Complete dependence groups are sampled uniformly with replacement and all their rows retained; each replicate uses the same group multiplicities for base and all fine-tuned variants. Internal metrics retain row weighting. For legacy analysis each unique (label, normalised visible narrative) unit receives total weight one across repeated subset appearances; 250/500 views remain different inputs but shared-normalised-text/250-prefix components are sampled together. There are 5,909 such units in 3,919 conservative legacy groups. The twelve subset-level scores are not treated as independent replicates, and subset-level Wilcoxon significance claims are removed.

Archived fine-tuned performance is the mean of three separately scored runs, not an ensemble or a pooled confusion matrix. Their sample SD uses \texttt{ddof=1}. Because original training selections varied with seed, archived across-run variation includes data-selection and optimisation variation; it is not isolated optimisation-seed uncertainty. Bootstrap intervals condition on the archived trained models and do not integrate training randomness. Revised runs instead use fixed selected rows. The classical baseline intervals likewise condition on its fitted model and selection procedure; hyperparameter-selection uncertainty is not included.
'''
    methods=methods.replace('@MEDIAN@',f"{lengths['median_characters']:.0f}").replace('@P90@',f"{lengths['p90_characters']:.0f}").replace('@COVER500@',f"{lengths['percent_at_most_500']:.2f}")

    results=r'''\section{Results}
\subsection{Archived internal adaptation results}
Tables~\ref{tab:internal-summary} and~\ref{tab:bootstrap-ci} describe the original archived models, trained before model-visible quarantine. They are historical adaptation results with the overlap limitation in Section~\ref{sec:revised-data}, not corrected-data training results. Qwen3-8B changes from 0.6294 to 0.8350 internal macro-F1 and Llama 3.1-8B from 0.6189 to 0.8378. The paired group-bootstrap macro-F1 gains are 0.2056 [0.1955, 0.2159] and 0.2189 [0.2096, 0.2288]. These intervals describe sampling uncertainty conditional on the existing models and do not resolve the original exposure problem.
'''
    results+=table(r'Archived internal results. FT values are mean plus/minus sample SD over the three historical runs. These models were not retrained on revised partitions.',
        'tab:internal-summary','lrrrrr','Model & Base mF1 & FT mF1 & Gain & Base wF1 & FT wF1','table_revision_archived_internal.tex',r'\footnotesize')
    results+=table(r'Archived internal observed-data point estimates and 95\% paired group-bootstrap intervals (10,000 replicates). Point estimates share the computation used in Table~\ref{tab:internal-summary}.',
        'tab:bootstrap-ci','llrrr','Model & Metric & Base & FT mean & Paired gain','table_revision_archived_ci.tex',r'\footnotesize')
    results+=r'''
\subsection{Legacy dependence and sensitivity analyses}
Figure~\ref{fig:legacy-overlap} shows shared unique normalised narratives; both narrative-only and label--narrative matrices, row-relative fractions and Jaccard values are retained in the analysis files. The apparent twelve evaluation replications include repeated inputs and cannot support an independence assumption. The unique-input-weighted, group-bootstrap legacy results in Table~\ref{tab:legacy-summary} use a different estimand from averaging twelve overlapping subset scores; the two summaries must not be interchanged.
\begin{figure}[htbp]
\centering
\includegraphics[width=0.90\linewidth]{figures/revision_legacy_overlap.pdf}
\caption{Pairwise shared unique normalised narratives across the twelve legacy subsets. B/I denote balanced/imbalanced sampling; the other fields identify sampling seed and character limit. Diagonal values count unique inputs within a subset.}
\label{fig:legacy-overlap}
\end{figure}
'''
    results+=table(r'Archived legacy macro-F1 with equal weight per unique visible label--narrative unit, clustered into 3,919 conservative groups. FT SD reflects the historical runs; gain intervals use paired group resampling.',
        'tab:legacy-summary','lrrr',r'Model & Base & FT mean$\pm$SD & Gain [95\% CI]','table_revision_archived_legacy.tex')
    results+=r'''
The internal sensitivity analysis removes whole test components matching either original training or validation pool, leaving 14,826 of 15,428 rows. A conservative legacy sensitivity excludes groups with a 250-character prefix shared with the original train/validation pools, leaving 11,986 of 12,000 appearances. Table~\ref{tab:sensitivity} reports these analyses. Retained adaptation gains are a retrospective sensitivity result; filtering evaluation rows cannot repair earlier training or model selection, and does not replace corrected-data reruns.
'''
    results+=table(r'Archived-model sensitivity results after conservative evaluation-group exclusions. These are post-hoc subsets, not a repaired training experiment.',
        'tab:sensitivity','llrrr',r'Model & Evaluation view & Rows & FT mF1 & Gain [95\% CI]','table_revision_sensitivity.tex',r'\footnotesize')
    results+=r'''
\subsection{Revised-data TF-IDF baseline}
The classical search configurations and results are reported separately from the archived LLM experiments in Tables~\ref{tab:tfidf-selection} and~\ref{tab:supervised-baselines}. The new baseline uses the revised training pool and frozen validation selection. Historical LLM and encoder scores cannot serve as a matched corrected-data comparison, even though internal evaluation records are preserved. A consolidated new-model ranking remains pending GPU reruns. All three lbfgs refits produce identical predictions at each budget; their score SD is zero by determinism and is not evidence of negligible stochastic training uncertainty.
'''
    results+=table(r'Validation-selected TF-IDF configurations; 48 candidates per training budget and a fixed 2,000-row validation set. No test metric enters selection.',
        'tab:tfidf-selection','rrrrrlr','Train rows & N-grams & Max feat. & Min df & C & Weight & Val mF1','table_revision_tfidf_selection.tex',r'\footnotesize')
    results+=table(r'New corrected-data classical baselines. Internal macro-F1 brackets are 95\% group-bootstrap intervals; legacy macro-F1 equally weights unique visible label--narrative units. No revised-data GPU score is available yet.',
        'tab:supervised-baselines','lrrrrr','Model & Train rows & Accuracy & Internal mF1 [CI] & wF1 & Legacy mF1','table_revision_tfidf.tex',r'\footnotesize')
    results+=r'''
\subsection{Narrative length and matched legacy views}
Table~\ref{tab:length-strata} describes archived internal performance by the original narrative length. These cohorts differ in complaint content and class composition, so their differences cannot be interpreted as a causal truncation effect.
'''
    results+=table(r'Descriptive archived internal macro-F1 by full narrative length in characters. FT is the historical three-run mean; every evaluated narrative is capped at 500 characters.',
        'tab:length-strata','llrrr','Model & Length & Rows & Base & FT mean','table_revision_length_strata.tex')
    results+=r'''
For a closer retrospective view, 250- and 500-character legacy files are matched within sampling seed and balanced/imbalanced condition using label and normalised 250-character prefix. Only one-to-one within-file matches are retained; ambiguous matches are excluded before examining predictions. Duplicate appearances share total weight one per distinct short/long label--view pair, and linked short/long views are resampled together. Table~\ref{tab:paired-truncation} uses only pairs whose visible input changes. Such matching is not a verified underlying CFPB complaint identity and does not create a prospective controlled experiment. An experiment exceeding 500 characters still requires new model inference and remains pending.
'''
    results+=table(r'Archived 250/500 matched-view macro-F1 on unambiguous changed-input pairs. Gains are 500 minus 250 with 95\% paired group-bootstrap intervals; pairs are weighted uniquely across appearances.',
        'tab:paired-truncation','llrrrr','Model & Variant & Unique pairs & 250 chars & 500 chars & Gain [CI]','table_revision_paired_truncation.tex',r'\footnotesize')
    results+=r'''
\subsection{Archived errors, output parsing and latency}
The archived confusion matrices in Figure~\ref{fig:confusion-matrices} describe the historical models. Fine-tuning reduces debt-collection-to-credit-reporting errors, but these matrices are not evidence that the revised training protocol has already been evaluated. Near-zero off-label counts describe observed compliance with the fixed prompt and parser; they do not establish token-level constrained generation or robustness to different taxonomies or prompts.
\begin{figure}[htbp]
\centering
\includegraphics[width=0.94\linewidth]{figures/internal_confusion_matrices.pdf}
\caption{Archived internal confusion matrices. FT matrices average the three historical runs. These figures precede corrected-data training and retain the archived evaluation limitations.}
\label{fig:confusion-matrices}
\end{figure}

The archived serial generation-loop means are 0.603/0.912 seconds per complaint for base/FT Qwen and 0.487/0.808 for base/FT Llama. The differences of 0.309 and 0.321 seconds are descriptive measurements from the original inference loop, excluding model loading. No controlled warmup/repetition protocol, matched batching study or new hardware comparison was conducted in this CPU revision. These means do not establish throughput, workload acceptability, regulatory readiness or deployment suitability.

\section{Discussion}
The archived data show substantial within-model adaptation gains, including under conservative group-based sensitivity analyses. The reanalysis also identifies a concrete failure of complete-narrative grouping to prevent model-visible overlap. These two observations must be reported together: conditional sampling intervals can quantify archived differences without validating the original train/test separation.

G\"ull\"u et al. \cite{gullu2025complaintllm} already demonstrate supervised LLM adaptation for the same financial categories, while Roumeliotis et al. \cite{roumeliotis2025think} motivate the zero-shot setting. The present contribution concerns the specified same-model QLoRA transition, seed/run variation, and adaptation costs. Differences in data allocation, training budget and F1 averaging preclude a numerical cross-paper ranking. The new tuned classical baseline provides a stronger revised-data reference, but deciding whether QLoRA offers a useful accuracy/cost trade-off requires matched corrected-data GPU runs.

A generative model may be useful in a broader organisational workflow, but cross-task reuse and changed-taxonomy adaptation were not tested. The evidence is limited to the stated five-label classification task. The fixed 10,000-row budget is a controlled resource choice, not an optimal training size or an upper bound on attainable performance. The 500-character cap likewise remains a protocol constraint; corpus coverage and retrospective matched views do not replace a longer-context test. Local execution can be technically feasible without establishing operational or regulatory readiness.

\section{Limitations}
\label{sec:limitations}
The archived LLM/encoder runs used the original partitions and cannot be converted into corrected-data runs by re-scoring. Archived seeds also changed selected training examples, so their SD is across-run rather than pure optimisation-seed variation. New GPU baseline tuning, fixed-selection three-seed training, a longer-context experiment and controlled latency measurements are outstanding. Bootstrap intervals condition on fixed predictions and the selected protocol; they exclude training- and model-selection uncertainty.

Prefix groups are conservative dependency controls, not verified complaint identities. The model-visible audit covers the stated archived tokenizer artifacts and transformations; final Qwen checkpoint identity is incomplete. Full semantic duplication, new tokenizers and longer-input transformations need additional checks. Reusing previously inspected evaluation records is retrospective; all evaluations remain CFPB-related and do not establish independent-source or temporal generalisation. The original pre-Kaggle extraction/product-aggregation code and immutable model checkpoint revisions are not recovered, although the local 162,421-to-162,415-row source derivation and inherited labels are verified. Prepared local code and prediction artifacts do not by themselves establish public or reviewer access.

\section{Conclusion}
This revision separates archived adaptation evidence from a newly audited training protocol and validation-tuned CPU baseline. The historical Qwen and Llama results show large within-model zero-shot-to-QLoRA changes, accompanied by residual original input overlap and dependent legacy evaluations. Group-based resampling and consistent point-estimate computation clarify what can be inferred from those archived predictions. The revised partitions and fixed sample manifests prepare a matched next-stage evaluation; corrected-data GPU experiments remain pending. The practical contribution should therefore be assessed through verified adaptation properties, classical-baseline competitiveness and measured resource costs, without claiming universal classification superiority, broad robustness or deployment readiness.
'''
    start=text.index(r'\section{Materials and methods}');end=text.index(r'\section*{Acknowledgment')
    text=text[:start]+methods+'\n'+results+'\n'+text[end:]
    start=text.index(r'\section*{Declaration of Generative AI');end=text.index(r'\section*{Data availability statement}')
    text=text[:start]+r'''\section*{Declaration of Generative AI and AI-assisted technologies, if any}
AI-assisted tools supported revision planning, analysis-code preparation, literature comparison, wording and LaTeX formatting. Numerical results were computed from recorded data and checked through deterministic recomputation and tests; new GPU experiment outcomes were not generated by an AI tool. The authors retain responsibility for source verification, interpretation, final review and the submitted manuscript.

'''+text[end:]
    start=text.index(r'\section*{Data availability statement}');end=text.index(r'\section*{Funding, if any}')
    text=text[:start]+r'''\section*{Data availability statement}
The available derived data, original and revised split/selection manifests, analysis scripts, archived prediction records and new classical-baseline configurations are assembled in a local reviewer package. This package has not yet been published or supplied to reviewers; an actual access mechanism remains to be arranged before submission. Third-party literature and base-model weights are not redistributed in the package. Source-derivation and checkpoint-identity gaps are documented rather than represented as complete reproducibility.

'''+text[end:]
    if r'\bibitem{tiwari2021complaints}' not in text:
        entry=r'\bibitem{tiwari2021complaints} Tiwari S. Consumer complaints dataset for NLP. Kaggle, 2021. Website https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp [project archive].'
        text=text.replace(r'\end{thebibliography}',entry+'\n\n'+r'\end{thebibliography}')
    paper.write_text(text)
    print('Integrated CPU analyses; archived results remain distinct from corrected-data fits and pending GPU work.')


if __name__=='__main__':main()

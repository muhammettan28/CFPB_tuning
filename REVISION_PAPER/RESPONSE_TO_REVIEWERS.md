# Response to reviewers

Thank you for the constructive reviews. We have revised the manuscript and expanded the experiments to clarify the benefits, costs and limitations of QLoRA for complaint classification. The principal changes are summarised below, followed by a point-by-point response with manuscript and supplementary locations.

- Audited model-visible overlap, excluded fitting conflicts and repeated the primary experiments on fixed selections while retaining all evaluation records.
- Tuned sparse and encoder baselines, repeated neural fits across three seeds and added direct paired QLoRA-versus-encoder comparisons.
- Replaced subset-level independence assumptions with unique-input weighting and paired group-bootstrap intervals.
- Added longer-input inference, controlled LLM latency, CPU baseline timings and training/search cost records.
- Clarified related-work positioning, model serialization and hardware differences, and provided the code, data selections and predictions in the public reproducibility repository.

## Reviewer #1

### R1.1 — Legacy dependence

> Legacy statistics: The 12 subsets contain only 5,909 unique pairs, so the Wilcoxon test (p = 2⁻¹²) assumes independence that isn’t shown. Provide an overlap matrix and use a group-aware paired analysis.

**Response.** We agree that the twelve files are not twelve independent experiments and have removed the subset-level Wilcoxon inference. Their 12,000 appearances contain 5,909 unique normalised label–narrative units. We supply both complete 12-by-12 overlap-count matrices, the file key and per-view descriptive scores. The aggregate equally weights unique units rather than repeated appearances. Label-blind full-text/prefix components link dependent units into 3,919 groups. The analysis resamples whole groups, with the same draw for every compared system. These are conservative text-dependency groups, not verified complaint identities. On the corrected-data legacy aggregate, Qwen and Llama macro-F1 adaptation gains are 0.1886 [0.1723, 0.2051] and 0.2000 [0.1838, 0.2166]. The intervals condition on fitted models; the files remain source-related views, not independent-source validation.

**Changes and locations.** Manuscript Section 3.5 (p. 6); Manuscript Section 4.2 (p. 8); Supplementary material Section S4 (p. 11); Supplementary material Table S17 (p. 11); Supplementary material Table S18 (p. 11); Supplementary material Table S13 (p. 9).

### R1.2 — Model-visible overlap and corrected allocation

> Leakage audit: Splits group full narratives, but models see 500-character truncations. Audit overlap after truncation and prompt construction, and narrow the “no leakage” claim.

**Response.** The additional audit found that full-narrative grouping had not prevented all model-visible collisions: 192 shared unique 500-character training inputs touch 584 internal-test rows in the archived allocation. We therefore do not describe the archived experiments as leakage-free. Label-blind connected components now include normalised full text, 500-character representations, reconstructed tokenizer inputs and a conservative legacy 250-character prefix guard. Giving existing evaluation records priority quarantines 2,925 training and 307 validation rows; no evaluation record is moved into fitting. We repeated the affected model study on the corrected allocation. Each of five actual pinned tokenizers was audited on 39,428 inputs, including the identical 10k/2k selections and all evaluation appearances; audited train/validation-to-evaluation token-identity intersections are empty. Longer-input and separate reference-parameter keys were also checked. Our claim is limited to the representations tested, with no semantic/paraphrase independence guarantee. The previously inspected internal test remains retrospective.

All 15,428 test record IDs and labels are unchanged. Zero-shot predictions were regenerated in the same pinned inference pipeline as the adapted models rather than copied from the submitted runs. Llama base macro-F1 changes from 0.6189 to 0.6247, with debt-to-reporting errors changing from 1,401 to 1,167 and banking-to-card errors from 333 to 263. Qwen base changes only from 0.6294 to 0.6295 (15 changed predictions, compared with 1,102 for Llama). This cannot be caused by removing training records from a model that is not trained. The source audit identifies a serialization change: the submitted code tokenises an already rendered chat template with default special-token insertion, while the paired pipeline uses add_special_tokens=False. CPU reconstruction with the pinned primary tokenizers yields two versus one leading BOS tokens for Llama; Qwen token IDs are unchanged for that audited prompt. Upstream model identities and the GPU/numerical environment also differ. We identify these implementation differences without attributing the score change to one factor in the absence of an isolated ablation.

Adapted-model macro-F1 changes from 0.8350 to 0.8331 for Qwen and 0.8378 to 0.8402 for Llama. Differences before rounding are -0.001984 and +0.002310, both within 0.003. These are descriptive cross-protocol differences, not estimates of the effect of leakage removal alone. The separate 14,826-row auxiliary filtering analysis gives base/adapted macro-F1 0.6290/0.8353 for Qwen and 0.6190/0.8381 for Llama; filtering test groups does not undo fitting or selection exposure.

**Changes and locations.** Manuscript Section 3.2 (p. 4); Manuscript Table 2 (p. 4); Supplementary material Section S1 (p. 2); Supplementary material Table S3 (p. 2); Supplementary material Section S7 (p. 16); Supplementary material Section S8 (p. 17).

### R1.3 — Data provenance and reproducibility access

> Reproducibility: Explain how the 162,415 rows were derived, give the label mapping and subset construction, and release code, manifests and predictions to reviewers.

**Response.** We reconstructed the local derivation: the source archive contains 162,421 rows, and its first 162,415 records match the working file by index, label and narrative. The six absent trailing records contain only the placeholder narrative “name”; the available evidence does not establish who removed them or when. Excluding ten missing-field records and 8,123 benchmark-related source rows yields the 154,282-row candidate partitioning pool. Model-visible overlap screening gives retained train/validation/internal counts of 120,502/15,120/15,428. We document the inherited five-label identity mapping, class counts, fixed selections and byte-identical benchmark files, including their balanced/imbalanced and 250/500-character conditions. The unavailable upstream CFPB-to-Kaggle extraction/product aggregation and sampling implementation remain disclosed limitations. Source/derived data, code, manifests, configurations, search records and saved predictions are provided in the public repository: https://github.com/muhammettan28/CFPB_tuning. Supplementary Section S8 and the repository README give the CPU replay procedure. It verifies 404 source hashes and recomputes nine complete main/reference result tables, including paired intervals, seed scores, length effects and latency, without neural weights or a GPU. Neural weights are excluded; exact primary-model commits and execution environments are recorded.

**Changes and locations.** Manuscript Section 3.1 (p. 3); Manuscript Section 3.2 (p. 4); Manuscript Section 6 (p. 11); Supplementary material Section S1 (p. 2); Supplementary material Table S16 (p. 11); Supplementary material Section S8 (p. 17).

### R1.4 — Implementation and immutable model identities

> Implementation details: State the Qwen3 thinking setting, checkpoint IDs, full QLoRA configuration and library versions.

**Response.** The revised Methods specify Qwen enable_thinking=False, single chat-template rendering without duplicate BOS, greedy decoding, one beam and a maximum of 32 new tokens. JSON prompting is followed by final-product-field regex parsing, with invalid/absent labels counted as errors and no retries. The new QLoRA runs use 4-bit NF4, double quantisation, FP16 compute, rank 16, alpha 32, dropout 0.05, all-linear targets and completion-only JSON/EOS loss. They use two epochs, learning rate 0.0002, physical batch 1/accumulation 16, paged AdamW 8-bit, cosine scheduling, warmup 0.03, non-reentrant checkpointing and validation-loss checkpoint selection. The supplement includes the exact prompt, full 40-character model commits and actual library/hardware versions; per-run files retain code, environment and weight hashes. Exact-input memoisation is disclosed for deterministic prediction generation only; latency and warmups make fresh calls. The primary reruns use a local RTX 5060 Ti 16 GB with CUDA 12.8; the submitted TRUBA computations used V100 hardware with the separately recorded software environment. The manuscript and supplement state the environments explicitly. New model pins identify the reruns but cannot recover missing submitted-model upstream commits or final Qwen artifacts.

**Changes and locations.** Manuscript Section 3.3 (p. 4); Supplementary material Table S4 (p. 3); Supplementary material Table S5 (p. 3); Supplementary material Table S6 (p. 4); Supplementary material Section S8 (p. 17).

### R1.5 — Matched baselines, searches and separate reference sensitivity

> Baselines: Tune the encoders and TF-IDF comparably (three seeds, validation search), and compare in one table with uncertainty. Margins over TF-IDF are small.

**Response.** All revised families share the same 10,000 training and 2,000 validation records. TF-IDF/LR has a 48-candidate validation search at each of the 10k and full-data budgets. BERT, DistilBERT and RoBERTa each have six validation candidates (three learning rates by two epoch counts), followed by fresh refits with optimisation seeds 42/43/44. All candidates and selected configurations are supplied. Sparse lbfgs refits are deterministic checks, not stochastic seed evidence. One main table now reports corrected-data accuracy, macro-F1, weighted F1, macro-F1 interval and separate seed SD. Qwen/Llama adapted macro-F1 is 0.8331/0.8402 versus 0.8068 for 10k TF-IDF; paired differences are 0.0263 [0.0199, 0.0327] and 0.0334 [0.0270, 0.0397]. Full-data TF-IDF reaches 0.8476 with 120,502 training rows and is explicitly unequal-budget context. Encoder-versus-10k-TF-IDF macro-F1 intervals include zero. We also added direct QLoRA-minus-encoder contrasts using the same 10,000 common group draws, averaging all three refits in each system before subtraction. Qwen gains are 0.0212--0.0227 and Llama gains 0.0283--0.0297 across BERT, DistilBERT and RoBERTa; all six internal paired intervals exclude zero. The full metric/view contrasts are in the supplementary table. The DistilBERT/RoBERTa winners are at the upper tested learning-rate and epoch boundaries; we now state that the finite grid does not establish a global optimum. We additionally report six refits using the encoder caps/physical batch in Ref. [14] Tables III–IV: BERT 128 tokens and RoBERTa 100 tokens, batch 16/accumulation 1. The main encoders remain at the inherited 500 narrative characters with a 512-token ceiling, batch 4/accumulation 4. Ref. [14] conditions are separate sensitivity runs, not a new universal RoBERTa limit or an exact replication. Internal macro-F1 differences from main are -0.0017 (BERT) and -0.0010 (RoBERTa), with paired intervals including zero; no main baseline was replaced using test results.

**Changes and locations.** Manuscript Section 3.4 (p. 5); Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Supplementary material Table S7 (p. 4); Supplementary material Table S8 (p. 5); Supplementary material Table S13 (p. 9); Supplementary material Section S7 (p. 16); Supplementary material Table S28 (p. 16).

### R1.6 — Paired bootstrap, point estimates and seed variation

> Bootstrap: Use a paired, group-level bootstrap with more replicates, report seed variation separately, and reconcile Tables 2 and 3.

**Response.** The revised analysis uses 10,000 paired whole-group bootstrap draws (seed 20261020), with shared group multiplicities across compared variants. Internal estimates score all 15,428 rows in 12,099 dependency groups. Legacy estimates use the unique-unit weighting described above. Points are computed on the observed data before resampling; means over three refits are means of separately scored models, not ensembles. Across-refit sample SD is reported separately from the conditional bootstrap interval. The interval does not include training or hyperparameter-selection uncertainty; selection/memberships are fixed and no multiplicity adjustment is claimed. The supplement explains the computationally equivalent multinomial compression of identical joint group-confusion profiles. The separate reference analysis uses its own common draws and compression, so the same random seed does not imply identical Monte Carlo draws between analyses. Historical Tables 2/3 are reconciled with the same observed estimator in explicitly historical-only supplementary tables. Original and corrected-training results are not mixed into one ranking. The supplement now also gives the explicit confusion-count, seed-average and paired resampling equations; the CPU replay checks all reported estimates against saved predictions.

**Changes and locations.** Manuscript Section 3.5 (p. 6); Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Supplementary material Section S3 (p. 6); Supplementary material Table S11 (p. 8); Supplementary material Table S32 (p. 18); Supplementary material Table S33 (p. 18).

### R1.7 — Contribution and related work

> Contribution and claims: Compare explicitly with [14] and [15], and remove or test claims about reuse, “external” benchmark and robustness.

**Response.** We expanded the comparison with Güllü et al. [14], explicitly acknowledging their financial five-category task and Llama-3.1 fine-tuning. The difference is our within-model zero-shot-to-explicit-NF4-QLoRA question for two 8B families, fixed labelled-data budget, actual refits and dependence-aware corrected evaluation—not first use of LLMs, a new taxonomy or a new PEFT algorithm. Missing descriptions in [14] are described as reporting differences, not proof of absent procedures. Their weighted scores and different split/budget are not ranked against our macro-F1. Roumeliotis et al. [15] is distinguished as zero-shot evaluation; its exclusion of narratives over 500 characters differs from our inherited truncation convention. “External benchmark” is replaced by source-related legacy views. Untested taxonomy reuse, independent-source generalisation and broad robustness claims have been removed. The discussion centres on measured adaptation gains, baseline competitiveness and costs.

**Changes and locations.** Manuscript Table 1 (p. 3); Manuscript Section 3.1 (p. 3); Manuscript Section 5 (p. 10); Manuscript Section 6 (p. 11); Supplementary material Section S7 (p. 16).

### R1.8 — Truncation and controlled latency

> Truncation and deployment: Report results by truncation and add a longer-context experiment. Control the latency measurements and avoid claiming acceptability or regulatory readiness.

**Response.** The 500-character narrative cap is inherited from the original protocol; 512 is its tokenizer ceiling. We do not claim that 500 characters is optimal, harmless or necessary for this GPU. Only 63.98% of usable source narratives fit that cap. We report historical length strata and matched changed-input views separately, and completed the same-checkpoint inference comparison at 500/512, 500/2048 and 1,500/2048 character/token settings. The latter is a reviewer-requested inference sensitivity, not a new main training cutoff. The 500-character token-cap control produces identical predictions with no extra clipping. At 1,500 versus 500 characters (both 2,048-token ceilings), adapted Qwen/Llama macro-F1 improves by 0.0072 [0.0048, 0.0097] and 0.0055 [0.0029, 0.0081]. In the predefined >500-character cohort (5,770 rows/5,131 groups), gains are 0.0156 [0.0105, 0.0209] and 0.0117 [0.0065, 0.0172]. Zero-shot Qwen instead decreases by 0.0072; zero-shot Llama's interval includes zero. Thus more text does not uniformly help. Controlled batch-one latency uses 100 preselected internal IDs, 20/class, ten validation warmups and three timed repetitions in each of 24 conditions (7,200 measured calls). At the main input setting, paired adapted-minus-base end-to-end mean increases are 0.1224 s [0.1166, 0.1284] for Qwen and 0.0802 s [0.0758, 0.0847] for Llama. We report generation/end-to-end means, medians, p95 and token counts. These timings use the local RTX 5060 Ti, whereas the submitted timings used TRUBA/V100 and different cohort/timing conditions; their numerical difference is not presented as a same-hardware speedup. We added four-thread CPU baseline timings on the same 100 IDs and repetition orders, plus recorded neural training runtimes and fresh selected-configuration sparse fit times. The cost tables identify device, precision, training/search scope and mean/median/p95 rather than pooling unlike measurements. Output lengths vary; separate job blocks are not a fully randomised crossover or thermal-equivalence test. Batch-one serial timing does not establish concurrent-service capacity, acceptability or regulatory readiness, and those claims are removed.

**Changes and locations.** Manuscript Section 3.6 (p. 6); Manuscript Table 5 (p. 9); Manuscript Table 6 (p. 10); Manuscript Table 7 (p. 10); Manuscript Section 6 (p. 11); Supplementary material Section S5 (p. 12); Supplementary material Table S23 (p. 14); Supplementary material Table S24 (p. 14); Supplementary material Section S6 (p. 14); Supplementary material Table S26 (p. 15); Supplementary material Table S35 (p. 20); Supplementary material Table S36 (p. 20).

## Reviewer #2

### R2.1 — Positioning against Ref. [14]

> The authors should strengthen the positioning against closely related recent studies, especially Reference [14], and clarify the distinction between the proposed approach and prior LLM-based complaint classification methods.

**Response.** We agree and revised the Introduction/Related work and a dedicated protocol-comparison table. Güllü et al. already fine-tune LLMs for the same financial label taxonomy. Our contribution is the explicit, budget-conditional paired adaptation protocol and its evidence, not the first application or a cross-paper score improvement. The report distinguishes the supervised comparison in [14] from zero-shot work in [15]. We also include the separate source-parameter encoder sensitivity without relabelling it as exact replication. See response R1.7 and R1.5 for the measured results.

**Changes and locations.** Manuscript Table 1 (p. 3); Manuscript Section 5 (p. 10); Supplementary material Section S7 (p. 16).

### R2.2 — Same-source benchmark terminology

> The term “external benchmark” should be reconsidered because the benchmark originates from the same CFPB data source.

**Response.** We replaced the external-benchmark claim with source-related legacy evaluation views. The twelve preserved files originate from the related CFPB resource, overlap extensively and are not twelve independent replications. We now quantify their overlap, equally weight unique visible units and use conservative group-aware inference. No independent-source generalisation follows.

**Changes and locations.** Manuscript Section 3.1 (p. 3); Manuscript Section 3.5 (p. 6); Supplementary material Section S4 (p. 11).

### R2.3 — Configuration, truncation and training budget

> The manuscript would also benefit from a more complete description of the QLoRA and inference configurations and clearer justification of the 500-character truncation and 10,000-example training budget.

**Response.** The new implementation is specified with immutable commits, complete QLoRA settings, exact prompt, thinking disabled, greedy generation and strict output parsing. See R1.4 for the full settings. The inherited 500-character narrative cap and separate 512-token ceiling keep the main paired input convention fixed; we make no optimality or hardware-necessity claim. The longer-input experiment reports benefits and failures rather than assuming the cap harmless. The common 10k/2k selection is a fixed labelled-data comparison budget inherited from the original study, not a demonstrated sample-efficiency optimum. All revised families share it; full-data TF-IDF is separate unequal-budget context. The 100-token RoBERTa setting appears only in the Ref. [14] Table IV supplementary sensitivity, while main RoBERTa uses 512 tokens. To make the configuration provenance explicit, we restored the six earlier Qwen validation conditions and their natural/balanced sampling comparison in Supplementary Section S8. These pre-quarantine scores explain the transferred settings; they are kept separate from the corrected-data validation searches and final comparisons. QLoRA was not retuned on the corrected allocation, whereas the sparse and encoder baselines received 48- and six-candidate validation searches. This provides explicitly tuned baseline comparators without an additional model-specific QLoRA search; it does not imply that the remaining configuration search space is exhausted.

**Changes and locations.** Manuscript Section 3.3 (p. 4); Manuscript Section 3.4 (p. 5); Manuscript Table 5 (p. 9); Supplementary material Section S2 (p. 3); Supplementary material Section S7 (p. 16); Supplementary material Table S34 (p. 19).

### R2.4 — Statistical methodology

> The statistical evaluation and bootstrap methodology should be described more clearly.

**Response.** We now define the observed-point estimator, legacy weights, conservative dependency components, 10,000 shared paired group-bootstrap draws and conditional scope. Seed sample SD is separate from the bootstrap interval; refit metrics are averaged, not prediction ensembles. Selection/training uncertainty and multiplicity are not covered. The complete seed/metric/contrast tables and historical point/interval reconciliation are supplied; see R1.6.

**Changes and locations.** Manuscript Section 3.5 (p. 6); Supplementary material Section S3 (p. 6); Supplementary material Table S11 (p. 8); Supplementary material Table S33 (p. 18).

### R2.5 — Constrained decoding, robustness and deployment

> Claims concerning constrained decoding, robustness, and deployment should also be moderated where appropriate.

**Response.** The text now accurately describes JSON prompting followed by regex extraction, not token-level constrained decoding. Invalid labels count as errors; observed compliance is not a guarantee. We removed untested reuse/general robustness and deployment-readiness claims. Completed same-checkpoint input sensitivity and controlled fresh-call latency quantify specific effects, with negative/uncertain results retained. They do not establish independent-source generalisation, production capacity, user acceptability or regulatory readiness; see R1.8.

**Changes and locations.** Manuscript Section 3.3 (p. 4); Manuscript Section 4.3 (p. 8); Manuscript Section 4.4 (p. 8); Manuscript Section 6 (p. 11); Supplementary material Table S15 (p. 10); Supplementary material Section S6 (p. 14).

### R2.6 — Practical PEFT contribution and TF-IDF competitiveness

> Most importantly, the results show that QLoRA achieves performance very close to the TF-IDF + Logistic Regression baseline; therefore, the practical contribution should be framed around the benefits and trade-offs of LLM-based PEFT rather than implying clear classification superiority.

**Response.** We reframed the abstract, discussion and conclusion around the measured benefits and costs of same-model adaptation. At the common 10k budget, Qwen/Llama QLoRA macro-F1 is 0.8331/0.8402 versus sparse 0.8068, with paired deltas reported rather than universal superiority. Full-data sparse classification reaches 0.8476 with 120,502 training examples; its larger budget is explicit. Qwen is below that full-data comparator by 0.0145 with an interval excluding zero; Llama's -0.0074 interval includes zero (unrounded upper bound +0.0000068). Tuned encoders remain competitive and their contrasts with 10k sparse include zero. Llama uses about one twelfth as many labelled training examples as full-data TF-IDF (10,000 versus 120,502), making labelled-data efficiency a concrete benefit. The interval containing zero does not establish equivalence or non-inferiority. We have made this result prominent in the Discussion and Conclusion and added direct QLoRA-versus-encoder intervals. The cost tables show the other side of the choice: mean Qwen/Llama refits take 6.9612/7.1541 GPU wall hours, compared with 0.0254--0.0453 for the encoders; the separately measured sparse fits take 0.0004/0.0134 CPU wall hours. These device-specific times use explicitly stated timer boundaries. We therefore recommend considering measured accuracy, label requirements and compute together; sparse classifiers remain strong efficient alternatives.

**Changes and locations.** Manuscript Table 3 (p. 7); Manuscript Table 4 (p. 7); Manuscript Table 6 (p. 10); Manuscript Table 7 (p. 10); Manuscript Section 5 (p. 10); Manuscript Section 7 (p. 11); Supplementary material Table S35 (p. 20); Supplementary material Table S36 (p. 20).

### R2.7 — Code/configuration access

> Finally, improving public access to the code and experimental configurations would substantially improve reproducibility.

**Response.** The code, experimental configurations and reproducibility evidence are provided in the public repository at https://github.com/muhammettan28/CFPB_tuning. The root experiments/ directory contains encoder searches/refits, QLoRA execution records, final predictions, reference-parameter runs and both TF-IDF training budgets. The analysis/, configs/, dataset/ and scripts/revision/ directories contain statistical results, frozen settings, data selections, manifests and execution/verification code. The README and Supplementary Section S8 document an executable CPU replay command and its dependencies. Replay verifies 404 source hashes and compares all numeric cells and row identities of nine recomputed result tables with the recorded results, without neural weights, a GPU or a local model registry. Repeating neural fitting requires the exact recorded model snapshots and a separate execution checkout. Neural weights and third-party literature are excluded; unavailable upstream preprocessing and auxiliary-model identities remain explicitly documented limitations.

**Changes and locations.** Supplementary material Section S8 (p. 17); Supplementary material Table S30 (p. 17); Manuscript Section 6 (p. 11).

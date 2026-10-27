# Main experiment protocol

## Data and fitting budgets

All matched-budget systems use the same 10,000 training and 2,000 validation
records in `dataset/revision_v1/`, selected with seed 42. Model-visible overlap
screening excludes fitting conflicts while retaining the 15,428 internal-test
records. Evaluation also includes twelve source-related benchmark views, whose
overlap is handled using unique-input weights and dependency groups.

Optimisation seeds 42/43/44 vary training randomness without changing membership.
Full-data TF-IDF uses all 120,502 retained training rows and is reported as an
unequal-budget comparison. Machine-readable settings are in
`configs/gpu_revision_v1.json`; data identities are in
`analysis/revision_v1/protocol.json`.

## Primary input convention

The main inputs retain at most 500 narrative characters with a 512-token ceiling.
Base and adapted LLMs share their prompt, quantisation, tokenizer and greedy
inference settings. Qwen thinking is disabled. Chat templates are rendered once
and tokenised with `add_special_tokens=False`. Output uses JSON prompting and
regex label extraction; absent or invalid labels count as errors, without retries.
This is not token-level constrained decoding.

## Model fitting and validation selection

QLoRA uses NF4 4-bit double quantisation, FP16 compute, rank 16, alpha 32,
dropout 0.05, all-linear targets and completion-only JSON/EOS loss. The fixed
settings are two epochs, learning rate 0.0002, physical batch one and accumulation
16. Validation loss selects the saved checkpoint. Each LLM has three refits.

Each encoder has six validation candidates: learning rates 0.00002/0.00003/0.00005
and two/three epochs. Validation macro-F1 selects the final-epoch configuration,
with a predefined candidate-ID tie rule. Three selected-configuration refits
use physical batch four and accumulation four. Reloaded predictions must match
all 2,000 validation predictions before test inference.

Each sparse budget has a separate 48-candidate validation search. Feature maps
fit training text only. Repeated deterministic sparse fits are parity checks,
not stochastic seed replicates. Complete settings, candidate scores and selected
configurations are retained with each experiment.

## Same-checkpoint input-length analysis

The conditions are 500 characters/512 tokens, 500/2,048 and 1,500/2,048 on the
same internal IDs. Models are trained with 500 characters throughout; the length
comparison changes inference inputs only. The first two conditions have identical
token inputs and predictions. The full-narrative greater-than-500-character
cohort contains 5,770 rows in 5,131 groups.

## Controlled latency

Fresh batch-one LLM calls use 100 fixed internal IDs, 20 per class, ten validation
warmups and three hash-ordered repetitions in each condition. CUDA is synchronised
around generation. End-to-end time includes tokenisation, transfer, generation,
decoding and parsing; model loading, warmup and disk I/O are excluded. Memoisation
is bypassed for all timed calls. Other GPU compute aborts a timing job.

Mean, median, p95, token counts and paired differences are reported. Variants run
in separate job blocks, so measured differences include output-length effects
and do not establish concurrent-service capacity or isolated adapter overhead.
CPU baseline timings use the same IDs and repetition orders, with four threads;
device, precision and timing boundaries are reported separately.

## Statistical analysis and verification

All 19 main prediction variants share 10,000 paired whole-group bootstrap draws
with RNG seed 20261020. Observed points precede resampling; three-refit systems
average separately scored models. Across-refit sample SD is separate from the
conditional 95% interval. Intervals exclude training/model-selection uncertainty
and have no family-wise multiplicity adjustment.

Results and source manifests are in `analysis/gpu_execution/`. The root README
and `docs/REPRODUCIBILITY.md` provide CPU verification commands. Execution-time
plan bytes are preserved under `experiments/provenance/execution_plans/`;
recorded settings and outputs identify the conditions actually run.

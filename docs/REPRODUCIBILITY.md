# Reproducing the reported results

The repository provides data selections, configurations, validation searches,
saved predictions, timing records and analysis code for five-category CFPB
complaint classification. Run the commands below from the repository root.

## Evidence locations

| Evidence | Location |
| --- | --- |
| Data, partitions and fixed selections | `dataset/revision_v1/` |
| Main neural searches, refits and predictions | `experiments/gpu_revision_v1/` |
| TF-IDF/Logistic Regression searches and predictions | `experiments/tfidf_revision_v1/` |
| Reference [14] encoder-parameter sensitivity | `experiments/ref14_encoder_v1/` |
| CPU baseline timing calls and sparse fit timings | `experiments/cpu_costs_v1/` |
| Main metrics, paired intervals and source manifests | `analysis/gpu_execution/` |
| Reference-setting metrics and paired intervals | `analysis/ref14_encoder/` |
| QLoRA-versus-encoder contrasts and cost summaries | `analysis/postprocessing/` |
| Recorded settings and model identities | `configs/` and per-run metadata |
| Execution and verification tools | `scripts/revision/` |

The main neural comparison uses the same 10,000 training and 2,000 validation
records across model families, with optimisation seeds 42/43/44. Full-data TF-IDF
uses 120,502 training records. Reference [14] settings form a separate sensitivity
analysis; they do not replace the main encoder baselines.

## Verify the results on CPU

Use a fresh virtual environment:

```bash
python -m pip install -r scripts/revision/requirements-cpu-revision.txt
CUDA_VISIBLE_DEVICES='' python scripts/revision/replay_reviewer_results.py --output /tmp/cfpb-review-replay
python scripts/revision/verify_comparison_evidence.py --output /tmp/cfpb-review-comparisons.json
```

The replay output directory must not already exist. No GPU, neural weights,
model registry or manuscript files are required. Replay verifies 404 source
hashes and recomputes nine complete result tables from the supplied predictions
and timings, including 10,000-draw paired group-bootstrap intervals. It compares
row identities, row order and numeric cells with the recorded tables at absolute
tolerance 1e-12, and writes results to the specified output directory. It does not
fit or select models, or rerun inference.

The second command checks 1,500 raw CPU timing calls, 17 fitting-time records,
cost summaries and the zero-shot protocol comparisons. Paired intervals condition
on the fitted models and fixed selections; they do not include training or
model-selection uncertainty. Evaluation is retrospective and source-related.

## Recorded paths and source integrity

Manifests retain the paths and SHA-256 values recorded during execution.
`scripts/revision/project_paths.py` resolves recorded paths to the current
repository layout, including `REVISION_PAPER/experiments/` to `experiments/`
and `REVISION_PAPER/data/revision_v1/` to `dataset/revision_v1/`. The complete
mapping is in `analysis/repository_layout/PATH_MAPPING.json`. These recorded
prefixes do not require a local `REVISION_PAPER/` directory for result replay.

Execution-time source snapshots are preserved in `experiments/provenance/` so
that recorded code hashes remain verifiable. Use the maintained entry points in
`scripts/revision/` to run commands. Written protocols in `protocols/` and per-run
metadata document the experimental conditions; their recorded path prefixes are
resolved as above.
Execution-time plan bytes are preserved in
`experiments/provenance/execution_plans/`; the method descriptions in `protocols/`
are separate from those source-identity records.

## Repeat model execution

Training and neural inference require the exact recorded base-model snapshots
and execution environment. Neural weights, optimizer states, downloaded models
and runtime caches are excluded from the repository. Model identities and
software/hardware versions are recorded in the configurations and per-run files.
Data and model licences remain those of their sources.

Use a separate execution checkout with empty experiment output directories and
fill in `configs/model_registry.template.json` as
`configs/model_registry.local.json`. Completed outputs and audited input code
are protected against overwrite or unverified changes. In that separate
checkout, preserve the supplied tokenizer-audit evidence before generating the
audit for the execution code:

```bash
python scripts/revision/run_revision_gpu.py audit --model MODEL --registry configs/model_registry.local.json
```

Replace `MODEL` with the configured model key. Reference-setting execution also
requires its corresponding token-cap audit. Keep the evidence checkout intact;
the supplied audit files identify the inputs used in the reported experiments.

## Interpretation of implementation costs

CPU timings use a fixed, balanced 100-record cohort, four threads, ten validation
warmups and three repetitions. Encoder timing uses seed-42 FP32 checkpoints;
the primary encoder accuracy results use GPU FP16 predictions. Sparse fitting
times cover a fresh vectorizer and selected classifier fit, with all 15,428
internal predictions checked against the retained sparse models.

Neural training times come from recorded Trainer runtimes. Encoder search totals
sum training phases, whereas sparse search totals cover the complete recorded
search. Device, precision and timer boundaries are reported separately. These
measurements describe implementation costs; they do not establish a same-device
architecture ranking, energy consumption or concurrent-service capacity.

## Provenance limitations

Source hashes identify the supplied data, not the unavailable upstream
CFPB-to-Kaggle preprocessing, product aggregation or sampling implementation.
Some auxiliary fits lack immutable upstream model identities and complete fitted
weights, limiting their reconstruction. The primary model identities are recorded
separately. Exact-input checks and conservative text groups do not establish
semantic independence or prospective generalisation.

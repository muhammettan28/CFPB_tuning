# Consumer complaint classification with QLoRA

Completed experiments, saved predictions and search records
for five-category CFPB complaint classification are organised below. Main and
reference-parameter studies are reported separately. Auxiliary analyses specify
their fitting populations and are excluded from the primary model ranking.

| Directory | Contents |
| --- | --- |
| `experiments/gpu_revision_v1/` | Main encoder searches/refits, QLoRA runs, final predictions, length sensitivity and controlled latency |
| `experiments/cpu_costs_v1/` | Raw CPU baseline latency calls, selected sparse fit timings, environment and source hashes |
| `experiments/ref14_encoder_v1/` | Separate reference [14] encoder-parameter sensitivity |
| `experiments/tfidf_revision_v1/` | Both sparse training budgets, all 48 candidates per budget, fitted sparse objects and predictions |
| `analysis/` | Result tables, statistical comparisons, data/tokenizer audits and frozen selection manifests |
| `dataset/` | Original data/splits and corrected `revision_v1/` selections/evaluation data |
| `configs/` | Frozen plans, signatures and model-registry template |
| `scripts/revision/` | Revision experiment execution, analysis, verification and document-generation tools |
| `protocols/` | Written experiment protocols |
| `experiment_archive/` | Saved predictions and execution evidence for auxiliary analyses |
| `scripts/`, `CLASSIC_TEST/`, `slurm_files/` | Original preparation, training, baseline and TRUBA utilities |

Start with [`experiments/`](experiments/) for experiment records and
[`analysis/`](analysis/) for numerical results. [Reproducibility instructions](docs/REPRODUCIBILITY.md)
explain the commands, evidence locations and recorded source paths.

## Check the reported results on CPU

From the repository root, using a fresh virtual environment:

```bash
python -m pip install -r scripts/revision/requirements-cpu-revision.txt
CUDA_VISIBLE_DEVICES='' python scripts/revision/replay_reviewer_results.py --output /tmp/cfpb-review-replay
python scripts/revision/verify_comparison_evidence.py --output /tmp/cfpb-review-comparisons.json
```

The replay output directory must not already exist. No neural weights, local
model registry, GPU, original-paper directory or source ODT is needed. Replay
verifies 404 source hashes and recomputes nine complete result tables, including
10,000-draw paired intervals. The comparison check independently reconstructs
the implementation costs and zero-shot protocol comparisons.

Neural weights, optimizer states, downloaded models and runtime caches stay local;
predictions, numeric results, settings, validation search records and recorded
code/environment evidence are included. Replaying saved results does not rerun
training. Manuscript preparation files are not needed for these checks.

Code, data selections, configurations, saved predictions and results are shared at
https://github.com/muhammettan28/CFPB_tuning. Start with the root
[`experiments/`](experiments/) directory for experiment records and the commands above
for CPU verification.

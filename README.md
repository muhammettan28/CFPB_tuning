# Consumer complaint classification: QLoRA revision

Completed experiments, saved predictions, search records and revised documents
for five-category CFPB complaint classification are organised below. Main and
reference-parameter studies are reported separately; older results are historical
evidence and are not part of the corrected-data ranking.

| Directory | Contents |
| --- | --- |
| `REVISION_PAPER/` | Revised manuscript, supplement, reviewer responses, LaTeX sources and their table/figure assets |
| `experiments/gpu_revision_v1/` | Main encoder searches/refits, QLoRA runs, final predictions, length sensitivity and controlled latency |
| `experiments/cpu_costs_v1/` | Raw CPU baseline latency calls, selected sparse fit timings, environment and source hashes |
| `experiments/ref14_encoder_v1/` | Separate reference [14] encoder-parameter sensitivity |
| `experiments/tfidf_revision_v1/` | Both sparse training budgets, all 48 candidates per budget, fitted sparse objects and predictions |
| `analysis/` | Result tables, statistical comparisons, data/tokenizer audits and frozen selection manifests |
| `dataset/` | Original data/splits and corrected `revision_v1/` selections/evaluation data |
| `configs/` | Frozen plans, signatures and model-registry template |
| `scripts/revision/` | Revision experiment execution, analysis, verification and document-generation tools |
| `protocols/` | Original written experiment protocols, preserved as historical records |
| `experiment_archive/` | Saved predictions and execution evidence used for historical-only analyses |
| `scripts/`, `CLASSIC_TEST/`, `slurm_files/` | Original preparation, training, baseline and TRUBA utilities |

Start with `REVISION_PAPER/manuscript.pdf`, `supplementary_material.pdf`, and
`response_to_reviewers.pdf`. The reviewer-comment index and editable response are
in that same directory. [Reproducibility instructions](docs/REPRODUCIBILITY.md)
explain the current paths and the preserved paths recorded in older metadata.

## Check the reported results on CPU

From the repository root, using a fresh virtual environment:

```bash
python -m pip install -r scripts/revision/requirements-cpu-revision.txt
CUDA_VISIBLE_DEVICES='' python scripts/revision/replay_reviewer_results.py --output /tmp/cfpb-review-replay
python scripts/revision/audit_document_rows.py --output /tmp/cfpb-review-table-cells.json
python scripts/revision/verify_restored_document_content.py --output /tmp/cfpb-review-content.json
python scripts/revision/verify_comparison_evidence.py --output /tmp/cfpb-review-comparisons.json
```

The replay output directory must not already exist. No neural weights, local
model registry, GPU, original-paper directory or source ODT is needed. Replay
verifies 404 source hashes and recomputes nine complete result tables, including
10,000-draw paired intervals. The table audit checks 39 table fragments, 583 rows
and 3,696 cells against their corresponding evidence.

Neural weights, optimizer states, downloaded models and runtime caches stay local;
predictions, numeric results, settings, validation search records and recorded
code/environment evidence are included. Replaying saved results does not rerun
training. Retired work/status files and author-only revision logs are excluded.
The original paper is preserved separately and unchanged in local `ORJ_PAPER/`.

Code, data selections, configurations, saved predictions and results are shared at
https://github.com/muhammettan28/CFPB_tuning. Start with the root
[`experiments/`](experiments/) directory for experiment records and the commands above
for CPU verification.

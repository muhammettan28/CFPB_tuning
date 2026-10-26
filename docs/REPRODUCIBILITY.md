# Reproducing the revision results

Run commands from the project root. `experiments/`, `analysis/`, `configs/`,
`protocols/`, and `scripts/revision/` are siblings of `REVISION_PAPER/`.
Corrected data are in `dataset/revision_v1/`. The paper directory contains only
publication sources/PDFs, reviewer responses and required LaTeX assets.

The main study completed 88 stages, including five small API/hardware smoke
checks. Encoder search stages each contain six candidates. The separate reference
sensitivity completed 12 stages (six actual refits and six evaluation stages).
All scientific predictions, observed metrics, search candidates, selected settings,
seed records and latency samples are retained. Smoke fits are excluded from
scientific results. Machine-readable results are in `analysis/gpu_execution/`
and `analysis/ref14_encoder/`; sparse search records are under `experiments/`.

## Standalone CPU replay

Follow the two commands in the root README. Replay verifies source hashes,
recomputes complete tables from the saved predictions/timings, and compares
numeric cells at absolute tolerance 1e-12, preserving row identities and order.
It writes only a new output directory and does not fit/select models. The table
audit independently checks model/condition/metric-specific cells. The original
source data and earlier split files also remain available under `dataset/`.

## Historical paths and source identities

Files were reorganised on 26 October 2026. Frozen protocol/configuration JSON,
run metadata, selection files, prediction files and their recorded SHA-256 values
were not rewritten. They may therefore contain `REVISION_PAPER/experiments/`,
`REVISION_PAPER/analysis/`, `REVISION_PAPER/configs/`,
`REVISION_PAPER/data/revision_v1/`, and `REVISION_PAPER/scripts/`.
`scripts/revision/project_paths.py` translates these to `experiments/`,
`analysis/`, `configs/`, `dataset/revision_v1/`, and `scripts/revision/` when reading
historical evidence. It also resolves the original protocol hash-map keys.

The written protocols are preserved verbatim; their old example commands are
historical. Current equivalents use `python scripts/revision/run_revision_gpu.py`
and `--registry configs/model_registry.local.json`. The complete mapping is
recorded in `analysis/repository_layout/PATH_MAPPING.json`. Original pre-move
tool sources and their hashes are preserved in `experiments/provenance/` and
must be read as historical evidence, rather than used as current entry points.
Compiled document label evidence for standalone response-location generation is
in `analysis/document_labels/`; new LaTeX build files stay in `.build/` locally.

## Model execution

Saved-result replay needs no model downloads. Repeating training/inference
requires the exact recorded base-model commits, the recorded GPU environment,
and a separate execution checkout with empty experiment output directories.
Local model-registry paths are deliberately omitted; fill in the template.
The execution scripts reject completed outputs and modified audited input code.
Because the maintained tools changed paths during relocation, before fitting in
that separate checkout run a fresh CPU tokenizer audit for each model using
`python scripts/revision/run_revision_gpu.py audit --model MODEL --registry
configs/model_registry.local.json`. Preserve the original audit files first:
the archived runs refer to their original hashes. This is a new audit of the
current execution code, not a replacement of historical run evidence. Reference
sensitivity also requires its separately recorded shorter-cap audit. Do not run
a new audit in this evidence checkout, where it would replace existing reports.

Neural weights, optimizer/RNG states, memoisation databases, downloaded models,
temporary status/plan files, draft packages and third-party literature are not
part of the Git/reviewer evidence set. They remain local when already present.
Unavailable upstream preprocessing and historical model identities remain the
limitations stated in the manuscript and response. No new model execution was
performed to reorganise these files. Repository access must be arranged for the
actual reviewers under the journal's requirements.

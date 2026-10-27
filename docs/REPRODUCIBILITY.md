# Reproducing the revision results

Run commands from the project root. `experiments/`, `analysis/`, `configs/`,
`protocols/`, and `scripts/revision/` contain the public reproducibility evidence.
Corrected data are in `dataset/revision_v1/`. Publication sources/PDFs, reviewer
responses and their LaTeX assets are kept locally in the Git-ignored
`REVISION_PAPER/` directory; they are not required for the public CPU checks.

The main study completed 88 stages, including five small API/hardware smoke
checks. Encoder search stages each contain six candidates. The separate reference
sensitivity completed 12 stages (six actual refits and six evaluation stages).
All scientific predictions, observed metrics, search candidates, selected settings,
seed records and latency samples are retained. Smoke fits are excluded from
scientific results. Machine-readable results are in `analysis/gpu_execution/`
and `analysis/ref14_encoder/`; sparse search records are under `experiments/`.

## Standalone CPU replay

Follow the commands in the root README. Replay verifies source hashes,
recomputes complete tables from the saved predictions/timings, and compares
numeric cells at absolute tolerance 1e-12, preserving row identities and order.
It writes only a new output directory and does not fit/select models. The original
source data and earlier split files also remain available under `dataset/`.

The author-only `audit_document_rows.py` and `verify_restored_document_content.py`
checks require the local publication sources. The first checks
model/condition/metric-specific table cells; the second additionally reconstructs
the four LLM confusion matrices, verifies historical Qwen selection conditions
and checks historical dataset descriptions on their original population. These
document checks are separate from the public saved-result replay.

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
performed to reorganise these files. The reproducibility repository is
https://github.com/muhammettan28/CFPB_tuning; experiment records are under its root
`experiments/` directory.

## Additional comparisons and implementation costs

The ninth replayed table is `analysis/postprocessing/qlora_encoder_contrasts.csv`.
It compares the means of all three QLoRA refits with all three encoder refits
within each original joint 19-variant group-bootstrap draw. The eight existing
main/reference result CSVs and their input predictions remain unchanged.

`experiments/cpu_costs_v1/` records 1,500 CPU baseline timing calls on the fixed
100-record latency cohort, with four threads, ten validation warmups and three
repetitions. Encoder timings use FP32 seed-42 checkpoints; they do not replace
the primary GPU FP16 accuracy results. Sparse fit timings include a fresh
vectorizer and selected classifier fit; all internal predictions match the
retained sparse models. Neural training times are exported from Trainer runtime
records; encoder search totals sum training phases, whereas sparse search totals
cover the complete recorded search. Device/precision/timer scopes are stated in
the supplementary cost tables. These are implementation costs, not a same-device
architecture benchmark or energy measurement.

`verify_comparison_evidence.py` reconstructs the cost summaries from the supplied
raw calls and training excerpts and checks the zero-shot protocol comparisons.
It requires no neural weights, GPU or publication files. Authors can add
`--include-documents` to check the local manuscript and reviewer-response claims;
that option requires `REVISION_PAPER/`. `build_comparison_evidence.py` can rebuild
these derived summaries using a freshly replayed bootstrap directory; it consumes
the supplied training excerpts when the author's verbose runtime logs are absent.

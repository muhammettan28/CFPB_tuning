# CFPB complaint classification

This repository contains the original data preparation, QLoRA training and
evaluation utilities for five-category consumer financial complaint classification.

| Directory | Contents |
| --- | --- |
| `scripts/` | Data preparation, model training and evaluation code |
| `CLASSIC_TEST/` | Classical and encoder baseline utilities |
| `dataset/` | Source data, evaluation views and original split files |
| `slurm_files/` | Original TRUBA execution recipes |

The revised manuscript, supplementary material, reviewer responses and completed
revision experiments are maintained locally in `REVISION_PAPER/`. This directory
is intentionally excluded from Git; these files are distributed in a separate
revision/reproducibility package, rather than being claimed to be available through
this repository. Its README identifies the current documents and CPU result replay.

Generated caches, logs, build auxiliaries and retired planning/status files are not
versioned. The original manuscript is preserved separately in `ORJ_PAPER/`.

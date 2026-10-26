# Corrected-data GPU study — results and evidence

Completed local experiment analysis: 2026-10-24.

All systems below use the frozen corrected data. Base LLMs have no adaptation budget; full-data TF-IDF uses a larger budget and is contextual. Checkpoint provenance and actual source/config versions are retained per run.

## Internal test macro-F1

| Model | Train rows | Macro-F1 | 95% paired-group CI | Run SD |
| --- | ---: | ---: | --- | ---: |
| Qwen3-8B zero-shot | 0 | 0.6295 | [0.6182, 0.6405] | — |
| Qwen3-8B QLoRA | 10,000 | 0.8331 | [0.8245, 0.8411] | 0.0048 |
| Llama-3.1-8B-Instruct zero-shot | 0 | 0.6247 | [0.6138, 0.6349] | — |
| Llama-3.1-8B-Instruct QLoRA | 10,000 | 0.8402 | [0.8312, 0.8483] | 0.0020 |
| BERT-base | 10,000 | 0.8119 | [0.8028, 0.8203] | 0.0010 |
| DistilBERT | 10,000 | 0.8104 | [0.8014, 0.8188] | 0.0065 |
| RoBERTa-base | 10,000 | 0.8105 | [0.8015, 0.8187] | 0.0039 |
| TF-IDF/LR 10k | 10,000 | 0.8068 | [0.7973, 0.8157] | 0.0000 |
| TF-IDF/LR full | 120,502 | 0.8476 | [0.8402, 0.8549] | 0.0000 |

## What is recorded

- `corrected_model_metrics.csv`: internal and unique-input-weighted legacy point estimates, intervals and run SD.
- `corrected_paired_contrasts.csv`: within-model adaptation and matched-10k TF-IDF comparisons, with paired intervals. Full-data contrasts are explicitly unequal-budget context.
- `corrected_long_input.csv`: matched inference-length effects and token-cap control; no longer-context retraining claim.
- `controlled_latency_summary.csv` / `controlled_latency_paired.csv`: synchronized real calls, all 100 IDs and three repetitions; cache bypassed.
- `GPU_RESULT_SOURCES.json`: SHA-256 evidence for every prediction and latency source.

## Interpretation limits

- Retrospective previously inspected test; not new prospective evaluation.
- Fixed-trained-model bootstrap excludes training/model-selection uncertainty.
- Latency variants are separate job blocks, not a fully randomized crossover.
- No deployment acceptability, regulatory readiness or independent-source generalization claim.
- Archived and new runs differ in data allocation, pinned artifact evidence, explicit serialization and hardware; their score difference is not a causal estimate of leakage removal alone.

Archived results remain historical and are not substituted for the corrected-data scores. Manuscript integration/rendering and final author review remain separate from completing these experiments. No reviewer/public upload was performed.

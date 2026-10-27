# Reference [14] encoder-parameter sensitivity

This analysis uses the BERT/RoBERTa parameters reported in Tables III-IV of
Gullu et al., *Customer Complaint Classification with Large Language Models*,
ACDSA 2025, DOI 10.1109/ACDSA65407.2025.11166338.

| Model | Learning rate | Epochs | Physical batch | Token cap |
| --- | ---: | ---: | ---: | ---: |
| BERT-base-uncased | 0.00002 | 3 | 16 | 128 |
| RoBERTa-base | 0.00005 | 3 | 16 | 100 |

Both learning-rate/epoch pairs match the primary validation-search winners.
The sensitivity changes physical batch from four with accumulation four to
16 with accumulation one, and token caps from 512 to 128/100. Their joint effect
is measured; it is not attributed to one parameter. The main RoBERTa condition
retains its 512-token ceiling.

All six refits use the fixed 10,000 training and 2,000 validation records,
500-character inputs, pinned checkpoints and optimisation seeds 42/43/44.
Token-cap audits cover the selected/evaluation inputs. Each refit must pass
prediction-reload parity on all 2,000 validation records before final inference.

This is a parameter sensitivity rather than an exact replication: data allocation,
preprocessing, PyTorch/FP16 and inherited execution settings differ from the
reference. No replacement baseline is selected using test performance.

All twelve prediction variants in this analysis share 10,000 paired whole-group
bootstrap draws. Benchmark views use unique-input weighting; seed SD is reported
separately from conditional intervals. The shared-group compression differs from
the 19-variant primary analysis, so equal RNG seeds do not imply identical
Monte Carlo draws between the two analyses.

Machine-readable settings are in `configs/ref14_encoder_v1.json`. Refits and
predictions are in `experiments/ref14_encoder_v1/`; metrics, paired contrasts and
source manifests are in `analysis/ref14_encoder/`. The execution-time plan bytes
are preserved under `experiments/provenance/execution_plans/`.

# Reference [14] encoder-parameter sensitivity

BERT/RoBERTa parameters in Tables III–IV were specified before supplementary final predictions. All six three-epoch refits use the same frozen 10,000 training and 2,000 validation examples. The published learning rates and epochs match the main validation winners; the reference conditions use physical batch to 16 (accumulation 1) and token limits to 128/100. Their joint effect is measured, not attributed to an individual hyperparameter.

This is a parameter-sensitivity extension, not an exact reproduction of the paper. We retain 500-character inputs, corrected split, pinned checkpoints, PyTorch/FP16 and the main warmup/weight-decay/scheduler settings. Source Table IV reports TensorFlow for RoBERTa. The source uses another split and preprocessing. All conditions are reported without selecting a replacement baseline on test scores.

| Evaluation | Model | Setting | Macro-F1 | 95% group-bootstrap CI | Run SD |
| --- | --- | --- | ---: | --- | ---: |
| internal | bert | main | 0.8119 | [0.8027, 0.8202] | 0.0010 |
| internal | bert | ref14 | 0.8102 | [0.8010, 0.8185] | 0.0026 |
| internal | roberta | main | 0.8105 | [0.8015, 0.8189] | 0.0039 |
| internal | roberta | ref14 | 0.8095 | [0.8003, 0.8180] | 0.0006 |
| legacy | bert | main | 0.7936 | [0.7807, 0.8059] | 0.0042 |
| legacy | bert | ref14 | 0.7943 | [0.7813, 0.8063] | 0.0070 |
| legacy | roberta | main | 0.7926 | [0.7800, 0.8046] | 0.0042 |
| legacy | roberta | ref14 | 0.7900 | [0.7771, 0.8021] | 0.0020 |

Intervals share 10,000 frozen-group bootstrap draws across main-setting and reference-setting predictions. Benchmark views receive the same unique-normalized-input weights as in the core analysis. Run SD is reported separately; the intervals condition on fitted checkpoints and exclude training/model-selection uncertainty. Evaluation is retrospective, not a new unseen prospective test. Read paired_contrasts.csv for reference-setting minus main-setting intervals. No numerical superiority over the published paper is inferred.

"""Read the saved evidence used to restore protocol provenance and error detail."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from project_paths import ROOT
from revision_gpu_common import DATA, EXP, LABELS, load_frame, sha_file

HISTORICAL = ROOT/'experiment_archive/outputs'
# Explicit run identities keep similarly rounded scores from being interchanged.
SELECTION_RUNS = [
    ('Pilot', 'qwen3_8b_search_summary.csv', 'qwen3_8b_qlora_pilot_5952287'),
    ('Balanced E1', 'qwen3_8b_search_summary.csv', 'qwen3_8b_search_E1_bal_lr2e4_r16_5960178_0'),
    ('Balanced E2', 'qwen3_8b_search_summary.csv', 'qwen3_8b_search_E2_bal_lr1e4_r16_5960178_1'),
    ('Balanced E3', 'qwen3_8b_search_summary.csv', 'qwen3_8b_search_E3_bal_lr2e4_r8_5960178_2'),
    ('Natural 2-epoch', 'qwen3_8b_second_stage_summary.csv', 'qwen3_8b_second_stage_S1_nat_lr2e4_r16_2ep_5981133_0'),
    ('Balanced 2-epoch', 'qwen3_8b_second_stage_summary.csv', 'qwen3_8b_second_stage_S2_bal_lr2e4_r16_2ep_5981133_1'),
]


def historical_protocol_rows():
    rows = []
    for name, filename, run in SELECTION_RUNS:
        path = HISTORICAL/filename
        with path.open() as stream:
            matches = [r for r in csv.DictReader(stream) if r['experiment_dir'] == run]
        if len(matches) != 1:
            raise RuntimeError('Historical validation run missing/ambiguous: '+run)
        source = matches[0]
        rows.append(dict(condition=name, run=run, sampling=source['sampling'],
                         learning_rate=float(source['learning_rate']), rank=int(source['lora_r']),
                         epochs=int(float(source['epochs'])), macro_f1=float(source['macro_f1']),
                         debt_f1=float(source['debt_collection_f1']),
                         source=str(path.relative_to(ROOT)), source_sha256=sha_file(path)))
    return rows


def verified_confusions():
    """Reconstruct all eight LLM matrices from predictions, then check stored means."""
    target = load_frame(DATA/'internal_test.csv')
    ids = target.record_id.astype(str).tolist()
    truth = target['product'].tolist()
    matrices, sources = {}, []
    with np.load(ROOT/'analysis/postprocessing/corrected_confusion_counts.npz') as stored:
        for model in ['qwen', 'llama']:
            for suffix, variants in [('base', ['base']), ('ft', ['42', '43', '44'])]:
                counts = []
                for variant in variants:
                    path = EXP/'evaluation/evaluate'/model/variant/'suite/internal_test_predictions.csv.gz'
                    pred = pd.read_csv(path, keep_default_na=False, dtype={'record_id': str})
                    if pred.record_id.tolist() != ids or pred.true_label.tolist() != truth:
                        raise RuntimeError('Restored error analysis has misaligned identities/labels')
                    if set(pred.prediction)-set(LABELS)-{'__off_label__'}:
                        raise RuntimeError('Unknown prediction in restored error analysis')
                    counts.append(confusion_matrix(truth, pred.prediction, labels=LABELS+['__off_label__']))
                    sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path)))
                key = model+'_'+suffix
                matrices[key] = np.mean(counts, axis=0)
                if not np.array_equal(matrices[key], stored[key]):
                    raise RuntimeError('Stored confusion matrix differs from saved predictions: '+key)
    return matrices, sources

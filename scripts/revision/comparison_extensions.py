"""Additional contrasts from the same saved joint bootstrap, without model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd

ENCODERS = ['bert', 'distilbert', 'roberta']
METRICS = ['accuracy', 'macro_f1', 'weighted_f1']


def paired_system_difference(points, samples, reference_indices, comparison_indices):
    """Average every refit in both systems before taking the paired difference."""
    a, b = list(reference_indices), list(comparison_indices)
    delta = points[b].mean(axis=0) - points[a].mean(axis=0)
    draws = samples[:, b].mean(axis=1) - samples[:, a].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return delta, low, high


def encoder_contrasts(directory):
    directory = Path(directory)
    seeds = pd.read_csv(directory/'corrected_per_seed_metrics.csv')
    rows = []
    for role in ['internal', 'legacy']:
        with np.load(directory/f'corrected_bootstrap_{role}.npz') as stored:
            names = stored['variant'].tolist()
            assert stored['metric'].tolist() == METRICS
            samples = stored['samples']
        values = seeds.loc[seeds.role.eq(role)].pivot(index='variant', columns='metric', values='value')
        points = values.loc[names, METRICS].to_numpy()
        for encoder in ENCODERS:
            ref = [names.index(f'{encoder}_{seed}') for seed in [42, 43, 44]]
            for model in ['qwen', 'llama']:
                comp = [names.index(f'{model}_{seed}') for seed in [42, 43, 44]]
                point, low, high = paired_system_difference(points, samples, ref, comp)
                for k, metric in enumerate(METRICS):
                    rows.append(dict(role=role, reference=encoder, comparison=model+'_ft',
                                     metric=metric, paired_difference=point[k], low=low[k], high=high[k]))
    return pd.DataFrame(rows)

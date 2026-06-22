from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values: np.ndarray, metric: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = []
    values = np.asarray(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(metric(sample))
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return {'lower': float(lower), 'upper': float(upper)}


def friedman_test(data_matrix: np.ndarray) -> Tuple[float, float]:
    if data_matrix.ndim != 2:
        raise ValueError('data_matrix must be 2D (datasets x methods)')
    stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
    return float(stat), float(p_value)


def nemenyi_posthoc(data_matrix: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    n, k = data_matrix.shape
    ranks = stats.rankdata(data_matrix, axis=1)
    avg_ranks = ranks.mean(axis=0)
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    diffs = np.abs(avg_ranks[:, None] - avg_ranks[None, :])
    significant = diffs > cd
    return pd.DataFrame(significant)

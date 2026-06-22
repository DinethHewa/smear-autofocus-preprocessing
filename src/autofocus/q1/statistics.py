from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def bootstrap_ci_mean(values: Sequence[float], *, n_resamples: int = 1000, conf_level: float = 0.95, seed: int = 123) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n_resamples)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    alpha = 1.0 - float(conf_level)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _rank_biserial(delta: np.ndarray) -> float:
    nonzero = np.asarray(delta, dtype=np.float64)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if nonzero.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    denom = pos + neg
    return float((pos - neg) / denom) if denom > 0 else float("nan")


def friedman_wilcoxon_holm(
    matrix: np.ndarray,
    method_names: Sequence[str],
    block_names: Sequence[str],
    *,
    family: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    arr = np.asarray(matrix, dtype=np.float64)
    warnings: List[str] = []
    friedman_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {arr.shape}")
    valid_cols = np.all(np.isfinite(arr), axis=0)
    aligned = arr[:, valid_cols]
    aligned_blocks = [str(name) for idx, name in enumerate(block_names) if valid_cols[idx]]
    if aligned.shape[0] < 3 or aligned.shape[1] < 3:
        note = "Need at least three methods and three finite blocks for Friedman."
        warnings.append(note)
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "statistic": float("nan"),
                "p_value": float("nan"),
                "valid": False,
                "note": note,
            }
        )
        return friedman_rows, pairwise_rows, warnings
    try:
        stat = friedmanchisquare(*[aligned[row_idx, :] for row_idx in range(aligned.shape[0])])
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "methods": ",".join(str(name) for name in method_names),
                "blocks": ",".join(aligned_blocks),
                "statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "valid": True,
                "note": "",
            }
        )
    except Exception as exc:
        note = f"Friedman failed: {exc}"
        warnings.append(note)
        friedman_rows.append(
            {
                "family": family,
                "n": int(aligned.shape[1]),
                "k": int(aligned.shape[0]),
                "statistic": float("nan"),
                "p_value": float("nan"),
                "valid": False,
                "note": note,
            }
        )
    raw_rows: List[Dict[str, Any]] = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            delta = aligned[j, :] - aligned[i, :]
            row: Dict[str, Any] = {
                "family": family,
                "method_a": str(method_names[i]),
                "method_b": str(method_names[j]),
                "n": int(np.isfinite(delta).sum()),
                "statistic": float("nan"),
                "p_raw": float("nan"),
                "p_holm": float("nan"),
                "delta_mean": float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan"),
                "delta_median": float(np.nanmedian(delta)) if np.isfinite(delta).any() else float("nan"),
                "rank_biserial": _rank_biserial(delta),
                "valid": False,
                "note": "",
            }
            nonzero = delta[np.isfinite(delta) & (delta != 0)]
            if nonzero.size < 3:
                row["note"] = "Too few non-zero paired differences for Wilcoxon."
                raw_rows.append(row)
                continue
            try:
                stat = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", correction=False, method="auto")
                row["statistic"] = float(stat.statistic)
                row["p_raw"] = float(stat.pvalue)
                row["valid"] = True
            except Exception as exc:
                row["note"] = f"Wilcoxon failed: {exc}"
            raw_rows.append(row)
    valid = [row for row in raw_rows if np.isfinite(row["p_raw"])]
    valid.sort(key=lambda row: row["p_raw"])
    m = len(valid)
    holm_values = []
    for rank, row in enumerate(valid, start=1):
        holm_values.append(min(1.0, (m - rank + 1) * float(row["p_raw"])))
    if holm_values:
        holm_values = np.maximum.accumulate(np.asarray(holm_values, dtype=np.float64)).tolist()
    for row, holm in zip(valid, holm_values):
        row["p_holm"] = float(holm)
    return friedman_rows, raw_rows, warnings


def nemenyi_cd(num_methods: int, num_blocks: int, *, alpha: float = 0.05) -> float:
    if num_methods < 2 or num_blocks < 1:
        return float("nan")
    try:
        from scipy.stats import studentized_range

        q_alpha = float(studentized_range.ppf(1.0 - alpha, num_methods, np.inf) / np.sqrt(2.0))
        return float(q_alpha * np.sqrt(num_methods * (num_methods + 1) / (6.0 * num_blocks)))
    except Exception:
        return float("nan")

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np

EPS = 1e-8
FAIL_PENALTY = 1e6


def _moving_average(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    pad = window // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y_pad, kernel, mode="valid")


def _count_local_maxima(y: np.ndarray, threshold: float) -> int:
    if y.size < 3:
        return 0
    count = 0
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > threshold:
            count += 1
    return count


def unimodal_peak_details(curve: np.ndarray) -> Tuple[float, Dict[str, float]]:
    y = np.asarray(curve, dtype=np.float64)
    if y.size == 0:
        return float(FAIL_PENALTY), {
            "pred_peak": float("nan"),
            "flat_penalty": 1.0,
            "multi_peak_penalty": float("nan"),
            "peak_prominence": float("nan"),
            "num_peaks": float("nan"),
            "nonfinite": 1.0,
        }

    y = np.where(np.isfinite(y), y, np.nan)
    if not np.isfinite(y).any():
        return float(FAIL_PENALTY), {
            "pred_peak": float("nan"),
            "flat_penalty": 1.0,
            "multi_peak_penalty": float("nan"),
            "peak_prominence": float("nan"),
            "num_peaks": float("nan"),
            "nonfinite": 1.0,
        }

    median = float(np.nanmedian(y))
    y = np.where(np.isfinite(y), y, median)
    y_s = _moving_average(y, window=3)

    med = float(np.median(y_s))
    std = float(np.std(y_s))
    peak_idx = int(np.argmax(y_s))
    peak_prom = float(y_s[peak_idx] - med)
    threshold = med + 0.25 * std
    num_peaks = _count_local_maxima(y_s, threshold)
    multi_penalty = max(0.0, float(num_peaks - 1))
    flat_penalty = 1.0 if std < 1e-6 else 0.0

    score = -peak_prom + 1.0 * multi_penalty + 10.0 * flat_penalty
    if not np.isfinite(score):
        score = float(FAIL_PENALTY)

    return float(score), {
        "pred_peak": float(peak_idx),
        "flat_penalty": float(flat_penalty),
        "multi_peak_penalty": float(multi_penalty),
        "peak_prominence": float(peak_prom),
        "num_peaks": float(num_peaks),
        "nonfinite": 0.0,
    }


def unimodal_peak_objective(curve: np.ndarray) -> float:
    score, _ = unimodal_peak_details(curve)
    return float(score)

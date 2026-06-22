from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

EPS = 1e-12
MeasureCallable = Callable[[np.ndarray], float]


def normalize_focus_curve(curve: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if curve_arr.size == 0:
        raise ValueError("Focus curve is empty")
    finite = np.isfinite(curve_arr)
    if not finite.any():
        return np.zeros_like(curve_arr, dtype=np.float64)
    clean = curve_arr.copy()
    clean[~finite] = float(np.nanmedian(clean[finite]))
    cmin = float(np.min(clean))
    cmax = float(np.max(clean))
    denom = cmax - cmin
    if denom <= eps:
        return np.zeros_like(clean, dtype=np.float64)
    return (clean - cmin) / (denom + eps)


def tie_break_peak(indices: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in indices)
    if not ordered:
        raise ValueError("No peak candidates supplied")
    return ordered[len(ordered) // 2]


def predict_peak_index(curve: np.ndarray) -> int:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if curve_arr.size == 0:
        raise ValueError("Focus curve is empty")
    finite = np.isfinite(curve_arr)
    if not finite.any():
        return 0
    best = float(np.nanmax(curve_arr))
    tied = np.where(np.isclose(curve_arr, best, equal_nan=False))[0].tolist()
    return tie_break_peak(tied)


def compute_focus_curve_for_stack(stack: np.ndarray, measure_func: MeasureCallable) -> np.ndarray:
    stack_arr = np.asarray(stack)
    if stack_arr.ndim != 3:
        raise ValueError(f"Expected stack shape (num_slices, H, W), got {stack_arr.shape}")
    scores = [float(measure_func(slice_2d)) for slice_2d in stack_arr]
    return np.asarray(scores, dtype=np.float64)


def polarity_align_curve(curve: np.ndarray, *, maximize: bool = True) -> np.ndarray:
    curve_arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    return curve_arr if maximize else -curve_arr


def curve_to_json(curve: np.ndarray) -> str:
    import json

    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    return json.dumps([None if not np.isfinite(value) else float(value) for value in arr])


def curve_from_json(text: str) -> np.ndarray:
    import json

    values = json.loads(text)
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)

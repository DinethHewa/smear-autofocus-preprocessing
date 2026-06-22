from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .curves import normalize_focus_curve, predict_peak_index

EPS = 1e-12

AUTOFOCUS_METRICS = (
    "absolute_peak_localization_error",
    "fwhm",
    "curvature_at_peak",
    "steep_slope_width",
    "steep_to_gradual_slope_ratio",
    "false_maxima_count",
    "noise_level",
    "rrmse_under_additive_noise",
    "range_around_global_maximum",
    "execution_time_per_slice",
)

METRIC_DIRECTION: Dict[str, bool] = {
    "absolute_peak_localization_error": True,
    "fwhm": True,
    "curvature_at_peak": False,
    "steep_slope_width": True,
    "steep_to_gradual_slope_ratio": False,
    "false_maxima_count": True,
    "noise_level": True,
    "rrmse_under_additive_noise": True,
    "range_around_global_maximum": True,
    "execution_time_per_slice": True,
}


@dataclass(frozen=True)
class MetricConfig:
    noise_std: float = 0.01
    noise_seed: int = 123
    rrmse_cap: int | None = None


def absolute_peak_localization_error(curve: np.ndarray, reference_idx: int) -> float:
    return float(abs(predict_peak_index(curve) - int(reference_idx)))


def full_width_half_maximum(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_val = float(np.max(y))
    if peak_val <= EPS:
        return float(len(y))
    idx = np.where(y >= 0.5 * peak_val)[0]
    if idx.size == 0:
        return float(len(y))
    return float(idx[-1] - idx[0] + 1)


def curvature_at_peak(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(y)
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return 0.0
    second_diff = y[peak_idx - 1] - 2.0 * y[peak_idx] + y[peak_idx + 1]
    return float(max(0.0, -second_diff))


def _local_minima_indices(curve: np.ndarray) -> List[int]:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    out: List[int] = []
    for idx in range(1, len(y) - 1):
        if y[idx] <= y[idx - 1] and y[idx] <= y[idx + 1]:
            out.append(idx)
    return out


def steep_slope_width(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(y)
    minima = _local_minima_indices(y)
    left = max([idx for idx in minima if idx < peak_idx], default=0)
    right = min([idx for idx in minima if idx > peak_idx], default=len(y) - 1)
    return float(right - left)


def steep_to_gradual_slope_ratio(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(y)
    local = []
    if peak_idx > 0:
        local.append(abs(float(y[peak_idx] - y[peak_idx - 1])))
    if peak_idx < len(y) - 1:
        local.append(abs(float(y[peak_idx] - y[peak_idx + 1])))
    local_mean = float(np.mean(local)) if local else 0.0
    diffs = np.abs(np.diff(y))
    mask = np.ones_like(diffs, dtype=bool)
    if 0 <= peak_idx - 1 < len(mask):
        mask[peak_idx - 1] = False
    if 0 <= peak_idx < len(mask):
        mask[peak_idx] = False
    background = diffs[mask]
    background_mean = float(np.mean(background)) if background.size else 0.0
    return float(local_mean / (background_mean + EPS))


def false_maxima_count(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_idx = predict_peak_index(y)
    count = 0
    for idx in range(1, len(y) - 1):
        if idx == peak_idx:
            continue
        if y[idx] > y[idx - 1] and y[idx] > y[idx + 1]:
            count += 1
    return float(count)


def noise_level(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    if len(y) < 3:
        return 0.0
    d2 = np.diff(y, n=2)
    return float(np.mean(d2**2))


def range_around_global_maximum(curve: np.ndarray) -> float:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    peak_val = float(np.max(y))
    if peak_val <= EPS:
        return float(len(y))
    peak_idx = predict_peak_index(y)
    threshold = 0.95 * peak_val
    left = peak_idx
    while left - 1 >= 0 and y[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    while right + 1 < len(y) and y[right + 1] >= threshold:
        right += 1
    return float(right - left + 1)


def add_noise_to_slice(slice_2d: np.ndarray, rng: np.random.Generator, *, noise_std: float) -> np.ndarray:
    image = np.asarray(slice_2d, dtype=np.float64)
    xmin = float(np.min(image))
    xmax = float(np.max(image))
    if xmax - xmin <= EPS:
        normalized = np.zeros_like(image, dtype=np.float64)
    else:
        normalized = (image - xmin) / (xmax - xmin + EPS)
    noisy = normalized + rng.normal(0.0, noise_std, size=normalized.shape)
    return np.clip(noisy, 0.0, 1.0)


def rrmse_under_additive_noise(
    clean_curve_norm: np.ndarray,
    stack: np.ndarray | None,
    measure_func: Callable[[np.ndarray], float] | None,
    rng: np.random.Generator,
    *,
    noise_std: float,
) -> float:
    if stack is None or measure_func is None:
        return float("nan")
    noisy_scores = [
        float(measure_func(add_noise_to_slice(slice_2d, rng, noise_std=noise_std)))
        for slice_2d in np.asarray(stack)
    ]
    noisy_curve = normalize_focus_curve(np.asarray(noisy_scores, dtype=np.float64))
    diff = np.asarray(clean_curve_norm, dtype=np.float64) - noisy_curve
    numerator = float(np.sqrt(np.mean(diff**2)))
    denominator = float(np.sqrt(np.mean(np.asarray(clean_curve_norm, dtype=np.float64) ** 2))) + EPS
    return float(numerator / denominator)


def compute_q1_metrics_for_stack(
    *,
    norm_curve: np.ndarray,
    reference_idx: int,
    execution_time: float,
    stack: np.ndarray | None = None,
    measure_func: Callable[[np.ndarray], float] | None = None,
    metric_config: MetricConfig | None = None,
    compute_rrmse: bool = True,
) -> Dict[str, float]:
    cfg = metric_config or MetricConfig()
    curve = np.asarray(norm_curve, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(cfg.noise_seed)
    return {
        "absolute_peak_localization_error": absolute_peak_localization_error(curve, reference_idx),
        "fwhm": full_width_half_maximum(curve),
        "curvature_at_peak": curvature_at_peak(curve),
        "steep_slope_width": steep_slope_width(curve),
        "steep_to_gradual_slope_ratio": steep_to_gradual_slope_ratio(curve),
        "false_maxima_count": false_maxima_count(curve),
        "noise_level": noise_level(curve),
        "rrmse_under_additive_noise": rrmse_under_additive_noise(
            curve,
            stack,
            measure_func,
            rng,
            noise_std=cfg.noise_std,
        )
        if compute_rrmse
        else float("nan"),
        "range_around_global_maximum": range_around_global_maximum(curve),
        "execution_time_per_slice": float(execution_time),
    }


def metric_definitions() -> List[Dict[str, str]]:
    descriptions = {
        "absolute_peak_localization_error": "Absolute difference between predicted and reference focus index.",
        "fwhm": "Full width at half maximum of the normalized focus curve.",
        "curvature_at_peak": "Local second-order peak sharpness proxy.",
        "steep_slope_width": "Width of the steep response region around the main peak.",
        "steep_to_gradual_slope_ratio": "Ratio of local peak slope to background slope level.",
        "false_maxima_count": "Count of local maxima excluding the global maximum.",
        "noise_level": "Mean squared second-difference fluctuation of the focus curve.",
        "rrmse_under_additive_noise": "Relative RMSE between clean and noisy normalized focus curves.",
        "range_around_global_maximum": "Width of the high-response plateau around the global maximum.",
        "execution_time_per_slice": "Average execution time per image/slice in seconds.",
    }
    rows = []
    for name in AUTOFOCUS_METRICS:
        rows.append(
            {
                "metric_name": name,
                "direction": "lower_is_better" if METRIC_DIRECTION[name] else "higher_is_better",
                "aggregation_transform": "identity" if METRIC_DIRECTION[name] else "reciprocal_for_value_scoring",
                "description": descriptions[name],
            }
        )
    return rows

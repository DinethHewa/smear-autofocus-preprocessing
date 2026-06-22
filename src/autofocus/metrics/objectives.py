from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


@dataclass
class ObjectiveConfig:
    alpha_softargmax: float = 6.0
    flat_threshold: float = 1e-6
    rough_weight: float = 0.05
    multi_peak_weight: float = 0.20
    peak_gap_weight: float = 0.10
    flat_weight: float = 0.10
    nonfinite_penalty: float = 1e6
    failure_penalty: float = 1e6
    confidence_floor: float = 0.0
    peak_gap_eps: float = 1e-8
    stability_eps: float = 1e-8
    stability_threshold: float = 1e4
    stability_penalty_weight: float = 1e6
    failure_rate_weight: float = 1e3
    unsupervised_multi_weight: float = 1.0
    unsupervised_flat_weight: float = 10.0
    unsupervised_peak_threshold_scale: float = 0.25
    unsupervised_smooth_window: int = 3


_OBJECTIVE_CONFIG = ObjectiveConfig()


def config_from_dict(cfg: Dict[str, float], failure_penalty: float | None = None) -> ObjectiveConfig:
    base = ObjectiveConfig()
    data = dict(cfg or {})
    if failure_penalty is not None:
        data.setdefault("failure_penalty", failure_penalty)
    for key, value in data.items():
        if hasattr(base, key):
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            setattr(base, key, value)
    return base


def set_objective_config(cfg: ObjectiveConfig) -> None:
    global _OBJECTIVE_CONFIG
    _OBJECTIVE_CONFIG = cfg


def get_objective_config() -> ObjectiveConfig:
    return _OBJECTIVE_CONFIG


def load_reference_peaks(ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference peaks and confidence from CSV or NPZ.

    CSV expects a reference peak column (reference_peak/ref_peak/peak) and optional confidence.
    NPZ expects peaks or reference_peak and optional confidence.
    """
    p = Path(ref_path)
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        peak_col = None
        for name in ("reference_peak", "ref_peak", "peak"):
            if name in df.columns:
                peak_col = name
                break
        if peak_col is None:
            raise ValueError("Reference CSV must contain a reference_peak/ref_peak/peak column")
        peaks = df[peak_col].to_numpy(dtype=int)
        if "confidence" in df.columns:
            conf = df["confidence"].to_numpy(dtype=float)
        else:
            conf = np.ones(len(peaks), dtype=float)
        conf = np.clip(conf, 0.0, 1.0)
        return peaks, conf

    data = np.load(p, allow_pickle=True)
    keys = data.files
    if "peaks" in keys:
        peaks = data["peaks"].astype(int)
    elif "reference_peak" in keys:
        peaks = data["reference_peak"].astype(int)
    else:
        raise ValueError(f"Reference NPZ missing peaks. Keys: {keys}")
    if "confidence" in keys:
        conf = data["confidence"].astype(float)
    else:
        conf = np.ones_like(peaks, dtype=float)
    conf = np.clip(conf, 0.0, 1.0)
    return peaks, conf


def softargmax_peak(curve: np.ndarray, alpha: float) -> float:
    if not np.isfinite(curve).any():
        return float("nan")
    c = np.where(np.isfinite(curve), curve.astype(np.float64), -np.inf)
    m = np.max(c)
    if not np.isfinite(m):
        return float("nan")
    e = np.exp(alpha * (c - m))
    z = e.sum()
    if z <= 0 or not np.isfinite(z):
        return float("nan")
    idx = np.arange(len(curve), dtype=np.float64)
    return float((idx * e).sum() / z)


def curve_quality_penalties(curve: np.ndarray) -> Dict[str, float]:
    """
    Compute curve penalties used by the Q1 objective.
    """
    cfg = _OBJECTIVE_CONFIG
    c = curve.astype(np.float64)
    if not np.isfinite(c).any():
        return {"nonfinite": 1.0, "flat": 1.0, "rough": np.nan, "peak_gap": np.nan, "multi_peak": 1.0}
    c = np.where(np.isfinite(c), c, np.nan)
    finite = np.isfinite(c)
    if finite.sum() < 3:
        return {"nonfinite": 1.0, "flat": 1.0, "rough": np.nan, "peak_gap": np.nan, "multi_peak": 1.0}
    cf = c.copy()
    dyn = float(np.nanmax(cf) - np.nanmin(cf))
    flat = 1.0 if dyn < cfg.flat_threshold else 0.0
    cf = np.where(np.isfinite(cf), cf, np.nanmedian(cf))
    d2 = np.diff(cf, n=2)
    rough = float(np.mean(np.abs(d2)))
    order = np.argsort(cf)[::-1]
    top1, top2 = cf[order[0]], cf[order[1]]
    peak_gap = float(top1 - top2)
    near = np.sum(cf >= (top1 - 0.05 * (abs(top1) + 1e-6)))
    multi_peak = 1.0 if near > 2 else 0.0
    return {"nonfinite": 0.0, "flat": flat, "rough": rough, "peak_gap": peak_gap, "multi_peak": multi_peak}


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y, kernel, mode="same")


def _count_local_maxima(y: np.ndarray, threshold: float) -> int:
    if len(y) < 3:
        return 0
    count = 0
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > threshold:
            count += 1
    return count


def weighted_peak_error(pred_peak: float, ref_peak: float, confidence: float, z_len: int) -> float:
    """
    Confidence-weighted peak error normalized by the stack depth.
    """
    if not np.isfinite(pred_peak) or not np.isfinite(ref_peak):
        return float("nan")
    cfg = _OBJECTIVE_CONFIG
    w = float(np.clip(confidence, 0.0, 1.0))
    if w < cfg.confidence_floor:
        return 0.0
    if w <= 0.0:
        return 0.0
    denom = max(1, int(z_len) - 1)
    return float(abs(pred_peak - ref_peak) / denom) * w


def stack_fitness(curve: np.ndarray, ref_peak: int, confidence: float) -> Dict[str, float]:
    """
    Compute per-stack fitness with confidence-weighted peak error and curve penalties.
    """
    cfg = _OBJECTIVE_CONFIG
    penalties = curve_quality_penalties(curve)
    pred_peak = softargmax_peak(curve, cfg.alpha_softargmax)

    if penalties["nonfinite"] > 0.0 or not np.isfinite(pred_peak):
        return {
            "pred_peak": float(pred_peak) if np.isfinite(pred_peak) else float("nan"),
            "peak_error": float("nan"),
            "penalty_nonfinite": float(penalties["nonfinite"]),
            "penalty_flat": float(penalties["flat"]),
            "penalty_rough": float("nan") if not np.isfinite(penalties["rough"]) else float(penalties["rough"]),
            "penalty_multi_peak": float(penalties["multi_peak"]),
            "peak_gap": float("nan") if not np.isfinite(penalties["peak_gap"]) else float(penalties["peak_gap"]),
            "total_fitness": float(cfg.failure_penalty),
            "is_failure": 1.0,
        }

    peak_error = weighted_peak_error(pred_peak, float(ref_peak), float(confidence), len(curve))
    base_error = 0.0 if not np.isfinite(peak_error) else float(peak_error)
    rough = penalties["rough"]
    if not np.isfinite(rough):
        rough = 1e3
    multi_peak = penalties["multi_peak"]
    if not np.isfinite(multi_peak):
        multi_peak = 1.0
    peak_gap = penalties["peak_gap"]
    if (not np.isfinite(peak_gap)) or peak_gap <= 0:
        peak_gap = 0.0

    # Objective: L = w*|p-r|/Z + a*rough + b*multi + c*(1/(peak_gap+eps)) + d*flat
    total = base_error
    total += cfg.rough_weight * float(rough)
    total += cfg.multi_peak_weight * float(multi_peak)
    total += cfg.peak_gap_weight * float(1.0 / (peak_gap + cfg.peak_gap_eps))
    total += cfg.flat_weight * float(penalties["flat"])

    if not np.isfinite(total):
        total = float(cfg.failure_penalty)
        is_failure = 1.0
    else:
        is_failure = 0.0

    return {
        "pred_peak": float(pred_peak),
        "peak_error": float(peak_error) if np.isfinite(peak_error) else float("nan"),
        "penalty_nonfinite": float(penalties["nonfinite"]),
        "penalty_flat": float(penalties["flat"]),
        "penalty_rough": float(rough),
        "penalty_multi_peak": float(multi_peak),
        "peak_gap": float(peak_gap),
        "total_fitness": float(total),
        "is_failure": float(is_failure),
    }


def unimodal_peak_fitness(curve: np.ndarray) -> Dict[str, float]:
    """
    Unsupervised objective focusing on unimodal peak structure.
    """
    cfg = _OBJECTIVE_CONFIG
    y = np.asarray(curve, dtype=np.float64)
    if not np.isfinite(y).any():
        return {
            "pred_peak": float("nan"),
            "peak_error": float("nan"),
            "penalty_nonfinite": 1.0,
            "penalty_flat": 1.0,
            "penalty_rough": float("nan"),
            "penalty_multi_peak": float("nan"),
            "peak_gap": float("nan"),
            "total_fitness": float(cfg.failure_penalty),
            "is_failure": 1.0,
        }

    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    y_s = _moving_average(y, int(cfg.unsupervised_smooth_window))
    median = float(np.median(y_s))
    std = float(np.std(y_s))
    peak_idx = int(np.argmax(y_s))
    peak_val = float(y_s[peak_idx])
    peak_prom = peak_val - median
    threshold = median + cfg.unsupervised_peak_threshold_scale * std
    num_peaks = _count_local_maxima(y_s, threshold)
    multi_penalty = max(0.0, float(num_peaks - 1))
    flat_penalty = 1.0 if std < EPS else 0.0
    total = -peak_prom + cfg.unsupervised_multi_weight * multi_penalty + cfg.unsupervised_flat_weight * flat_penalty
    if not np.isfinite(total):
        total = float(cfg.failure_penalty)
        is_failure = 1.0
    else:
        is_failure = 0.0

    quality = curve_quality_penalties(y_s.astype(np.float32))
    return {
        "pred_peak": float(peak_idx),
        "peak_error": float("nan"),
        "penalty_nonfinite": float(quality["nonfinite"]),
        "penalty_flat": float(flat_penalty),
        "penalty_rough": float(quality["rough"]) if np.isfinite(quality["rough"]) else float("nan"),
        "penalty_multi_peak": float(multi_penalty),
        "peak_gap": float(quality["peak_gap"]) if np.isfinite(quality["peak_gap"]) else float("nan"),
        "total_fitness": float(total),
        "is_failure": float(is_failure),
        "peak_prominence": float(peak_prom),
        "num_peaks": int(num_peaks),
    }


def aggregate_dataset_fitness(per_stack_df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate per-stack results into dataset-level summary.
    """
    cfg = _OBJECTIVE_CONFIG
    values = per_stack_df["total_fitness"].to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    if finite_mask.any():
        mean_fitness = float(np.mean(values[finite_mask]))
        std_fitness = float(np.std(values[finite_mask]))
    else:
        mean_fitness = float(cfg.failure_penalty)
        std_fitness = float(cfg.failure_penalty)
    hard_failure_rate = float(per_stack_df["is_failure"].mean()) if len(per_stack_df) else 1.0
    stability_ratio = float(abs(mean_fitness) / (std_fitness + cfg.stability_eps))
    dataset_score = mean_fitness + std_fitness
    return {
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "dataset_score": dataset_score,
        "HardFailureRate": hard_failure_rate,
        "StabilityRatio": stability_ratio,
        "n_stacks": int(len(per_stack_df)),
    }


def generalization_score(
    dataset_summaries: List[Dict[str, float]],
    dataset_weights: Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Compute weighted mean and std across datasets to form a generalization score.
    """
    cfg = _OBJECTIVE_CONFIG
    if not dataset_summaries:
        return {
            "generalization_score": float(cfg.failure_penalty),
            "weighted_mean": float(cfg.failure_penalty),
            "weighted_std": float(cfg.failure_penalty),
            "stability_ratio": 0.0,
        }
    weights = []
    means = []
    for ds in dataset_summaries:
        name = str(ds.get("dataset"))
        if "dataset_score" in ds:
            means.append(float(ds["dataset_score"]))
        else:
            means.append(float(ds["mean_fitness"]))
        if dataset_weights and name in dataset_weights:
            weights.append(float(dataset_weights[name]))
        else:
            weights.append(1.0)
    w = np.asarray(weights, dtype=float)
    m = np.asarray(means, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    weighted_mean = float(np.sum(w * m))
    weighted_std = float(np.sqrt(np.sum(w * (m - weighted_mean) ** 2)))
    score = weighted_mean + weighted_std
    stability_ratio = float(abs(weighted_mean) / (weighted_std + cfg.stability_eps))
    return {
        "generalization_score": float(score),
        "weighted_mean": float(weighted_mean),
        "weighted_std": float(weighted_std),
        "stability_ratio": float(stability_ratio),
    }

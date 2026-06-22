from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .metrics import AUTOFOCUS_METRICS, METRIC_DIRECTION

EPS = 1e-12
DEFAULT_ALPHA = 0.7
DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    "absolute_peak_localization_error": 0.20,
    "range_around_global_maximum": 0.15,
    "false_maxima_count": 0.10,
    "fwhm": 0.10,
    "noise_level": 0.10,
    "steep_to_gradual_slope_ratio": 0.10,
    "execution_time_per_slice": 0.10,
    "steep_slope_width": 0.05,
    "curvature_at_peak": 0.05,
    "rrmse_under_additive_noise": 0.05,
}


def average_ranks(values: Sequence[float], *, lower_better: bool = True) -> List[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, np.inf if lower_better else -np.inf)
    order = np.argsort(vals if lower_better else -vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=np.float64)
    idx = 0
    while idx < len(vals):
        jdx = idx
        current = vals[order[idx]]
        while jdx + 1 < len(vals) and np.isclose(vals[order[jdx + 1]], current):
            jdx += 1
        avg_rank = (idx + jdx) / 2.0 + 1.0
        ranks[order[idx : jdx + 1]] = avg_rank
        idx = jdx + 1
    return ranks.tolist()


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(vals) & np.isfinite(w)
    if not finite.any():
        return float("nan")
    vals = vals[finite]
    w = w[finite]
    if np.sum(w) <= 0:
        return float(np.mean(vals))
    return float(np.sum(vals * w) / (np.sum(w) + EPS))


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(vals) & np.isfinite(w)
    if not finite.any():
        return float("nan")
    vals = vals[finite]
    w = w[finite]
    if np.sum(w) <= 0:
        return float(np.std(vals, ddof=0))
    mean_value = weighted_mean(vals, w)
    return float(np.sqrt(np.sum(w * (vals - mean_value) ** 2) / (np.sum(w) + EPS)))


def align_metric_value(metric_name: str, raw_value: float) -> float:
    if not np.isfinite(raw_value):
        return float("nan")
    lower_is_better = bool(METRIC_DIRECTION[metric_name])
    if lower_is_better:
        return float(raw_value)
    return float(1.0 / (float(raw_value) + EPS))


def minmax_normalize_across_methods(values_by_method: Mapping[str, float]) -> Dict[str, float]:
    keys = list(values_by_method.keys())
    vals = np.asarray([values_by_method[key] for key in keys], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return {key: float("nan") for key in keys}
    vmin = float(np.min(vals[finite]))
    vmax = float(np.max(vals[finite]))
    if vmax - vmin <= EPS:
        return {key: 0.0 if np.isfinite(values_by_method[key]) else float("nan") for key in keys}
    return {
        key: float((float(values_by_method[key]) - vmin) / (vmax - vmin + EPS))
        if np.isfinite(values_by_method[key])
        else float("nan")
        for key in keys
    }


def dataset_metric_summary_from_per_stack(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, method), subset in sorted(grouped.items()):
        base = {
            "dataset": dataset,
            "method": method,
            "n_stacks": len(subset),
            "failure_count": int(sum(1 for row in subset if int(float(row.get("failure", 0))) != 0)),
        }
        for metric in AUTOFOCUS_METRICS:
            values = np.asarray([float(row.get(metric, float("nan"))) for row in subset], dtype=float)
            base[f"{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            base[f"{metric}_std"] = float(np.nanstd(values)) if np.isfinite(values).any() else float("nan")
        out.append(base)
    return out


def _dataset_metric_raw(dataset_summary: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in dataset_summary:
        dataset = str(row["dataset"])
        method = str(row["method"])
        raw.setdefault(dataset, {})[method] = {
            metric: float(row.get(f"{metric}_mean", float("nan"))) for metric in metric_names
        }
    return raw


def compute_rank_based_summary(
    dataset_summary: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[float]]]]:
    raw = _dataset_metric_raw(dataset_summary, metric_names)
    if not raw:
        return [], {}
    methods = sorted({method for dataset in raw.values() for method in dataset})
    rank_cells = {method: {"ranks": []} for method in methods}
    for dataset in sorted(raw):
        available = raw[dataset]
        for metric in metric_names:
            aligned = [align_metric_value(metric, available.get(method, {}).get(metric, float("nan"))) for method in methods]
            ranks = average_ranks(aligned, lower_better=True)
            for method, rank_value in zip(methods, ranks):
                rank_cells[method]["ranks"].append(float(rank_value))
    rows: List[Dict[str, Any]] = []
    for method in methods:
        ranks = rank_cells[method]["ranks"]
        mean_rank = float(np.mean(ranks))
        std_rank = float(np.std(ranks, ddof=0))
        rows.append(
            {
                "method": method,
                "overall_rank_mean": mean_rank,
                "overall_rank_std": std_rank,
                "rank_generalization_score": float(alpha * mean_rank + (1.0 - alpha) * std_rank),
                "alpha": float(alpha),
            }
        )
    rows.sort(key=lambda row: row["rank_generalization_score"])
    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank
    return rows, rank_cells


def compute_value_based_summary(
    dataset_summary: Sequence[Mapping[str, Any]],
    *,
    dataset_stack_counts: Mapping[str, int] | None = None,
    dataset_weights: Mapping[str, float] | None = None,
    metric_names: Sequence[str] = AUTOFOCUS_METRICS,
    metric_weights: Mapping[str, float] = DEFAULT_METRIC_WEIGHTS,
    alpha: float = DEFAULT_ALPHA,
    weighting_mode: str = "equal_dataset",
) -> List[Dict[str, Any]]:
    raw = _dataset_metric_raw(dataset_summary, metric_names)
    if not raw:
        return []
    methods = sorted({method for dataset in raw.values() for method in dataset})
    dataset_scores: Dict[str, Dict[str, float]] = {dataset: {} for dataset in raw}
    for dataset in sorted(raw):
        normalized_by_metric: Dict[str, Dict[str, float]] = {}
        for metric in metric_names:
            aligned = {
                method: align_metric_value(metric, raw[dataset].get(method, {}).get(metric, float("nan")))
                for method in methods
            }
            normalized_by_metric[metric] = minmax_normalize_across_methods(aligned)
        for method in methods:
            values = []
            weights = []
            for metric in metric_names:
                value = normalized_by_metric[metric].get(method, float("nan"))
                if np.isfinite(value):
                    values.append(value)
                    weights.append(float(metric_weights.get(metric, 1.0)))
            dataset_scores[dataset][method] = weighted_mean(values, weights) if values else float("nan")
    rows: List[Dict[str, Any]] = []
    for method in methods:
        values = [dataset_scores[dataset][method] for dataset in sorted(raw)]
        if dataset_weights:
            weights = [float(dataset_weights.get(dataset, 1.0)) for dataset in sorted(raw)]
        elif weighting_mode == "per_stack":
            counts = dataset_stack_counts or {}
            weights = [float(counts.get(dataset, 1.0)) for dataset in sorted(raw)]
        elif weighting_mode == "equal_dataset":
            weights = [1.0 for _ in sorted(raw)]
        else:
            raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
        mean_value = weighted_mean(values, weights)
        std_value = weighted_std(values, weights)
        rows.append(
            {
                "method": method,
                "weighted_mean": mean_value,
                "weighted_std": std_value,
                "value_generalization_score": float(alpha * mean_value + (1.0 - alpha) * std_value),
                "dataset_weighting_mode": weighting_mode,
                "alpha": float(alpha),
            }
        )
    rows.sort(key=lambda row: row["value_generalization_score"])
    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank
    return rows

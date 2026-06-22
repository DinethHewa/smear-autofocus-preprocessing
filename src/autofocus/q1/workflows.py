from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.io import load_stack
from ..metrics.focus_measures import FOCUS_MEASURES, compute_focus_curve
from ..preprocess.ops import is_op_available
from ..preprocess.param_spaces import DEFAULT_PARAMS
from ..preprocess.pipeline import Pipeline
from .aggregation import compute_rank_based_summary, compute_value_based_summary, dataset_metric_summary_from_per_stack
from .curves import curve_to_json, normalize_focus_curve, polarity_align_curve, predict_peak_index
from .metrics import MetricConfig, compute_q1_metrics_for_stack


def pipeline_from_steps(steps: Sequence[Mapping[str, Any]]) -> Pipeline:
    return Pipeline.from_dict({"steps": [dict(step) for step in steps]}).resolved_defaults()


def configured_methods(config: Mapping[str, Any], *, include_single_ops: bool = True) -> List[Dict[str, Any]]:
    workflows_cfg = config.get("workflows", {}) if isinstance(config.get("workflows"), dict) else {}
    methods: List[Dict[str, Any]] = []
    for item in workflows_cfg.get("candidates", []) or []:
        name = str(item["name"])
        steps = list(item.get("steps") or [])
        methods.append(
            {
                "method": name,
                "method_type": str(item.get("type", "configured")),
                "pipeline": pipeline_from_steps(steps),
                "pipeline_json": json.dumps({"steps": steps}, sort_keys=True, default=str),
            }
        )
    if include_single_ops:
        for op_name in workflows_cfg.get("single_ops", []) or []:
            op_name = str(op_name)
            if not is_op_available(op_name):
                continue
            params = dict(DEFAULT_PARAMS.get(op_name, {}))
            steps = [{"name": op_name, "enabled": True, "category": "single", "params": params}]
            methods.append(
                {
                    "method": f"single::{op_name}",
                    "method_type": "single_operator",
                    "pipeline": pipeline_from_steps(steps),
                    "pipeline_json": json.dumps({"steps": steps}, sort_keys=True, default=str),
                }
            )
    if not methods:
        steps = [{"name": "identity", "enabled": True, "category": "baseline", "params": {}}]
        methods.append(
            {
                "method": "identity",
                "method_type": "baseline",
                "pipeline": pipeline_from_steps(steps),
                "pipeline_json": json.dumps({"steps": steps}, sort_keys=True, default=str),
            }
        )
    return methods


def _label_lookup(labels_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {str(row["stack_id"]): row for row in labels_df.to_dict("records")}


def _filter_manifest(manifest_df: pd.DataFrame, *, datasets: Sequence[str] | None = None, outer_split: Sequence[str] | None = None, inner_split: Sequence[str] | None = None) -> pd.DataFrame:
    df = manifest_df.copy()
    if datasets is not None:
        df = df[df["dataset"].isin(list(datasets))].copy()
    if outer_split is not None:
        df = df[df["outer_split"].isin(list(outer_split))].copy()
    if inner_split is not None:
        df = df[df["inner_split"].isin(list(inner_split))].copy()
    if df.empty:
        raise ValueError("No manifest rows left after applying evaluation filters")
    return df


def evaluate_methods(
    *,
    manifest_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    methods: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    datasets: Sequence[str] | None = None,
    outer_split: Sequence[str] | None = None,
    inner_split: Sequence[str] | None = None,
    write_curves: bool = True,
    compute_metrics: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    focus_cfg = config.get("focus_measure", {}) if isinstance(config.get("focus_measure"), dict) else {}
    measure_name = str(focus_cfg.get("name", "variance_of_laplacian"))
    if measure_name not in FOCUS_MEASURES:
        raise ValueError(f"Unknown focus_measure.name for Q1 pipeline: {measure_name}")
    maximize = bool(focus_cfg.get("maximize", True))
    focus_backend = str(focus_cfg.get("backend", focus_cfg.get("curve_backend", "cpu")))
    cuda_cfg = focus_cfg.get("cuda", {}) if isinstance(focus_cfg.get("cuda"), dict) else {}
    measure_func = FOCUS_MEASURES[measure_name]
    metric_cfg_raw = config.get("statistics", {}) if isinstance(config.get("statistics"), dict) else {}
    metric_cfg = MetricConfig(
        noise_std=float(metric_cfg_raw.get("noise_std", 0.01)),
        noise_seed=int(metric_cfg_raw.get("noise_seed", 123)),
    )
    compute_rrmse = bool(metric_cfg_raw.get("compute_rrmse", True))
    df = _filter_manifest(manifest_df, datasets=datasets, outer_split=outer_split, inner_split=inner_split)
    labels = _label_lookup(labels_df)
    per_stack_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    for method_entry in methods:
        method = str(method_entry["method"])
        method_type = str(method_entry.get("method_type", "unknown"))
        pipeline: Pipeline = method_entry["pipeline"]
        pipeline_json = str(method_entry.get("pipeline_json", json.dumps(pipeline.to_dict(), sort_keys=True, default=str)))
        for row in df.to_dict("records"):
            stack_id = str(row["stack_id"])
            label = labels.get(stack_id)
            if label is None:
                raise KeyError(f"Missing reference label for stack_id={stack_id}")
            start = time.perf_counter()
            failure = 0
            failure_reason = ""
            raw_curve = np.asarray([], dtype=np.float64)
            norm_curve = np.asarray([], dtype=np.float64)
            pred_idx = -1
            metrics: Dict[str, float] = {}
            try:
                stack = load_stack(row["path"], stack_index=int(row["stack_index"]))
                processed = np.stack([pipeline.apply(stack[idx]) for idx in range(stack.shape[0])], axis=0)
                raw_curve = polarity_align_curve(
                    compute_focus_curve(
                        processed,
                        name=measure_name,
                        backend=focus_backend,
                        cuda_validate=bool(cuda_cfg.get("validate", False)),
                        cuda_validation_min_corr=float(cuda_cfg.get("validation_min_corr", 0.95)),
                        cuda_validation_max_peak_delta=int(cuda_cfg.get("validation_max_peak_delta", 1)),
                        cuda_fallback=bool(cuda_cfg.get("fallback", True)),
                    ),
                    maximize=maximize,
                )
                norm_curve = normalize_focus_curve(raw_curve)
                pred_idx = predict_peak_index(norm_curve)
                runtime = float(time.perf_counter() - start)
                per_image_time = runtime / max(1, int(processed.shape[0]))
                if compute_metrics:
                    metrics = compute_q1_metrics_for_stack(
                        norm_curve=norm_curve,
                        reference_idx=int(label["reference_focus_index"]),
                        execution_time=per_image_time,
                        stack=processed,
                        measure_func=measure_func,
                        metric_config=metric_cfg,
                        compute_rrmse=compute_rrmse,
                    )
            except Exception as exc:
                runtime = float(time.perf_counter() - start)
                per_image_time = runtime
                failure = 1
                failure_reason = f"{type(exc).__name__}: {exc}"
                if compute_metrics:
                    metrics = {name: float("nan") for name in (
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
                    )}
                    metrics["execution_time_per_slice"] = float(per_image_time)
                failure_rows.append(
                    {
                        "dataset": row["dataset"],
                        "stack_id": stack_id,
                        "method": method,
                        "failure_reason": failure_reason,
                    }
                )
            base = {
                "dataset": str(row["dataset"]),
                "stack_id": stack_id,
                "group_id": str(row["group_id"]),
                "stack_index": int(row["stack_index"]),
                "outer_split": str(row["outer_split"]),
                "inner_split": str(row["inner_split"]),
                "method": method,
                "method_type": method_type,
                "pipeline_json": pipeline_json,
                "focus_measure": measure_name,
                "predicted_focus_index": int(pred_idx),
                "reference_focus_index": int(label["reference_focus_index"]),
                "label_source": str(label["label_source"]),
                "label_confidence": float(label.get("label_confidence", 0.0)),
                "runtime_sec_per_stack": runtime,
                "failure": int(failure),
                "failure_reason": failure_reason,
            }
            per_stack_rows.append({**base, **metrics})
            if write_curves:
                curve_rows.append(
                    {
                        **base,
                        "raw_curve": curve_to_json(raw_curve),
                        "normalized_curve": curve_to_json(norm_curve),
                    }
                )
    dataset_summary = dataset_metric_summary_from_per_stack(per_stack_rows)
    return per_stack_rows, dataset_summary, curve_rows


def summarize_methods(per_stack_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dataset_summary = dataset_metric_summary_from_per_stack(per_stack_rows)
    stack_counts: Dict[str, int] = {}
    for row in per_stack_rows:
        dataset = str(row["dataset"])
        stack_counts[dataset] = max(stack_counts.get(dataset, 0), 0) + (1 if str(row.get("method")) == str(per_stack_rows[0].get("method")) else 0)
    agg_cfg = config.get("aggregation", {}) if isinstance(config.get("aggregation"), dict) else {}
    alpha = float(agg_cfg.get("alpha", 0.7))
    metric_weights = dict(agg_cfg.get("metric_weights") or {})
    from .aggregation import DEFAULT_METRIC_WEIGHTS

    if not metric_weights:
        metric_weights = DEFAULT_METRIC_WEIGHTS
    rank_rows, rank_cells = compute_rank_based_summary(dataset_summary, alpha=alpha)
    value_rows = compute_value_based_summary(
        dataset_summary,
        dataset_stack_counts=stack_counts,
        metric_weights=metric_weights,
        alpha=alpha,
        weighting_mode=str(agg_cfg.get("dataset_weighting", "equal_dataset")),
    )
    method_rows: List[Dict[str, Any]] = []
    by_rank = {row["method"]: row for row in rank_rows}
    by_value = {row["method"]: row for row in value_rows}
    methods = sorted(set(by_rank) | set(by_value))
    for method in methods:
        failures = [int(float(row.get("failure", 0))) for row in per_stack_rows if str(row.get("method")) == method]
        runtimes = [float(row.get("execution_time_per_slice", float("nan"))) for row in per_stack_rows if str(row.get("method")) == method]
        method_rows.append(
            {
                "method": method,
                "rank_generalization_score": by_rank.get(method, {}).get("rank_generalization_score", float("nan")),
                "rank_final_rank": by_rank.get(method, {}).get("final_rank", ""),
                "value_generalization_score": by_value.get(method, {}).get("value_generalization_score", float("nan")),
                "value_final_rank": by_value.get(method, {}).get("final_rank", ""),
                "failure_count": int(sum(failures)),
                "failure_rate": float(np.mean(failures)) if failures else float("nan"),
                "mean_execution_time": float(np.nanmean(runtimes)) if np.isfinite(runtimes).any() else float("nan"),
                "score_direction": "lower_is_better",
            }
        )
    method_rows.sort(key=lambda row: (float(row["rank_generalization_score"]), float(row["value_generalization_score"])))
    metadata = {"rank_cells": rank_cells, "dataset_stack_counts": stack_counts}
    return dataset_summary, rank_rows, value_rows, {"method_summary": method_rows, **metadata}


def select_best_method(method_summary: Sequence[Mapping[str, Any]], *, exclude_oracle: bool = True) -> str:
    candidates = [dict(row) for row in method_summary]
    if exclude_oracle:
        candidates = [row for row in candidates if "oracle" not in str(row.get("method", "")).lower()]
    if not candidates:
        raise ValueError("No candidate methods available")
    candidates.sort(key=lambda row: (float(row.get("rank_generalization_score", float("inf"))), float(row.get("value_generalization_score", float("inf")))))
    return str(candidates[0]["method"])


def method_by_name(methods: Sequence[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    for method in methods:
        if str(method["method"]) == str(name):
            return dict(method)
    raise KeyError(f"Unknown method: {name}")

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import math

import pandas as pd
import yaml

from ..metrics import objectives


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _parse_failures(df: pd.DataFrame) -> Tuple[int, int]:
    total = int(len(df))
    for col in ("failed", "is_failure"):
        if col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                mask = series.fillna(0).astype(float) != 0.0
            else:
                mask = series.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
            return int(mask.sum()), total
    return 0, total


def _compute_ms_per_image(df: pd.DataFrame, total: int) -> float | None:
    if "runtime_sec" in df.columns and total > 0:
        runtimes = pd.to_numeric(df["runtime_sec"], errors="coerce").fillna(0.0)
        return float(1000.0 * runtimes.sum() / total)
    if "ms_per_image" in df.columns:
        vals = pd.to_numeric(df["ms_per_image"], errors="coerce")
        if vals.notna().any():
            return float(vals.mean())
    return None


def build_eval_summary(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_id = run_path.name

    config_used = run_path / "config_used.yaml"
    cfg = _read_yaml(config_used) if config_used.exists() else {}
    evaluation_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    ref_paths = evaluation_cfg.get("reference_paths", {})
    cached = evaluation_cfg.get("cached_scores", {}) if isinstance(evaluation_cfg.get("cached_scores"), dict) else {}
    cached_refs = cached.get("ref_paths", [])
    has_refs = False
    if isinstance(ref_paths, dict):
        has_refs = bool(ref_paths)
    elif isinstance(ref_paths, list):
        has_refs = bool(ref_paths)
    if not has_refs and isinstance(cached_refs, list):
        has_refs = bool(cached_refs)
    objective_mode = "reference" if has_refs else "unsupervised"
    unsup_name = "unimodal_peak" if objective_mode == "unsupervised" else None

    dataset_summary = None
    for name in ("dataset_summary_eval.csv", "dataset_summary.csv"):
        path = run_path / name
        if path.exists():
            dataset_summary = _read_csv(path)
            break

    generalization_score = None
    stability_ratio = None
    run_summary_path = run_path / "run_summary.json"
    if run_summary_path.exists():
        try:
            data = json.loads(run_summary_path.read_text(encoding="utf-8"))
            if "generalization_score" in data:
                generalization_score = float(data["generalization_score"])
            if "global_stability_ratio" in data:
                stability_ratio = float(data["global_stability_ratio"])
        except Exception:
            pass
    if dataset_summary is not None and "dataset" in dataset_summary.columns and "mean_fitness" in dataset_summary.columns:
        rows = []
        for row in dataset_summary.to_dict("records"):
            entry = {"dataset": str(row["dataset"]), "mean_fitness": float(row["mean_fitness"])}
            if "dataset_score" in row:
                entry["dataset_score"] = float(row["dataset_score"])
            rows.append(entry)
        dataset_weights = {}
        metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics"), dict) else {}
        for key, value in (metrics_cfg.get("dataset_weights", {}) or {}).items():
            try:
                dataset_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        gen = objectives.generalization_score(rows, dataset_weights)
        generalization_score = float(gen["generalization_score"])
        stability_ratio = float(gen["stability_ratio"])

    per_stack = None
    for name in ("per_stack_results_eval.csv", "eval_results_eval.csv", "per_stack_results.csv"):
        path = run_path / name
        if path.exists():
            per_stack = _read_csv(path)
            break

    failures_failed = 0
    failures_total = 0
    ms_per_image = None
    if per_stack is not None:
        failures_failed, failures_total = _parse_failures(per_stack)
        ms_per_image = _compute_ms_per_image(per_stack, failures_total)

    if generalization_score is None or math.isnan(generalization_score):
        generalization_score = None
    if stability_ratio is None or math.isnan(stability_ratio):
        stability_ratio = None
    if ms_per_image is None or math.isnan(ms_per_image):
        ms_per_image = None

    return {
        "run_id": run_id,
        "generalization_score": generalization_score,
        "stability_ratio": stability_ratio,
        "ms_per_image": ms_per_image,
        "failures_failed": failures_failed,
        "failures_total": failures_total,
        "objective_mode": objective_mode,
        "unsupervised_objective": unsup_name if objective_mode == "unsupervised" else None,
    }


def write_eval_summary(
    run_dir: str | Path,
    run_id: str,
    objective_mode: str | None,
    unsupervised_name: str | None,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    summary = build_eval_summary(run_path)
    summary["run_id"] = run_id
    if objective_mode:
        summary["objective_mode"] = objective_mode
    if summary.get("objective_mode") == "unsupervised":
        summary["unsupervised_objective"] = unsupervised_name or "unimodal_peak"
    else:
        summary["unsupervised_objective"] = None
    out_path = run_path / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

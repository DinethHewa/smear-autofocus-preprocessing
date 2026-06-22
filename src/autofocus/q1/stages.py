from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .aggregation import AUTOFOCUS_METRICS, DEFAULT_METRIC_WEIGHTS, compute_rank_based_summary, compute_value_based_summary
from .config import Q1RunContext, environment_metadata, ensure_stage_writable, source_tree_hash, stable_hash_file, stable_hash_mapping, write_stage_summary
from .exports import read_csv, read_json, write_csv, write_focus_curve_figure, write_heatmap_figure, write_json, write_latex_table, write_simple_bar_figure, write_text
from .gp_search import method_entry_from_pipeline, pipeline_from_pipeline_json, run_q1_gp_search
from .labels import build_manifest, build_reference_labels
from .metrics import metric_definitions
from .parameter_space import parameter_space_rows
from .workflows import configured_methods, evaluate_methods, method_by_name, select_best_method, summarize_methods


def _log(ctx: Q1RunContext, stage: str, message: str) -> None:
    path = ctx.logs_dir / f"{stage}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _manifest_path(ctx: Q1RunContext) -> Path:
    return ctx.output_dir / "reproducibility" / "manifest_used.csv"


def _labels_path(ctx: Q1RunContext) -> Path:
    return ctx.output_dir / "labels" / "reference_labels.csv"


def _load_manifest(ctx: Q1RunContext) -> pd.DataFrame:
    path = _manifest_path(ctx)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest. Run stage 01 first: {path}")
    return pd.read_csv(path)


def _load_labels(ctx: Q1RunContext) -> pd.DataFrame:
    path = _labels_path(ctx)
    if not path.exists():
        raise FileNotFoundError(f"Missing reference labels. Run stage 02 first: {path}")
    return pd.read_csv(path)


def _write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resume_existing(ctx: Q1RunContext, stage: str, out_path: Path, **payload: Any) -> Dict[str, Any] | None:
    if ctx.resume and out_path.exists() and not ctx.overwrite:
        resumed = {"resumed_from_existing": True, "primary_output": str(out_path), **payload}
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using existing output {out_path}")
        return resumed
    return None


CURVE_FIELDNAMES = [
    "dataset",
    "stack_id",
    "group_id",
    "stack_index",
    "outer_split",
    "inner_split",
    "method",
    "method_type",
    "pipeline_json",
    "focus_measure",
    "predicted_focus_index",
    "reference_focus_index",
    "label_source",
    "label_confidence",
    "runtime_sec_per_stack",
    "failure",
    "failure_reason",
    "raw_curve",
    "normalized_curve",
]

METRIC_FIELDNAMES = [
    "dataset",
    "stack_id",
    "group_id",
    "stack_index",
    "outer_split",
    "inner_split",
    "method",
    "method_type",
    "pipeline_json",
    "focus_measure",
    "predicted_focus_index",
    "reference_focus_index",
    "label_source",
    "label_confidence",
    "runtime_sec_per_stack",
    "failure",
    "failure_reason",
    *AUTOFOCUS_METRICS,
]

Q1_PROCEDURE_EXTRA_FIELDNAMES = [
    "heldout_dataset",
    "optimized_for_dataset",
    "evaluation_dataset",
    "selection_role",
    "source_method",
    "workflow_origin",
    "selection_endpoint",
    "train_datasets",
    "evaluation_scope",
]

Q1_PROCEDURE_METRIC_FIELDNAMES = [*METRIC_FIELDNAMES, *Q1_PROCEDURE_EXTRA_FIELDNAMES]
Q1_PROCEDURE_CURVE_FIELDNAMES = [*CURVE_FIELDNAMES, *Q1_PROCEDURE_EXTRA_FIELDNAMES]


def _stage_summary_path(ctx: Q1RunContext, stage: str) -> Path:
    return ctx.output_dir / "summaries" / f"{stage}.json"


def _stage_summary_payload(ctx: Q1RunContext, stage: str) -> Dict[str, Any]:
    summary_path = _stage_summary_path(ctx, stage)
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summary_is_complete(ctx: Q1RunContext, stage: str) -> bool:
    payload = _stage_summary_payload(ctx, stage)
    return str(payload.get("status")) in {"complete", "skipped_existing"}


def _is_true_gp_payload(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("optimizer_status", "")) == "genetic_programming"
        and str(payload.get("source", "")) == "new_gp_candidate_curves_and_q1_metrics"
    )


def _progress_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _csv_contains_true_gp_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method_type = str(row.get("method_type", ""))
                workflow_origin = str(row.get("workflow_origin", ""))
                source_method = str(row.get("source_method", ""))
                if "gp" in method_type and (
                    workflow_origin == "fresh_gp_final_evaluation"
                    or source_method in {"generalized_gp_workflow", "dataset_specific_gp_workflow"}
                ):
                    return True
    except Exception:
        return False
    return False


def _has_true_gp_resume_evidence(ctx: Q1RunContext, stage: str, out_path: Path, progress_path: Path | None = None) -> bool:
    if _is_true_gp_payload(_stage_summary_payload(ctx, stage)):
        return True
    if progress_path is not None and _is_true_gp_payload(_progress_payload(progress_path)):
        return True
    return _csv_contains_true_gp_rows(out_path)


def _has_fresh_transfer_resume_evidence(ctx: Q1RunContext, stage: str, progress_path: Path, per_stack_path: Path) -> bool:
    summary = _stage_summary_payload(ctx, stage)
    if str(summary.get("source", "")) == "fresh_workflow_evaluation":
        return True
    progress = _progress_payload(progress_path)
    if str(progress.get("source", "")) == "fresh_workflow_evaluation":
        return True
    return _csv_contains_true_gp_rows(per_stack_path)


def _read_completed_curve_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "method" not in (reader.fieldnames or []) or "stack_id" not in (reader.fieldnames or []):
            raise ValueError(f"Existing curve CSV is missing method/stack_id columns: {path}")
        for row in reader:
            method = str(row.get("method", ""))
            stack_id = str(row.get("stack_id", ""))
            if method and stack_id:
                completed.add((method, stack_id))
    return completed


def _read_completed_stack_method_pairs(path: Path, *, artifact_name: str) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "method" not in (reader.fieldnames or []) or "stack_id" not in (reader.fieldnames or []):
            raise ValueError(f"Existing {artifact_name} CSV is missing method/stack_id columns: {path}")
        for row in reader:
            method = str(row.get("method", ""))
            stack_id = str(row.get("stack_id", ""))
            if method and stack_id:
                completed.add((method, stack_id))
    return completed


def _append_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size <= 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, {**dict(payload), "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def _progress_message(stage: str, completed: int, total: int, *, method: str, dataset: str, start_time: datetime) -> str:
    elapsed = max(0.001, (datetime.now(timezone.utc) - start_time).total_seconds())
    rate = completed / elapsed
    remaining = max(0, total - completed)
    eta_sec = remaining / rate if rate > 0 else float("nan")
    eta_text = f"{eta_sec / 3600.0:.2f}h" if np.isfinite(eta_sec) else "unknown"
    pct = 100.0 * completed / total if total else 100.0
    return (
        f"{stage}: {completed}/{total} stack-method pairs "
        f"({pct:.2f}%), method={method}, dataset={dataset}, "
        f"rate={rate:.2f}/s, eta={eta_text}"
    )


def _dataset_progress_message(stage: str, completed: int, total: int, *, dataset: str, detail: str, start_time: datetime) -> str:
    elapsed = max(0.001, (datetime.now(timezone.utc) - start_time).total_seconds())
    rate = completed / elapsed
    remaining = max(0, total - completed)
    eta_sec = remaining / rate if rate > 0 else float("nan")
    eta_text = f"{eta_sec / 3600.0:.2f}h" if np.isfinite(eta_sec) else "unknown"
    pct = 100.0 * completed / total if total else 100.0
    suffix = f", {detail}" if detail else ""
    return f"{stage}: {completed}/{total} datasets ({pct:.2f}%), dataset={dataset}{suffix}, eta={eta_text}"


def _stage_step_message(stage: str, completed: int, total: int, *, step: str) -> str:
    pct = 100.0 * completed / total if total else 100.0
    return f"{stage}: {completed}/{total} steps ({pct:.2f}%), step={step}"


def _optimization_cfg(ctx: Q1RunContext, section: str) -> Dict[str, Any]:
    optimization = ctx.config.get("optimization", {}) if isinstance(ctx.config.get("optimization"), dict) else {}
    value = optimization.get(section, {}) if isinstance(optimization.get(section), dict) else {}
    return dict(value)


def _require_supported_q1_optimizer(ctx: Q1RunContext, section: str, stage: str) -> str:
    opt_cfg = _optimization_cfg(ctx, section)
    optimizer = str(opt_cfg.get("optimizer", "deterministic_candidate_search"))
    if optimizer in {"", "deterministic_candidate_search", "disabled_for_paper1"}:
        return optimizer
    allow_fallback = bool(opt_cfg.get("allow_deterministic_fallback", False))
    if allow_fallback:
        return "deterministic_candidate_search_fallback"
    if optimizer == "genetic_programming":
        return optimizer
    raise NotImplementedError(f"{stage}: unsupported optimizer for optimization.{section}: {optimizer}")


def stage01_build_stacks_and_manifest(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "01_build_stacks_and_manifest"
    out_path = _manifest_path(ctx)
    progress_path = ctx.output_dir / "reproducibility" / "manifest.progress.json"
    if ctx.resume and out_path.exists() and not ctx.overwrite:
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "complete",
                "run_id": ctx.run_id,
                "manifest_path": str(out_path),
                "resumed_from_existing": True,
                "progress_percent": 100.0,
            },
        )
        resumed = {"resumed_from_existing": True, "primary_output": str(out_path), "manifest_path": str(out_path), "progress_json": str(progress_path)}
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using existing output {out_path}")
        print(f"{stage}: complete existing manifest found; skipping manifest build: {out_path}", flush=True)
        return resumed
    ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "manifest_path": str(out_path),
            "progress_percent": 0.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(f"{stage}: building manifest; output={out_path}", flush=True)
    manifest = build_manifest(ctx.config, smoke=ctx.smoke)
    _write_df(out_path, manifest)
    config_out = ctx.output_dir / "reproducibility" / "config_used.yaml"
    if ctx.overwrite or not config_out.exists():
        shutil.copy2(ctx.config_path, config_out)
    dataset_summary = (
        manifest.groupby("dataset")
        .agg(
            stack_count=("stack_id", "count"),
            n_slices_min=("n_slices", "min"),
            n_slices_max=("n_slices", "max"),
            n_slices_median=("n_slices", "median"),
        )
        .reset_index()
    )
    _write_df(ctx.output_dir / "reproducibility" / "dataset_summary.csv", dataset_summary)
    payload = {
        "manifest_path": str(out_path),
        "progress_json": str(progress_path),
        "num_rows": int(len(manifest)),
        "datasets": sorted(manifest["dataset"].unique().tolist()),
    }
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "manifest_path": str(out_path),
            "num_rows": int(len(manifest)),
            "progress_percent": 100.0,
        },
    )
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote manifest: {out_path}")
    return payload


def stage02_build_reference_labels(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "02_build_reference_labels"
    out_path = _labels_path(ctx)
    progress_path = ctx.output_dir / "labels" / "reference_labels.progress.json"
    if ctx.resume and out_path.exists() and not ctx.overwrite:
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "complete",
                "run_id": ctx.run_id,
                "reference_labels": str(out_path),
                "resumed_from_existing": True,
                "progress_percent": 100.0,
            },
        )
        resumed = {"resumed_from_existing": True, "primary_output": str(out_path), "reference_labels": str(out_path), "progress_json": str(progress_path)}
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using existing output {out_path}")
        print(f"{stage}: complete existing labels found; skipping label build: {out_path}", flush=True)
        return resumed
    ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "reference_labels": str(out_path),
            "progress_percent": 0.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    manifest = _load_manifest(ctx)
    print(f"{stage}: building reference labels for {len(manifest)} stacks; output={out_path}", flush=True)
    labels, summary, loo = build_reference_labels(manifest, ctx.config)
    _write_df(out_path, labels)
    _write_df(ctx.output_dir / "labels" / "label_provenance_summary.csv", summary)
    if not loo.empty:
        _write_df(ctx.output_dir / "labels" / "leave_one_out_reference_labels.csv", loo)
    payload = {
        "reference_labels": str(out_path),
        "progress_json": str(progress_path),
        "label_sources": summary.to_dict(orient="records"),
        "loo_rows": int(len(loo)),
    }
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "reference_labels": str(out_path),
            "num_labels": int(len(labels)),
            "loo_rows": int(len(loo)),
            "progress_percent": 100.0,
        },
    )
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote labels: {out_path}")
    return payload


def stage03_run_preprocessed_focus_curves(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "03_run_preprocessed_focus_curves"
    out_path = ctx.output_dir / "q1_curves" / "preprocessed_focus_curves.csv"
    progress_path = ctx.output_dir / "q1_curves" / "preprocessed_focus_curves.progress.json"
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        resumed = {
            "resumed_from_existing": True,
            "primary_output": str(out_path),
            "curves_csv": str(out_path),
            "progress_json": str(progress_path),
        }
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using complete existing output {out_path}")
        return resumed
    if ctx.overwrite:
        for path in (out_path, progress_path):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    manifest = _load_manifest(ctx)
    labels = _load_labels(ctx)
    methods = configured_methods(ctx.config)
    output_cfg = ctx.config.get("outputs", {}) if isinstance(ctx.config.get("outputs"), dict) else {}
    progress_interval = int(output_cfg.get("progress_interval_stack_methods", 500) or 500)
    chunk_size = int(output_cfg.get("stage03_chunk_stack_count", 1) or 1)
    chunk_size = max(1, chunk_size)
    completed_pairs = _read_completed_curve_pairs(out_path) if ctx.resume and out_path.exists() else set()
    total_pairs = int(len(methods) * len(manifest))
    completed_count = len(completed_pairs)
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "curves_csv": str(out_path),
            "total_stack_method_pairs": total_pairs,
            "completed_stack_method_pairs": completed_count,
            "remaining_stack_method_pairs": max(0, total_pairs - completed_count),
            "progress_percent": 100.0 * completed_count / total_pairs if total_pairs else 100.0,
            "resume_mode": bool(ctx.resume),
            "chunk_stack_count": chunk_size,
        },
    )
    _log(
        ctx,
        stage,
        f"started curve generation: total_pairs={total_pairs}, already_completed={completed_count}, output={out_path}",
    )
    print(
        f"{stage}: starting/resuming with {completed_count}/{total_pairs} completed stack-method pairs; output={out_path}",
        flush=True,
    )

    for method_entry in methods:
        method_name = str(method_entry["method"])
        pending_mask = ~manifest["stack_id"].astype(str).map(lambda stack_id: (method_name, stack_id) in completed_pairs)
        pending = manifest[pending_mask].copy()
        if pending.empty:
            continue
        for start in range(0, len(pending), chunk_size):
            chunk = pending.iloc[start : start + chunk_size].copy()
            _, _, curve_rows = evaluate_methods(
                manifest_df=chunk,
                labels_df=labels,
                methods=[method_entry],
                config=ctx.config,
                write_curves=True,
                compute_metrics=False,
            )
            _append_csv_rows(out_path, curve_rows, CURVE_FIELDNAMES)
            for row in curve_rows:
                completed_pairs.add((str(row["method"]), str(row["stack_id"])))
            completed_count = len(completed_pairs)
            current_dataset = str(chunk.iloc[-1].get("dataset", ""))
            should_report = (
                completed_count == total_pairs
                or completed_count % progress_interval == 0
                or progress_interval <= 1
            )
            if should_report:
                message = _progress_message(
                    stage,
                    completed_count,
                    total_pairs,
                    method=method_name,
                    dataset=current_dataset,
                    start_time=start_time,
                )
                _log(ctx, stage, message)
                print(message, flush=True)
                _write_progress(
                    progress_path,
                    {
                        "stage": stage,
                        "status": "running",
                        "run_id": ctx.run_id,
                        "curves_csv": str(out_path),
                        "total_stack_method_pairs": total_pairs,
                        "completed_stack_method_pairs": completed_count,
                        "remaining_stack_method_pairs": max(0, total_pairs - completed_count),
                        "progress_percent": 100.0 * completed_count / total_pairs if total_pairs else 100.0,
                        "current_method": method_name,
                        "current_dataset": current_dataset,
                        "chunk_stack_count": chunk_size,
                    },
                )
    final_count = len(_read_completed_curve_pairs(out_path))
    if final_count < total_pairs:
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "incomplete",
                "run_id": ctx.run_id,
                "curves_csv": str(out_path),
                "total_stack_method_pairs": total_pairs,
                "completed_stack_method_pairs": final_count,
                "remaining_stack_method_pairs": total_pairs - final_count,
                "progress_percent": 100.0 * final_count / total_pairs if total_pairs else 100.0,
            },
        )
        raise RuntimeError(
            f"Stage 03 incomplete: wrote {final_count}/{total_pairs} stack-method curve rows. "
            f"Resume with --resume after checking {progress_path}."
        )
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "curves_csv": str(out_path),
            "total_stack_method_pairs": total_pairs,
            "completed_stack_method_pairs": final_count,
            "remaining_stack_method_pairs": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "curves_csv": str(out_path),
        "progress_json": str(progress_path),
        "num_curve_rows": final_count,
        "num_methods": len(methods),
        "total_stack_method_pairs": total_pairs,
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote curves: {out_path}; rows={final_count}")
    return payload


def stage04_evaluate_workflows_q1_metrics(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "04_evaluate_workflows_q1_metrics"
    metrics_dir = ctx.output_dir / "q1_metrics"
    out_path = metrics_dir / "per_stack_metrics.csv"
    progress_path = metrics_dir / "per_stack_metrics.progress.json"
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        resumed = {
            "resumed_from_existing": True,
            "primary_output": str(out_path),
            "per_stack_metrics": str(out_path),
            "progress_json": str(progress_path),
        }
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using complete existing output {out_path}")
        print(f"{stage}: complete existing output found; skipping Stage 04 metrics: {out_path}", flush=True)
        return resumed
    if ctx.overwrite:
        for path in (
            out_path,
            progress_path,
            metrics_dir / "dataset_metric_summary.csv",
            metrics_dir / "method_rank_summary.csv",
            metrics_dir / "method_value_summary.csv",
            metrics_dir / "method_summary.csv",
            metrics_dir / "rank_cells.json",
            metrics_dir / "metric_definitions.csv",
        ):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    manifest = _load_manifest(ctx)
    labels = _load_labels(ctx)
    methods = configured_methods(ctx.config)
    output_cfg = ctx.config.get("outputs", {}) if isinstance(ctx.config.get("outputs"), dict) else {}
    progress_interval = int(output_cfg.get("progress_interval_stack_methods", 500) or 500)
    chunk_size = int(output_cfg.get("stage04_chunk_stack_count", output_cfg.get("stage03_chunk_stack_count", 1)) or 1)
    chunk_size = max(1, chunk_size)
    completed_pairs = _read_completed_stack_method_pairs(out_path, artifact_name="metric") if ctx.resume and out_path.exists() else set()
    total_pairs = int(len(methods) * len(manifest))
    completed_count = len(completed_pairs)
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "per_stack_metrics": str(out_path),
            "total_stack_method_pairs": total_pairs,
            "completed_stack_method_pairs": completed_count,
            "remaining_stack_method_pairs": max(0, total_pairs - completed_count),
            "progress_percent": 100.0 * completed_count / total_pairs if total_pairs else 100.0,
            "resume_mode": bool(ctx.resume),
            "chunk_stack_count": chunk_size,
        },
    )
    _log(
        ctx,
        stage,
        f"started Q1 metric evaluation: total_pairs={total_pairs}, already_completed={completed_count}, output={out_path}",
    )
    print(
        f"{stage}: starting/resuming with {completed_count}/{total_pairs} completed stack-method pairs; output={out_path}",
        flush=True,
    )

    for method_entry in methods:
        method_name = str(method_entry["method"])
        pending_mask = ~manifest["stack_id"].astype(str).map(lambda stack_id: (method_name, stack_id) in completed_pairs)
        pending = manifest[pending_mask].copy()
        if pending.empty:
            continue
        for start in range(0, len(pending), chunk_size):
            chunk = pending.iloc[start : start + chunk_size].copy()
            per_stack_rows, _, _ = evaluate_methods(
                manifest_df=chunk,
                labels_df=labels,
                methods=[method_entry],
                config=ctx.config,
                write_curves=False,
                compute_metrics=True,
            )
            _append_csv_rows(out_path, per_stack_rows, METRIC_FIELDNAMES)
            for row in per_stack_rows:
                completed_pairs.add((str(row["method"]), str(row["stack_id"])))
            completed_count = len(completed_pairs)
            current_dataset = str(chunk.iloc[-1].get("dataset", ""))
            should_report = (
                completed_count == total_pairs
                or completed_count % progress_interval == 0
                or progress_interval <= 1
            )
            if should_report:
                message = _progress_message(
                    stage,
                    completed_count,
                    total_pairs,
                    method=method_name,
                    dataset=current_dataset,
                    start_time=start_time,
                )
                _log(ctx, stage, message)
                print(message, flush=True)
                _write_progress(
                    progress_path,
                    {
                        "stage": stage,
                        "status": "running",
                        "run_id": ctx.run_id,
                        "per_stack_metrics": str(out_path),
                        "total_stack_method_pairs": total_pairs,
                        "completed_stack_method_pairs": completed_count,
                        "remaining_stack_method_pairs": max(0, total_pairs - completed_count),
                        "progress_percent": 100.0 * completed_count / total_pairs if total_pairs else 100.0,
                        "current_method": method_name,
                        "current_dataset": current_dataset,
                        "chunk_stack_count": chunk_size,
                    },
                )

    final_pairs = _read_completed_stack_method_pairs(out_path, artifact_name="metric")
    final_count = len(final_pairs)
    if final_count < total_pairs:
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "incomplete",
                "run_id": ctx.run_id,
                "per_stack_metrics": str(out_path),
                "total_stack_method_pairs": total_pairs,
                "completed_stack_method_pairs": final_count,
                "remaining_stack_method_pairs": total_pairs - final_count,
                "progress_percent": 100.0 * final_count / total_pairs if total_pairs else 100.0,
            },
        )
        raise RuntimeError(
            f"Stage 04 incomplete: wrote {final_count}/{total_pairs} stack-method metric rows. "
            f"Resume with --resume after checking {progress_path}."
        )

    per_stack = read_csv(out_path)
    dataset_summary, rank_rows, value_rows, method_payload = summarize_methods(per_stack, ctx.config)
    write_csv(metrics_dir / "dataset_metric_summary.csv", dataset_summary)
    write_csv(metrics_dir / "method_rank_summary.csv", rank_rows)
    write_csv(metrics_dir / "method_value_summary.csv", value_rows)
    write_csv(metrics_dir / "method_summary.csv", method_payload["method_summary"])
    write_json(metrics_dir / "rank_cells.json", method_payload["rank_cells"])
    write_csv(metrics_dir / "metric_definitions.csv", metric_definitions())
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "per_stack_metrics": str(out_path),
            "total_stack_method_pairs": total_pairs,
            "completed_stack_method_pairs": final_count,
            "remaining_stack_method_pairs": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "per_stack_metrics": str(out_path),
        "progress_json": str(progress_path),
        "dataset_metric_summary": str(metrics_dir / "dataset_metric_summary.csv"),
        "method_summary": str(metrics_dir / "method_summary.csv"),
        "num_rows": final_count,
        "total_stack_method_pairs": total_pairs,
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote q1 metrics: {out_path}; rows={final_count}")
    return payload


def _operator_frequency(method_names: Sequence[str]) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for name in method_names:
        if "::" in name:
            counts[name.split("::", 1)[1]] += 1
        else:
            counts[name] += 1
    return [{"operator_or_workflow": key, "count": value} for key, value in sorted(counts.items())]


def _base_seed(ctx: Q1RunContext) -> int:
    return int(ctx.config.get("seed", ctx.config.get("random_seed", 123)))


def _method_names(methods: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(method["method"]) for method in methods}


def _alias_method_entry(method_entry: Mapping[str, Any], alias: str, *, method_type: str | None = None) -> Dict[str, Any]:
    out = dict(method_entry)
    out["source_method"] = str(method_entry.get("method", alias))
    out["method"] = alias
    if method_type is not None:
        out["method_type"] = method_type
    return out


def _pipeline_operator_names_from_json(pipeline_json: str) -> List[str]:
    if not pipeline_json:
        return []
    try:
        pipeline = pipeline_from_pipeline_json(pipeline_json)
    except Exception:
        return []
    return [str(step.name) for step in pipeline.steps if getattr(step, "enabled", True)]


def _add_procedure_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_by_alias: Mapping[str, str],
    extras: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        enriched = dict(row)
        enriched.update(extras)
        enriched["source_method"] = source_by_alias.get(method, enriched.get("source_method", method))
        out.append(enriched)
    return out


def _fresh_evaluate_procedures(
    *,
    ctx: Q1RunContext,
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    methods: Sequence[Mapping[str, Any]],
    datasets: Sequence[str] | None,
    outer_split: Sequence[str] | None,
    inner_split: Sequence[str] | None,
    extras_by_method: Mapping[str, Mapping[str, Any]],
    default_extras: Mapping[str, Any],
    write_curves: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_stack, _, curves = evaluate_methods(
        manifest_df=manifest,
        labels_df=labels,
        methods=methods,
        config=ctx.config,
        datasets=datasets,
        outer_split=outer_split,
        inner_split=inner_split,
        write_curves=write_curves,
        compute_metrics=True,
    )
    source_by_alias = {str(method["method"]): str(method.get("source_method", method["method"])) for method in methods}
    metric_rows: List[Dict[str, Any]] = []
    for row in per_stack:
        method = str(row.get("method", ""))
        extras = {**dict(default_extras), **dict(extras_by_method.get(method, {}))}
        metric_rows.extend(_add_procedure_fields([row], source_by_alias=source_by_alias, extras=extras))
    curve_rows: List[Dict[str, Any]] = []
    for row in curves:
        method = str(row.get("method", ""))
        extras = {**dict(default_extras), **dict(extras_by_method.get(method, {}))}
        curve_rows.extend(_add_procedure_fields([row], source_by_alias=source_by_alias, extras=extras))
    return metric_rows, curve_rows


def _select_single_operator_baselines(
    *,
    ctx: Q1RunContext,
    all_methods: Sequence[Mapping[str, Any]],
    all_metric_rows: Sequence[Mapping[str, Any]],
    eval_dataset: str,
    train_datasets: Sequence[str],
) -> tuple[str, str]:
    single_method_names = [str(method.get("method", "")) for method in all_methods if str(method.get("method", "")).startswith("single::")]
    if not single_method_names:
        return "", ""
    fixed_rows = [
        dict(row)
        for row in all_metric_rows
        if str(row.get("dataset")) in set(train_datasets)
        and str(row.get("method")) in set(single_method_names)
        and str(row.get("outer_split")) == "trainval"
        and str(row.get("inner_split")) in {"train", "val"}
    ]
    best_fixed = ""
    if fixed_rows:
        _, _, _, fixed_summary = summarize_methods(fixed_rows, ctx.config)
        best_fixed = select_best_method(fixed_summary["method_summary"], exclude_oracle=False)
    oracle_rows = [
        dict(row)
        for row in all_metric_rows
        if str(row.get("dataset")) == eval_dataset and str(row.get("method")) in set(single_method_names)
    ]
    oracle = ""
    if oracle_rows:
        _, _, _, oracle_summary = summarize_methods(oracle_rows, ctx.config)
        oracle = select_best_method(oracle_summary["method_summary"], exclude_oracle=False)
    return best_fixed, oracle


def _comparison_method_entries(
    *,
    all_methods: Sequence[Mapping[str, Any]],
    best_fixed_single: str,
    oracle_single: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Mapping[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    extras: Dict[str, Mapping[str, Any]] = {}
    names = _method_names(all_methods)
    if "identity" in names:
        entries.append(_alias_method_entry(method_by_name(all_methods, "identity"), "identity", method_type="baseline"))
        extras["identity"] = {"selection_role": "identity_baseline", "workflow_origin": "fresh_baseline_evaluation"}
    if "conventional_clahe_unsharp" in names:
        entries.append(
            _alias_method_entry(
                method_by_name(all_methods, "conventional_clahe_unsharp"),
                "conventional_clahe_unsharp",
                method_type="conventional_baseline",
            )
        )
        extras["conventional_clahe_unsharp"] = {"selection_role": "conventional_baseline", "workflow_origin": "fresh_baseline_evaluation"}
    if best_fixed_single:
        entries.append(_alias_method_entry(method_by_name(all_methods, best_fixed_single), "best_fixed_single", method_type="single_operator_strategy"))
        extras["best_fixed_single"] = {"selection_role": "best_fixed_single", "workflow_origin": "fresh_single_operator_evaluation"}
    if oracle_single:
        entries.append(
            _alias_method_entry(
                method_by_name(all_methods, oracle_single),
                "oracle_single_exploratory",
                method_type="oracle_single_operator_exploratory",
            )
        )
        extras["oracle_single_exploratory"] = {
            "selection_role": "dataset_oracle_single_exploratory",
            "workflow_origin": "fresh_oracle_evaluation",
        }
    return entries, extras


def _method_entry_from_workflow_record(
    record: Mapping[str, Any],
    *,
    alias: str,
    method_type: str,
    all_methods: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pipeline_json = str(record.get("pipeline_json", ""))
    if pipeline_json:
        return method_entry_from_pipeline(alias, pipeline_from_pipeline_json(pipeline_json), method_type=method_type)
    selected_method = str(record.get("selected_method", ""))
    if not selected_method:
        raise ValueError(f"Workflow record for alias={alias} has neither pipeline_json nor selected_method")
    return _alias_method_entry(method_by_name(all_methods, selected_method), alias, method_type=method_type)


def _assert_gp_resume_compatible(ctx: Q1RunContext, stage: str, out_path: Path, progress_path: Path | None = None) -> None:
    if not (ctx.resume and out_path.exists() and not ctx.overwrite):
        return
    if _has_true_gp_resume_evidence(ctx, stage, out_path, progress_path):
        return
    raise RuntimeError(
        f"{stage}: existing output is not a true GP output and cannot be resumed for optimizer=genetic_programming: {out_path}. "
        "Use a new --run-id for the GP experiment, or rerun this stage with --overwrite only if you intend to replace the old exploratory output."
    )


def _stage05_generalized_gp(
    ctx: Q1RunContext,
    *,
    stage: str,
    out_dir: Path,
    out_path: Path,
    progress_path: Path,
    best_workflows_path: Path,
    optimizer_status: str,
    manifest: pd.DataFrame,
    all_methods: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
) -> Dict[str, Any]:
    _assert_gp_resume_compatible(ctx, stage, out_path, progress_path)
    labels = _load_labels(ctx)
    opt_cfg = _optimization_cfg(ctx, "generalized")
    metrics_path = ctx.output_dir / "q1_metrics" / "per_stack_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 04 metrics for selecting comparison baselines: {metrics_path}")
    all_metric_rows = read_csv(metrics_path)
    curves_path = out_dir / "lodo_selected_focus_curves.csv"
    existing_rows = read_csv(out_path) if out_path.exists() and ctx.resume else []
    completed_heldouts = {str(row.get("heldout_dataset", "")) for row in existing_rows if row.get("heldout_dataset")}
    best_workflows: List[Dict[str, Any]] = []
    if best_workflows_path.exists() and ctx.resume:
        loaded = read_json(best_workflows_path)
        if isinstance(loaded, list):
            best_workflows = [dict(row) for row in loaded]
    selected_operator_names: List[str] = []
    for row in best_workflows:
        selected_operator_names.extend(_pipeline_operator_names_from_json(str(row.get("pipeline_json", ""))))
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "optimizer_status": optimizer_status,
            "source": "new_gp_candidate_curves_and_q1_metrics",
            "lodo_results_per_stack": str(out_path),
            "gp_search_root": str(out_dir / "gp_search"),
            "total_heldout_datasets": len(datasets),
            "completed_heldout_datasets": len(completed_heldouts),
            "remaining_heldout_datasets": max(0, len(datasets) - len(completed_heldouts)),
            "progress_percent": 100.0 * len(completed_heldouts) / len(datasets) if datasets else 100.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(
        f"{stage}: starting/resuming true GP with {len(completed_heldouts)}/{len(datasets)} held-out datasets complete; "
        f"candidate curves/metrics are computed fresh",
        flush=True,
    )
    _log(ctx, stage, f"started true GP LODO: datasets={datasets}, completed={sorted(completed_heldouts)}")

    for heldout_index, heldout in enumerate(datasets):
        if heldout in completed_heldouts:
            print(f"{stage}: skipping completed heldout_dataset={heldout}", flush=True)
            continue
        train_datasets = [dataset for dataset in datasets if dataset != heldout] or [heldout]
        search_dir = out_dir / "gp_search" / heldout

        def gp_progress(event: Mapping[str, Any]) -> None:
            event_type = str(event.get("event", ""))
            if event_type in {"candidate_started", "candidate_chunk"}:
                message = (
                    f"{stage}: GP heldout={heldout} generation={event.get('generation')}/{event.get('generations')} "
                    f"candidate={event.get('candidate_index')}/{event.get('population_size')} "
                    f"stacks={event.get('completed_stack_method_pairs')}/{event.get('total_stack_method_pairs')} "
                    f"({float(event.get('progress_percent', 0.0)):.2f}%), "
                    f"dataset={event.get('current_dataset', '')}, checkpoint={event.get('candidate_metric_csv', '')}"
                )
            elif event_type == "candidate":
                message = (
                    f"{stage}: GP heldout={heldout} generation={event.get('generation')}/{event.get('generations')} "
                    f"candidate={event.get('candidate_index')}/{event.get('population_size')} "
                    f"evaluated={event.get('evaluated')} cache_hits={event.get('cache_hits')} "
                    f"cached={event.get('cached')}"
                )
            else:
                message = (
                    f"{stage}: GP heldout={heldout} generation={event.get('generation')}/{event.get('generations')} "
                    f"best_score={float(event.get('best_score', float('nan'))):.6g} "
                    f"evaluated={event.get('evaluated')} cache_hits={event.get('cache_hits')}"
                )
            _log(ctx, stage, message)
            print(message, flush=True)
            _write_progress(
                progress_path,
                {
                    "stage": stage,
                    "status": "running",
                    "run_id": ctx.run_id,
                    "optimizer_status": optimizer_status,
                    "source": "new_gp_candidate_curves_and_q1_metrics",
                    "current_heldout_dataset": heldout,
                    "current_gp_event": dict(event),
                    "total_heldout_datasets": len(datasets),
                    "completed_heldout_datasets": len(completed_heldouts),
                    "remaining_heldout_datasets": max(0, len(datasets) - len(completed_heldouts)),
                    "progress_percent": 100.0 * len(completed_heldouts) / len(datasets) if datasets else 100.0,
                },
            )

        result = run_q1_gp_search(
            manifest_df=manifest,
            labels_df=labels,
            config=ctx.config,
            opt_cfg=opt_cfg,
            search_dir=search_dir,
            search_name=f"generalized_lodo_{heldout}",
            seed=_base_seed(ctx) + 1000 + heldout_index,
            datasets=train_datasets,
            outer_split=["trainval"],
            inner_split=["train", "val"],
            resume=ctx.resume,
            progress_cb=gp_progress,
        )
        gp_method = method_entry_from_pipeline("generalized_gp_workflow", result.best_pipeline, method_type="q1_gp_generalized_lodo")
        best_fixed_single, oracle_single = _select_single_operator_baselines(
            ctx=ctx,
            all_methods=all_methods,
            all_metric_rows=all_metric_rows,
            eval_dataset=heldout,
            train_datasets=train_datasets,
        )
        comparison_entries, comparison_extras = _comparison_method_entries(
            all_methods=all_methods,
            best_fixed_single=best_fixed_single,
            oracle_single=oracle_single,
        )
        methods_to_evaluate = [gp_method, *comparison_entries]
        extras_by_method: Dict[str, Mapping[str, Any]] = {
            "generalized_gp_workflow": {
                "selection_role": "generalized_gp_selected",
                "workflow_origin": "fresh_gp_final_evaluation",
            },
            **comparison_extras,
        }
        metric_rows, curve_rows = _fresh_evaluate_procedures(
            ctx=ctx,
            manifest=manifest,
            labels=labels,
            methods=methods_to_evaluate,
            datasets=[heldout],
            outer_split=None,
            inner_split=None,
            extras_by_method=extras_by_method,
            default_extras={
                "heldout_dataset": heldout,
                "evaluation_dataset": heldout,
                "selection_endpoint": "q1_gp_rank_value_score_on_training_datasets",
                "train_datasets": json.dumps(train_datasets),
                "evaluation_scope": "lodo_heldout_dataset_all_splits",
            },
            write_curves=True,
        )
        _append_csv_rows(out_path, metric_rows, Q1_PROCEDURE_METRIC_FIELDNAMES)
        _append_csv_rows(curves_path, curve_rows, Q1_PROCEDURE_CURVE_FIELDNAMES)
        selected_operator_names.extend(_pipeline_operator_names_from_json(result.best_pipeline_json))
        best_workflows = [row for row in best_workflows if str(row.get("heldout_dataset", "")) != heldout]
        best_workflows.append(
            {
                "heldout_dataset": heldout,
                "train_datasets": train_datasets,
                "selected_method": "generalized_gp_workflow",
                "best_fixed_single_on_training": best_fixed_single,
                "dataset_oracle_single_on_heldout": oracle_single,
                "selection_endpoint": "q1_gp_rank_value_score_on_training_datasets",
                "pipeline_json": result.best_pipeline_json,
                "best_score": result.best_score,
                "best_genome": result.best_genome,
                "gp_progress_csv": str(result.progress_csv),
                "gp_best_by_generation_csv": str(result.best_by_generation_csv),
                "gp_candidate_dataset_metrics_csv": str(result.candidate_summary_csv),
                "gp_state_json": str(result.state_json),
                "optimizer_status": optimizer_status,
                "source": "new_gp_candidate_curves_and_q1_metrics",
            }
        )
        write_json(best_workflows_path, best_workflows)
        completed_heldouts.add(heldout)
        message = _dataset_progress_message(
            stage,
            len(completed_heldouts),
            len(datasets),
            dataset=heldout,
            detail=f"selected_method=generalized_gp_workflow, best_score={result.best_score:.6g}",
            start_time=start_time,
        )
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running",
                "run_id": ctx.run_id,
                "optimizer_status": optimizer_status,
                "source": "new_gp_candidate_curves_and_q1_metrics",
                "lodo_results_per_stack": str(out_path),
                "selected_curves_csv": str(curves_path),
                "total_heldout_datasets": len(datasets),
                "completed_heldout_datasets": len(completed_heldouts),
                "remaining_heldout_datasets": max(0, len(datasets) - len(completed_heldouts)),
                "progress_percent": 100.0 * len(completed_heldouts) / len(datasets) if datasets else 100.0,
                "current_heldout_dataset": heldout,
                "selected_method": "generalized_gp_workflow",
                "best_score": result.best_score,
            },
        )
    if len(completed_heldouts) < len(datasets):
        raise RuntimeError(
            f"Stage 05 GP incomplete: completed {len(completed_heldouts)}/{len(datasets)} held-out datasets. "
            f"Resume with --resume after checking {progress_path}."
        )
    per_stack_all = read_csv(out_path)
    dataset_summary, rank_rows, value_rows, method_payload = summarize_methods(per_stack_all, ctx.config)
    write_csv(out_dir / "lodo_results_per_dataset.csv", dataset_summary)
    write_csv(out_dir / "method_rank_summary.csv", rank_rows)
    write_csv(out_dir / "method_value_summary.csv", value_rows)
    write_json(best_workflows_path, best_workflows)
    write_csv(out_dir / "operator_frequency.csv", _operator_frequency(selected_operator_names))
    write_csv(out_dir / "complexity_runtime_summary.csv", method_payload["method_summary"])
    write_csv(out_dir / "failure_summary.csv", [{"method": row["method"], "failure_count": row["failure_count"], "failure_rate": row["failure_rate"]} for row in method_payload["method_summary"]])
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "optimizer_status": optimizer_status,
            "source": "new_gp_candidate_curves_and_q1_metrics",
            "lodo_results_per_stack": str(out_path),
            "selected_curves_csv": str(curves_path),
            "total_heldout_datasets": len(datasets),
            "completed_heldout_datasets": len(completed_heldouts),
            "remaining_heldout_datasets": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "paper1_dir": str(out_dir),
        "progress_json": str(progress_path),
        "selected_curves_csv": str(curves_path),
        "num_lodo_rows": len(per_stack_all),
        "optimizer_status": optimizer_status,
        "source": "new_gp_candidate_curves_and_q1_metrics",
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote true GP paper1 LODO outputs: {out_dir}")
    return payload


def _stage06_dataset_specific_gp(
    ctx: Q1RunContext,
    *,
    stage: str,
    out_dir: Path,
    out_path: Path,
    progress_path: Path,
    best_workflows_path: Path,
    optimizer_status: str,
    manifest: pd.DataFrame,
    datasets: Sequence[str],
) -> Dict[str, Any]:
    _assert_gp_resume_compatible(ctx, stage, out_path, progress_path)
    labels = _load_labels(ctx)
    opt_cfg = _optimization_cfg(ctx, "dataset_specific")
    curves_path = out_dir / "dataset_specific_selected_focus_curves.csv"
    existing_rows = read_csv(out_path) if out_path.exists() and ctx.resume else []
    completed_datasets = {str(row.get("optimized_for_dataset", "")) for row in existing_rows if row.get("optimized_for_dataset")}
    best_workflows: List[Dict[str, Any]] = []
    if best_workflows_path.exists() and ctx.resume:
        loaded = read_json(best_workflows_path)
        if isinstance(loaded, list):
            best_workflows = [dict(row) for row in loaded]
    selected_operator_names: List[str] = []
    for row in best_workflows:
        selected_operator_names.extend(_pipeline_operator_names_from_json(str(row.get("pipeline_json", ""))))
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "optimizer_status": optimizer_status,
            "source": "new_gp_candidate_curves_and_q1_metrics",
            "dataset_specific_results_per_stack": str(out_path),
            "gp_search_root": str(out_dir / "gp_search"),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
            "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(
        f"{stage}: starting/resuming true dataset-specific GP with {len(completed_datasets)}/{len(datasets)} datasets complete; "
        f"candidate curves/metrics are computed fresh",
        flush=True,
    )
    _log(ctx, stage, f"started true dataset-specific GP: datasets={datasets}, completed={sorted(completed_datasets)}")

    for dataset_index, dataset in enumerate(datasets):
        if dataset in completed_datasets:
            print(f"{stage}: skipping completed dataset={dataset}", flush=True)
            continue
        search_dir = out_dir / "gp_search" / dataset

        def gp_progress(event: Mapping[str, Any]) -> None:
            event_type = str(event.get("event", ""))
            if event_type in {"candidate_started", "candidate_chunk"}:
                message = (
                    f"{stage}: GP dataset={dataset} generation={event.get('generation')}/{event.get('generations')} "
                    f"candidate={event.get('candidate_index')}/{event.get('population_size')} "
                    f"stacks={event.get('completed_stack_method_pairs')}/{event.get('total_stack_method_pairs')} "
                    f"({float(event.get('progress_percent', 0.0)):.2f}%), "
                    f"dataset={event.get('current_dataset', '')}, checkpoint={event.get('candidate_metric_csv', '')}"
                )
            elif event_type == "candidate":
                message = (
                    f"{stage}: GP dataset={dataset} generation={event.get('generation')}/{event.get('generations')} "
                    f"candidate={event.get('candidate_index')}/{event.get('population_size')} "
                    f"evaluated={event.get('evaluated')} cache_hits={event.get('cache_hits')} "
                    f"cached={event.get('cached')}"
                )
            else:
                message = (
                    f"{stage}: GP dataset={dataset} generation={event.get('generation')}/{event.get('generations')} "
                    f"best_score={float(event.get('best_score', float('nan'))):.6g} "
                    f"evaluated={event.get('evaluated')} cache_hits={event.get('cache_hits')}"
                )
            _log(ctx, stage, message)
            print(message, flush=True)
            _write_progress(
                progress_path,
                {
                    "stage": stage,
                    "status": "running",
                    "run_id": ctx.run_id,
                    "optimizer_status": optimizer_status,
                    "source": "new_gp_candidate_curves_and_q1_metrics",
                    "current_dataset": dataset,
                    "current_gp_event": dict(event),
                    "total_datasets": len(datasets),
                    "completed_datasets": len(completed_datasets),
                    "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
                    "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
                },
            )

        result = run_q1_gp_search(
            manifest_df=manifest,
            labels_df=labels,
            config=ctx.config,
            opt_cfg=opt_cfg,
            search_dir=search_dir,
            search_name=f"dataset_specific_{dataset}",
            seed=_base_seed(ctx) + 2000 + dataset_index,
            datasets=[dataset],
            outer_split=["trainval"],
            inner_split=["train", "val"],
            resume=ctx.resume,
            progress_cb=gp_progress,
        )
        selected_method_name = f"{dataset}_specific_gp_workflow"
        gp_method = method_entry_from_pipeline(selected_method_name, result.best_pipeline, method_type="q1_gp_dataset_specific")
        dataset_outer_splits = set(manifest[manifest["dataset"] == dataset]["outer_split"].astype(str))
        use_outer_test = bool(opt_cfg.get("use_outer_test_when_available", True))
        test_outer = ["test"] if use_outer_test and "test" in dataset_outer_splits else None
        evaluation_scope = "dataset_outer_test_split" if test_outer else "dataset_all_splits_no_outer_test_available"
        metric_rows, curve_rows = _fresh_evaluate_procedures(
            ctx=ctx,
            manifest=manifest,
            labels=labels,
            methods=[gp_method],
            datasets=[dataset],
            outer_split=test_outer,
            inner_split=None,
            extras_by_method={
                selected_method_name: {
                    "selection_role": "dataset_specific_gp_selected",
                    "workflow_origin": "fresh_gp_final_evaluation",
                }
            },
            default_extras={
                "optimized_for_dataset": dataset,
                "evaluation_dataset": dataset,
                "selection_endpoint": "q1_gp_rank_value_score_on_dataset_trainval",
                "train_datasets": json.dumps([dataset]),
                "evaluation_scope": evaluation_scope,
            },
            write_curves=True,
        )
        _append_csv_rows(out_path, metric_rows, Q1_PROCEDURE_METRIC_FIELDNAMES)
        _append_csv_rows(curves_path, curve_rows, Q1_PROCEDURE_CURVE_FIELDNAMES)
        selected_operator_names.extend(_pipeline_operator_names_from_json(result.best_pipeline_json))
        best_workflows = [row for row in best_workflows if str(row.get("dataset", "")) != dataset]
        best_workflows.append(
            {
                "dataset": dataset,
                "selected_method": selected_method_name,
                "selection_endpoint": "q1_gp_rank_value_score_on_dataset_trainval",
                "evaluation_scope": evaluation_scope,
                "pipeline_json": result.best_pipeline_json,
                "best_score": result.best_score,
                "best_genome": result.best_genome,
                "gp_progress_csv": str(result.progress_csv),
                "gp_best_by_generation_csv": str(result.best_by_generation_csv),
                "gp_candidate_dataset_metrics_csv": str(result.candidate_summary_csv),
                "gp_state_json": str(result.state_json),
                "optimizer_status": optimizer_status,
                "source": "new_gp_candidate_curves_and_q1_metrics",
            }
        )
        write_json(best_workflows_path, best_workflows)
        completed_datasets.add(dataset)
        message = _dataset_progress_message(
            stage,
            len(completed_datasets),
            len(datasets),
            dataset=dataset,
            detail=f"selected_method={selected_method_name}, best_score={result.best_score:.6g}",
            start_time=start_time,
        )
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running",
                "run_id": ctx.run_id,
                "optimizer_status": optimizer_status,
                "source": "new_gp_candidate_curves_and_q1_metrics",
                "dataset_specific_results_per_stack": str(out_path),
                "selected_curves_csv": str(curves_path),
                "total_datasets": len(datasets),
                "completed_datasets": len(completed_datasets),
                "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
                "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
                "current_dataset": dataset,
                "selected_method": selected_method_name,
                "best_score": result.best_score,
            },
        )
    if len(completed_datasets) < len(datasets):
        raise RuntimeError(
            f"Stage 06 GP incomplete: completed {len(completed_datasets)}/{len(datasets)} datasets. "
            f"Resume with --resume after checking {progress_path}."
        )
    per_stack_all = read_csv(out_path)
    dataset_summary, _, _, _ = summarize_methods(per_stack_all, ctx.config)
    write_json(best_workflows_path, best_workflows)
    write_csv(out_dir / "dataset_specific_results_per_dataset.csv", dataset_summary)
    write_csv(out_dir / "operator_frequency_by_dataset.csv", _operator_frequency(selected_operator_names))
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "optimizer_status": optimizer_status,
            "source": "new_gp_candidate_curves_and_q1_metrics",
            "dataset_specific_results_per_stack": str(out_path),
            "selected_curves_csv": str(curves_path),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "paper2_dir": str(out_dir),
        "progress_json": str(progress_path),
        "selected_curves_csv": str(curves_path),
        "num_dataset_specific_rows": len(per_stack_all),
        "optimizer_status": optimizer_status,
        "source": "new_gp_candidate_curves_and_q1_metrics",
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote true dataset-specific GP outputs: {out_dir}")
    return payload


def stage05_generalized_workflow_search_lodo(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "05_generalized_workflow_search_lodo"
    out_dir = ctx.output_dir / "paper1_generalized"
    out_path = out_dir / "lodo_results_per_stack.csv"
    progress_path = out_dir / "lodo_results.progress.json"
    best_workflows_path = out_dir / "generalized_best_workflows.json"
    optimizer_status = _require_supported_q1_optimizer(ctx, "generalized", stage)
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        summary = _stage_summary_payload(ctx, stage)
        if optimizer_status == "genetic_programming":
            progress = _progress_payload(progress_path)
            if _is_true_gp_payload(summary) and str(progress.get("status", "complete")) == "complete":
                resumed = {
                    "resumed_from_existing": True,
                    "primary_output": str(out_path),
                    "paper1_dir": str(out_dir),
                    "progress_json": str(progress_path),
                    "optimizer_status": optimizer_status,
                    "source": summary.get("source", ""),
                }
                write_stage_summary(ctx, stage, resumed, status="skipped_existing")
                _log(ctx, stage, f"resume: using complete existing output {out_path}")
                print(f"{stage}: complete existing output found; skipping Stage 05 LODO: {out_path}", flush=True)
                return resumed
            if _has_true_gp_resume_evidence(ctx, stage, out_path, progress_path):
                _log(ctx, stage, f"resume: existing GP output appears partial or stale; continuing Stage 05 from checkpoints: {out_path}")
                print(f"{stage}: existing GP output is partial/stale; resuming from checkpoints: {out_path}", flush=True)
            else:
                raise RuntimeError(
                    f"{stage}: complete existing output is not compatible with optimizer=genetic_programming: {out_path}. "
                    "Use a fresh --run-id for the true GP run, or --overwrite this run intentionally."
                )
        else:
            resumed = {
                "resumed_from_existing": True,
                "primary_output": str(out_path),
                "paper1_dir": str(out_dir),
                "progress_json": str(progress_path),
                "optimizer_status": optimizer_status,
                "source": summary.get("source", ""),
            }
            write_stage_summary(ctx, stage, resumed, status="skipped_existing")
            _log(ctx, stage, f"resume: using complete existing output {out_path}")
            print(f"{stage}: complete existing output found; skipping Stage 05 LODO: {out_path}", flush=True)
            return resumed
    if ctx.overwrite:
        for path in (
            out_path,
            progress_path,
            best_workflows_path,
            out_dir / "lodo_results_per_dataset.csv",
            out_dir / "method_rank_summary.csv",
            out_dir / "method_value_summary.csv",
            out_dir / "operator_frequency.csv",
            out_dir / "complexity_runtime_summary.csv",
            out_dir / "failure_summary.csv",
        ):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    manifest = _load_manifest(ctx)
    all_methods = configured_methods(ctx.config)
    datasets = sorted(manifest["dataset"].unique().tolist())
    if optimizer_status == "genetic_programming":
        return _stage05_generalized_gp(
            ctx,
            stage=stage,
            out_dir=out_dir,
            out_path=out_path,
            progress_path=progress_path,
            best_workflows_path=best_workflows_path,
            optimizer_status=optimizer_status,
            manifest=manifest,
            all_methods=all_methods,
            datasets=datasets,
        )
    metrics_path = ctx.output_dir / "q1_metrics" / "per_stack_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 04 metrics for Stage 05 LODO reuse: {metrics_path}")
    all_metric_rows = read_csv(metrics_path)
    if not all_metric_rows:
        raise ValueError(f"Stage 04 metrics are empty: {metrics_path}")
    available_methods = {str(row.get("method", "")) for row in all_metric_rows}
    expected_methods = {str(method["method"]) for method in all_methods}
    missing_methods = sorted(expected_methods - available_methods)
    if missing_methods:
        raise ValueError(f"Stage 04 metrics do not include all configured methods. Missing: {missing_methods[:10]}")

    existing_rows = read_csv(out_path) if out_path.exists() and ctx.resume else []
    completed_heldouts = {str(row.get("heldout_dataset", "")) for row in existing_rows if row.get("heldout_dataset")}
    best_workflows: List[Dict[str, Any]] = []
    if best_workflows_path.exists() and ctx.resume:
        loaded = read_json(best_workflows_path)
        if isinstance(loaded, list):
            best_workflows = [dict(row) for row in loaded]
    selected_methods: List[str] = [str(row.get("selected_method", "")) for row in best_workflows if row.get("selected_method")]
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "source_metrics_csv": str(metrics_path),
            "lodo_results_per_stack": str(out_path),
            "total_heldout_datasets": len(datasets),
            "completed_heldout_datasets": len(completed_heldouts),
            "remaining_heldout_datasets": max(0, len(datasets) - len(completed_heldouts)),
            "progress_percent": 100.0 * len(completed_heldouts) / len(datasets) if datasets else 100.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    _log(
        ctx,
        stage,
        f"started LODO from Stage 04 metrics: datasets={datasets}, already_completed={sorted(completed_heldouts)}, source={metrics_path}",
    )
    print(
        f"{stage}: starting/resuming with {len(completed_heldouts)}/{len(datasets)} held-out datasets complete; source={metrics_path}",
        flush=True,
    )
    for heldout in datasets:
        if heldout in completed_heldouts:
            print(f"{stage}: skipping completed heldout_dataset={heldout}", flush=True)
            continue
        train_datasets = [dataset for dataset in datasets if dataset != heldout]
        if not train_datasets:
            train_datasets = [heldout]
        train_rows = [
            dict(row)
            for row in all_metric_rows
            if str(row.get("dataset")) in set(train_datasets)
            and str(row.get("outer_split")) == "trainval"
            and str(row.get("inner_split")) in {"train", "val"}
        ]
        if not train_rows:
            raise ValueError(f"No Stage 04 training metric rows available for heldout_dataset={heldout}")
        _, _, _, train_summary = summarize_methods(train_rows, ctx.config)
        selected_name = select_best_method(train_summary["method_summary"])
        selected_methods.append(selected_name)
        selected_method = method_by_name(all_methods, selected_name)
        compare_methods = [method_by_name(all_methods, "identity")] if any(str(m["method"]) == "identity" for m in all_methods) else []
        for method in all_methods:
            if method["method"] not in {entry["method"] for entry in compare_methods}:
                compare_methods.append(method)
        if selected_name not in {entry["method"] for entry in compare_methods}:
            compare_methods.append(selected_method)
        compare_method_names = {str(method["method"]) for method in compare_methods}
        heldout_rows = [
            dict(row)
            for row in all_metric_rows
            if str(row.get("dataset")) == heldout and str(row.get("method")) in compare_method_names
        ]
        if not heldout_rows:
            raise ValueError(f"No Stage 04 held-out metric rows available for heldout_dataset={heldout}")
        _, _, _, heldout_summary = summarize_methods(heldout_rows, ctx.config)
        single_candidates = [
            str(row["method"])
            for row in heldout_summary["method_summary"]
            if str(row.get("method", "")).startswith("single::")
        ]
        oracle_single = single_candidates[0] if single_candidates else ""
        fixed_candidates = [
            str(row["method"])
            for row in train_summary["method_summary"]
            if str(row.get("method", "")).startswith("single::")
        ]
        best_fixed_single = fixed_candidates[0] if fixed_candidates else ""
        for row in heldout_rows:
            row["heldout_dataset"] = heldout
            if row["method"] == selected_name:
                row["selection_role"] = "generalized_selected"
            elif best_fixed_single and row["method"] == best_fixed_single:
                row["selection_role"] = "best_fixed_single"
            elif oracle_single and row["method"] == oracle_single:
                row["selection_role"] = "dataset_oracle_single_exploratory"
            else:
                row["selection_role"] = "comparison"
        _append_csv_rows(out_path, heldout_rows, list(heldout_rows[0].keys()))
        best_workflows = [row for row in best_workflows if str(row.get("heldout_dataset", "")) != heldout]
        best_workflows.append(
            {
                "heldout_dataset": heldout,
                "train_datasets": train_datasets,
                "selected_method": selected_name,
                "best_fixed_single_on_training": best_fixed_single,
                "dataset_oracle_single_on_heldout": oracle_single,
                "selection_endpoint": "rank_generalization_score_on_training_datasets",
                "pipeline_json": selected_method.get("pipeline_json", ""),
                "optimizer_status": optimizer_status,
                "source": "stage04_per_stack_metrics",
                "todo": "Replace or augment with full GP optimizer for final Paper 1 experiments.",
            }
        )
        write_json(best_workflows_path, best_workflows)
        completed_heldouts.add(heldout)
        elapsed = max(0.001, (datetime.now(timezone.utc) - start_time).total_seconds())
        rate = len(completed_heldouts) / elapsed
        remaining = max(0, len(datasets) - len(completed_heldouts))
        eta_sec = remaining / rate if rate > 0 else float("nan")
        eta_text = f"{eta_sec / 3600.0:.2f}h" if np.isfinite(eta_sec) else "unknown"
        message = (
            f"{stage}: {len(completed_heldouts)}/{len(datasets)} held-out datasets "
            f"({100.0 * len(completed_heldouts) / len(datasets):.2f}%), "
            f"heldout_dataset={heldout}, selected_method={selected_name}, eta={eta_text}"
        )
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running",
                "run_id": ctx.run_id,
                "source_metrics_csv": str(metrics_path),
                "lodo_results_per_stack": str(out_path),
                "total_heldout_datasets": len(datasets),
                "completed_heldout_datasets": len(completed_heldouts),
                "remaining_heldout_datasets": remaining,
                "progress_percent": 100.0 * len(completed_heldouts) / len(datasets) if datasets else 100.0,
                "current_heldout_dataset": heldout,
                "selected_method": selected_name,
            },
        )
    if len(completed_heldouts) < len(datasets):
        raise RuntimeError(
            f"Stage 05 incomplete: completed {len(completed_heldouts)}/{len(datasets)} held-out datasets. "
            f"Resume with --resume after checking {progress_path}."
        )
    per_stack_all = read_csv(out_path)
    dataset_summary, rank_rows, value_rows, method_payload = summarize_methods(per_stack_all, ctx.config)
    write_csv(out_dir / "lodo_results_per_dataset.csv", dataset_summary)
    write_csv(out_dir / "method_rank_summary.csv", rank_rows)
    write_csv(out_dir / "method_value_summary.csv", value_rows)
    write_json(best_workflows_path, best_workflows)
    write_csv(out_dir / "operator_frequency.csv", _operator_frequency(selected_methods))
    write_csv(out_dir / "complexity_runtime_summary.csv", method_payload["method_summary"])
    write_csv(out_dir / "failure_summary.csv", [{"method": row["method"], "failure_count": row["failure_count"], "failure_rate": row["failure_rate"]} for row in method_payload["method_summary"]])
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "source_metrics_csv": str(metrics_path),
            "lodo_results_per_stack": str(out_path),
            "total_heldout_datasets": len(datasets),
            "completed_heldout_datasets": len(completed_heldouts),
            "remaining_heldout_datasets": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "paper1_dir": str(out_dir),
        "progress_json": str(progress_path),
        "num_lodo_rows": len(per_stack_all),
        "optimizer_status": optimizer_status,
        "source": "stage04_per_stack_metrics",
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote paper1 LODO outputs: {out_dir}")
    return payload


def stage06_dataset_specific_workflow_search(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "06_dataset_specific_workflow_search"
    out_dir = ctx.output_dir / "paper2_dataset_specific"
    out_path = out_dir / "dataset_specific_results_per_stack.csv"
    progress_path = out_dir / "dataset_specific_results.progress.json"
    best_workflows_path = out_dir / "dataset_specific_best_workflows.json"
    optimizer_status = _require_supported_q1_optimizer(ctx, "dataset_specific", stage)
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        summary = _stage_summary_payload(ctx, stage)
        if optimizer_status == "genetic_programming":
            progress = _progress_payload(progress_path)
            if _is_true_gp_payload(summary) and str(progress.get("status", "complete")) == "complete":
                resumed = {
                    "resumed_from_existing": True,
                    "primary_output": str(out_path),
                    "paper2_dir": str(out_dir),
                    "progress_json": str(progress_path),
                    "optimizer_status": optimizer_status,
                    "source": summary.get("source", ""),
                }
                write_stage_summary(ctx, stage, resumed, status="skipped_existing")
                _log(ctx, stage, f"resume: using complete existing output {out_path}")
                print(f"{stage}: complete existing output found; skipping Stage 06 dataset-specific search: {out_path}", flush=True)
                return resumed
            if _has_true_gp_resume_evidence(ctx, stage, out_path, progress_path):
                _log(ctx, stage, f"resume: existing GP output appears partial or stale; continuing Stage 06 from checkpoints: {out_path}")
                print(f"{stage}: existing GP output is partial/stale; resuming from checkpoints: {out_path}", flush=True)
            else:
                raise RuntimeError(
                    f"{stage}: complete existing output is not compatible with optimizer=genetic_programming: {out_path}. "
                    "Use a fresh --run-id for the true GP run, or --overwrite this run intentionally."
                )
        else:
            resumed = {
                "resumed_from_existing": True,
                "primary_output": str(out_path),
                "paper2_dir": str(out_dir),
                "progress_json": str(progress_path),
                "optimizer_status": optimizer_status,
                "source": summary.get("source", ""),
            }
            write_stage_summary(ctx, stage, resumed, status="skipped_existing")
            _log(ctx, stage, f"resume: using complete existing output {out_path}")
            print(f"{stage}: complete existing output found; skipping Stage 06 dataset-specific search: {out_path}", flush=True)
            return resumed
    if ctx.overwrite:
        for path in (
            out_path,
            progress_path,
            best_workflows_path,
            out_dir / "dataset_specific_results_per_dataset.csv",
            out_dir / "operator_frequency_by_dataset.csv",
        ):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    manifest = _load_manifest(ctx)
    all_methods = configured_methods(ctx.config)
    datasets = sorted(manifest["dataset"].unique().tolist())
    if optimizer_status == "genetic_programming":
        return _stage06_dataset_specific_gp(
            ctx,
            stage=stage,
            out_dir=out_dir,
            out_path=out_path,
            progress_path=progress_path,
            best_workflows_path=best_workflows_path,
            optimizer_status=optimizer_status,
            manifest=manifest,
            datasets=datasets,
        )
    metrics_path = ctx.output_dir / "q1_metrics" / "per_stack_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 04 metrics for Stage 06 dataset-specific reuse: {metrics_path}")
    all_metric_rows = read_csv(metrics_path)
    if not all_metric_rows:
        raise ValueError(f"Stage 04 metrics are empty: {metrics_path}")
    available_methods = {str(row.get("method", "")) for row in all_metric_rows}
    expected_methods = {str(method["method"]) for method in all_methods}
    missing_methods = sorted(expected_methods - available_methods)
    if missing_methods:
        raise ValueError(f"Stage 04 metrics do not include all configured methods. Missing: {missing_methods[:10]}")

    existing_rows = read_csv(out_path) if out_path.exists() and ctx.resume else []
    completed_datasets = {str(row.get("optimized_for_dataset", "")) for row in existing_rows if row.get("optimized_for_dataset")}
    best_workflows: List[Dict[str, Any]] = []
    if best_workflows_path.exists() and ctx.resume:
        loaded = read_json(best_workflows_path)
        if isinstance(loaded, list):
            best_workflows = [dict(row) for row in loaded]
    selected_methods: List[str] = [str(row.get("selected_method", "")) for row in best_workflows if row.get("selected_method")]
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "source_metrics_csv": str(metrics_path),
            "dataset_specific_results_per_stack": str(out_path),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
            "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(
        f"{stage}: starting/resuming with {len(completed_datasets)}/{len(datasets)} datasets complete; source={metrics_path}",
        flush=True,
    )
    _log(
        ctx,
        stage,
        f"started dataset-specific search from Stage 04 metrics: datasets={datasets}, already_completed={sorted(completed_datasets)}, source={metrics_path}",
    )
    for dataset in datasets:
        if dataset in completed_datasets:
            print(f"{stage}: skipping completed dataset={dataset}", flush=True)
            continue
        train_rows = [
            dict(row)
            for row in all_metric_rows
            if str(row.get("dataset")) == dataset
            and str(row.get("outer_split")) == "trainval"
            and str(row.get("inner_split")) in {"train", "val"}
        ]
        if not train_rows:
            raise ValueError(f"No Stage 04 training metric rows available for dataset={dataset}")
        _, _, _, train_summary = summarize_methods(train_rows, ctx.config)
        selected_name = select_best_method(train_summary["method_summary"])
        selected_methods.append(selected_name)
        selected_method = method_by_name(all_methods, selected_name)
        test_outer = ["test"] if "test" in set(manifest[manifest["dataset"] == dataset]["outer_split"]) else None
        test_rows = [
            dict(row)
            for row in all_metric_rows
            if str(row.get("dataset")) == dataset
            and str(row.get("method")) == selected_name
            and (test_outer is None or str(row.get("outer_split")) in set(test_outer))
        ]
        if not test_rows:
            raise ValueError(f"No Stage 04 test metric rows available for dataset={dataset}, method={selected_name}")
        for row in test_rows:
            row["optimized_for_dataset"] = dataset
        _append_csv_rows(out_path, test_rows, list(test_rows[0].keys()))
        best_workflows = [row for row in best_workflows if str(row.get("dataset", "")) != dataset]
        best_workflows.append(
            {
                "dataset": dataset,
                "selected_method": selected_name,
                "selection_endpoint": "rank_generalization_score_on_dataset_trainval",
                "pipeline_json": selected_method.get("pipeline_json", ""),
                "optimizer_status": optimizer_status,
                "source": "stage04_per_stack_metrics",
                "todo": "Replace or augment with dataset-specific GP optimizer for final Paper 2 experiments.",
            }
        )
        write_json(best_workflows_path, best_workflows)
        completed_datasets.add(dataset)
        message = _dataset_progress_message(
            stage,
            len(completed_datasets),
            len(datasets),
            dataset=dataset,
            detail=f"selected_method={selected_name}",
            start_time=start_time,
        )
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running",
                "run_id": ctx.run_id,
                "source_metrics_csv": str(metrics_path),
                "dataset_specific_results_per_stack": str(out_path),
                "total_datasets": len(datasets),
                "completed_datasets": len(completed_datasets),
                "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
                "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
                "current_dataset": dataset,
                "selected_method": selected_name,
            },
        )
    if len(completed_datasets) < len(datasets):
        raise RuntimeError(
            f"Stage 06 incomplete: completed {len(completed_datasets)}/{len(datasets)} datasets. "
            f"Resume with --resume after checking {progress_path}."
        )
    per_stack_all = read_csv(out_path)
    dataset_summary, _, _, method_payload = summarize_methods(per_stack_all, ctx.config)
    write_json(best_workflows_path, best_workflows)
    write_csv(out_dir / "dataset_specific_results_per_dataset.csv", dataset_summary)
    write_csv(out_dir / "operator_frequency_by_dataset.csv", _operator_frequency(selected_methods))
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "source_metrics_csv": str(metrics_path),
            "dataset_specific_results_per_stack": str(out_path),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "paper2_dir": str(out_dir),
        "progress_json": str(progress_path),
        "num_dataset_specific_rows": len(per_stack_all),
        "optimizer_status": optimizer_status,
        "source": "stage04_per_stack_metrics",
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote dataset-specific outputs: {out_dir}")
    return payload


def _score_for_dataset_method(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_summary, rank_rows, value_rows, method_payload = summarize_methods(rows, config)
    method_row = method_payload["method_summary"][0] if method_payload["method_summary"] else {}
    return {
        "rank_score": method_row.get("rank_generalization_score", float("nan")),
        "value_score": method_row.get("value_generalization_score", float("nan")),
        "runtime": method_row.get("mean_execution_time", float("nan")),
        "failures": method_row.get("failure_count", 0),
    }


def stage07_compare_generalized_vs_specific(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "07_compare_generalized_vs_specific"
    out_dir = ctx.output_dir / "paper2_dataset_specific"
    out_path = out_dir / "transfer_matrix_rank_score.csv"
    progress_path = out_dir / "transfer_matrix.progress.json"
    value_path = out_dir / "transfer_matrix_value_score.csv"
    runtime_path = out_dir / "transfer_matrix_runtime.csv"
    failure_path = out_dir / "transfer_matrix_failures.csv"
    gap_path = out_dir / "generalization_gap_summary.csv"
    overfit_path = out_dir / "overfitting_summary.csv"
    tests_path = out_dir / "dataset_specific_vs_generalized_tests.csv"
    per_stack_path = out_dir / "transfer_results_per_stack.csv"
    curves_path = out_dir / "transfer_focus_curves.csv"
    manifest = _load_manifest(ctx)
    labels = _load_labels(ctx)
    all_methods = configured_methods(ctx.config)
    paper1_best = read_json(ctx.output_dir / "paper1_generalized" / "generalized_best_workflows.json")
    paper2_best = read_json(out_dir / "dataset_specific_best_workflows.json")
    if not isinstance(paper1_best, list) or not isinstance(paper2_best, list):
        raise ValueError("Stage 07 requires list JSON outputs from Stage 05 and Stage 06")
    best_has_gp = any(str(row.get("optimizer_status", "")) == "genetic_programming" for row in [*paper1_best, *paper2_best])
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        summary = _stage_summary_payload(ctx, stage)
        summary_source = str(summary.get("source", ""))
        if summary_source == "fresh_workflow_evaluation" or (not best_has_gp and summary_source == "stage04_per_stack_metrics"):
            resumed = {
                "resumed_from_existing": True,
                "primary_output": str(out_path),
                "transfer_matrix_rank_score": str(out_path),
                "progress_json": str(progress_path),
                "source": summary_source,
            }
            write_stage_summary(ctx, stage, resumed, status="skipped_existing")
            _log(ctx, stage, f"resume: using complete existing output {out_path}")
            print(f"{stage}: complete existing output found; skipping Stage 07 transfer comparison: {out_path}", flush=True)
            return resumed
        if best_has_gp and _has_fresh_transfer_resume_evidence(ctx, stage, progress_path, per_stack_path):
            _log(ctx, stage, f"resume: existing transfer output appears partial or stale; continuing Stage 07 from checkpoints: {out_path}")
            print(f"{stage}: existing transfer output is partial/stale; resuming from checkpoints: {out_path}", flush=True)
        else:
            raise RuntimeError(
                f"{stage}: existing transfer output is not compatible with the current GP workflow records: {out_path}. "
                "Use a fresh --run-id for true GP transfer evaluation, or --overwrite intentionally."
            )
    if ctx.overwrite:
        for path in (out_path, progress_path, value_path, runtime_path, failure_path, gap_path, overfit_path, tests_path, per_stack_path, curves_path):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    datasets = sorted(manifest["dataset"].unique().tolist())
    generalized_records = {str(row["heldout_dataset"]): dict(row) for row in paper1_best if row.get("heldout_dataset")}
    specific_records = {str(row["dataset"]): dict(row) for row in paper2_best if row.get("dataset")}
    metrics_path = ctx.output_dir / "q1_metrics" / "per_stack_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 04 metrics for selecting Stage 07 single-operator baselines: {metrics_path}")
    all_metric_rows = read_csv(metrics_path)
    if not all_metric_rows:
        raise ValueError(f"Stage 04 metrics are empty: {metrics_path}")
    matrix_columns = [
        "dataset",
        "generalized_workflow",
        *[f"{dataset}_specific_workflow" for dataset in datasets],
        "identity",
        "best_fixed_single",
        "oracle_single_exploratory",
    ]
    existing_rank_rows = read_csv(out_path) if out_path.exists() and ctx.resume else []
    completed_datasets = {str(row.get("dataset", "")) for row in existing_rank_rows if row.get("dataset")}
    start_time = datetime.now(timezone.utc)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "source_metrics_csv_for_baseline_selection": str(metrics_path),
            "source": "fresh_workflow_evaluation",
            "transfer_matrix_rank_score": str(out_path),
            "transfer_results_per_stack": str(per_stack_path),
            "transfer_focus_curves": str(curves_path),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
            "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(
        f"{stage}: starting/resuming with {len(completed_datasets)}/{len(datasets)} evaluation datasets complete; "
        f"all transfer workflows are evaluated fresh",
        flush=True,
    )
    _log(
        ctx,
        stage,
        f"started fresh transfer comparison: datasets={datasets}, already_completed={sorted(completed_datasets)}, baseline_selection_source={metrics_path}",
    )

    for eval_dataset in datasets:
        if eval_dataset in completed_datasets:
            print(f"{stage}: skipping completed evaluation dataset={eval_dataset}", flush=True)
            continue
        if eval_dataset not in generalized_records:
            raise ValueError(f"Missing generalized workflow record for heldout/evaluation dataset={eval_dataset}")
        methods_to_evaluate: List[Dict[str, Any]] = [
            _method_entry_from_workflow_record(
                generalized_records[eval_dataset],
                alias="generalized_workflow",
                method_type="q1_generalized_workflow_transfer",
                all_methods=all_methods,
            )
        ]
        extras_by_method: Dict[str, Mapping[str, Any]] = {
            "generalized_workflow": {
                "selection_role": "generalized_workflow_transfer",
                "workflow_origin": str(generalized_records[eval_dataset].get("source", "fresh_or_configured_workflow")),
            }
        }
        for train_dataset in datasets:
            if train_dataset not in specific_records:
                raise ValueError(f"Missing dataset-specific workflow record for dataset={train_dataset}")
            alias = f"{train_dataset}_specific_workflow"
            methods_to_evaluate.append(
                _method_entry_from_workflow_record(
                    specific_records[train_dataset],
                    alias=alias,
                    method_type="q1_dataset_specific_workflow_transfer",
                    all_methods=all_methods,
                )
            )
            extras_by_method[alias] = {
                "selection_role": "dataset_specific_workflow_transfer",
                "optimized_for_dataset": train_dataset,
                "workflow_origin": str(specific_records[train_dataset].get("source", "fresh_or_configured_workflow")),
            }
        train_pool = [dataset for dataset in datasets if dataset != eval_dataset] or [eval_dataset]
        best_fixed_name, oracle_name = _select_single_operator_baselines(
            ctx=ctx,
            all_methods=all_methods,
            all_metric_rows=all_metric_rows,
            eval_dataset=eval_dataset,
            train_datasets=train_pool,
        )
        comparison_entries, comparison_extras = _comparison_method_entries(
            all_methods=all_methods,
            best_fixed_single=best_fixed_name,
            oracle_single=oracle_name,
        )
        methods_to_evaluate.extend(comparison_entries)
        extras_by_method.update(comparison_extras)
        rank_row = {"dataset": eval_dataset}
        value_row = {"dataset": eval_dataset}
        runtime_row = {"dataset": eval_dataset}
        failure_row = {"dataset": eval_dataset}
        transfer_rows, transfer_curves = _fresh_evaluate_procedures(
            ctx=ctx,
            manifest=manifest,
            labels=labels,
            methods=methods_to_evaluate,
            datasets=[eval_dataset],
            outer_split=None,
            inner_split=None,
            extras_by_method=extras_by_method,
            default_extras={
                "evaluation_dataset": eval_dataset,
                "selection_endpoint": "fresh_transfer_matrix_q1_metrics",
                "train_datasets": "",
                "evaluation_scope": "transfer_evaluation_dataset_all_splits",
            },
            write_curves=True,
        )
        _append_csv_rows(per_stack_path, transfer_rows, Q1_PROCEDURE_METRIC_FIELDNAMES)
        _append_csv_rows(curves_path, transfer_curves, Q1_PROCEDURE_CURVE_FIELDNAMES)
        _, _, _, transfer_summary = summarize_methods(transfer_rows, ctx.config)
        score_lookup = {
            str(row["method"]): {
                "rank_score": row.get("rank_generalization_score", float("nan")),
                "value_score": row.get("value_generalization_score", float("nan")),
                "runtime": row.get("mean_execution_time", float("nan")),
                "failures": row.get("failure_count", 0),
            }
            for row in transfer_summary["method_summary"]
        }
        scores_by_col: Dict[str, Dict[str, Any]] = {}
        for col in matrix_columns[1:]:
            score = score_lookup.get(col, {})
            scores_by_col[col] = score
            rank_row[col] = score.get("rank_score", float("nan"))
            value_row[col] = score.get("value_score", float("nan"))
            runtime_row[col] = score.get("runtime", float("nan"))
            failure_row[col] = score.get("failures", 0)
        _append_csv_rows(out_path, [rank_row], matrix_columns)
        _append_csv_rows(value_path, [value_row], matrix_columns)
        _append_csv_rows(runtime_path, [runtime_row], matrix_columns)
        _append_csv_rows(failure_path, [failure_row], matrix_columns)
        own_col = f"{eval_dataset}_specific_workflow"
        if own_col in scores_by_col and "generalized_workflow" in scores_by_col:
            gap_row = {
                "dataset": eval_dataset,
                "specific_minus_generalized_rank_score": float(scores_by_col[own_col]["rank_score"]) - float(scores_by_col["generalized_workflow"]["rank_score"]),
                "specific_minus_generalized_value_score": float(scores_by_col[own_col]["value_score"]) - float(scores_by_col["generalized_workflow"]["value_score"]),
                "interpretation": "specific_better" if float(scores_by_col[own_col]["rank_score"]) < float(scores_by_col["generalized_workflow"]["rank_score"]) else "generalized_preferable_or_tied",
            }
            _append_csv_rows(gap_path, [gap_row], list(gap_row.keys()))
        completed_datasets.add(eval_dataset)
        message = _dataset_progress_message(
            stage,
            len(completed_datasets),
            len(datasets),
            dataset=eval_dataset,
            detail="fresh_transfer_metrics_written",
            start_time=start_time,
        )
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running",
                "run_id": ctx.run_id,
                "source_metrics_csv_for_baseline_selection": str(metrics_path),
                "source": "fresh_workflow_evaluation",
                "transfer_matrix_rank_score": str(out_path),
                "transfer_results_per_stack": str(per_stack_path),
                "transfer_focus_curves": str(curves_path),
                "total_datasets": len(datasets),
                "completed_datasets": len(completed_datasets),
                "remaining_datasets": max(0, len(datasets) - len(completed_datasets)),
                "progress_percent": 100.0 * len(completed_datasets) / len(datasets) if datasets else 100.0,
                "current_dataset": eval_dataset,
            },
        )
    if len(completed_datasets) < len(datasets):
        raise RuntimeError(
            f"Stage 07 incomplete: completed {len(completed_datasets)}/{len(datasets)} evaluation datasets. "
            f"Resume with --resume after checking {progress_path}."
        )
    rank_matrix = read_csv(out_path)
    gap_rows = read_csv(gap_path) if gap_path.exists() else []
    overfit_rows: List[Dict[str, Any]] = []
    for train_dataset in datasets:
        col_name = f"{train_dataset}_specific_workflow"
        other_scores = [float(row.get(col_name, float("nan"))) for row in rank_matrix if row["dataset"] != train_dataset]
        overfit_rows.append(
            {
                "workflow": col_name,
                "own_dataset_rank_score": float(next(row[col_name] for row in rank_matrix if row["dataset"] == train_dataset)),
                "other_dataset_mean_rank_score": float(np.nanmean(other_scores)) if other_scores else float("nan"),
                "overfitting_gap": float(np.nanmean(other_scores)) - float(next(row[col_name] for row in rank_matrix if row["dataset"] == train_dataset)) if other_scores else float("nan"),
                "interpretation": "check_for_overfitting_if_positive_gap_is_large",
            }
        )
    write_csv(overfit_path, overfit_rows)
    write_csv(tests_path, gap_rows)
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "complete",
            "run_id": ctx.run_id,
            "source_metrics_csv_for_baseline_selection": str(metrics_path),
            "source": "fresh_workflow_evaluation",
            "transfer_matrix_rank_score": str(out_path),
            "transfer_results_per_stack": str(per_stack_path),
            "transfer_focus_curves": str(curves_path),
            "total_datasets": len(datasets),
            "completed_datasets": len(completed_datasets),
            "remaining_datasets": 0,
            "progress_percent": 100.0,
        },
    )
    payload = {
        "transfer_matrix_rank_score": str(out_path),
        "progress_json": str(progress_path),
        "transfer_results_per_stack": str(per_stack_path),
        "transfer_focus_curves": str(curves_path),
        "num_datasets": len(datasets),
        "source": "fresh_workflow_evaluation",
    }
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote generalized-vs-specific comparison: {out_dir}")
    return payload


def _rank_matrix_from_dataset_summary(dataset_summary_rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, List[str], List[str]]:
    from .aggregation import align_metric_value, average_ranks

    datasets = sorted({str(row["dataset"]) for row in dataset_summary_rows})
    methods = sorted({str(row["method"]) for row in dataset_summary_rows})
    lookup = {(str(row["dataset"]), str(row["method"])): row for row in dataset_summary_rows}
    cells = {method: [] for method in methods}
    blocks: List[str] = []
    for dataset in datasets:
        for metric in AUTOFOCUS_METRICS:
            aligned = [
                align_metric_value(metric, float(lookup.get((dataset, method), {}).get(f"{metric}_mean", float("nan"))))
                for method in methods
            ]
            ranks = average_ranks(aligned, lower_better=True)
            for method, rank in zip(methods, ranks):
                cells[method].append(float(rank))
            blocks.append(f"{dataset}::{metric}")
    matrix = np.asarray([cells[method] for method in methods], dtype=float)
    return matrix, methods, blocks


def stage08_statistics_and_sensitivity(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "08_statistics_and_sensitivity"
    out_dir = ctx.output_dir / "statistics"
    out_path = out_dir / "bootstrap_ci.csv"
    progress_path = out_dir / "statistics.progress.json"
    outputs = {
        "bootstrap_ci": out_path,
        "friedman_results": out_dir / "friedman_results.csv",
        "nemenyi_cd_data": out_dir / "nemenyi_cd_data.csv",
        "pairwise_wilcoxon_holm": out_dir / "pairwise_wilcoxon_holm.csv",
        "alpha_sensitivity": out_dir / "alpha_sensitivity.csv",
        "metric_weight_sensitivity": out_dir / "metric_weight_sensitivity.csv",
        "dataset_weight_sensitivity": out_dir / "dataset_weight_sensitivity.csv",
        "rank_stability": out_dir / "rank_stability.csv",
        "runtime_performance_tradeoff": out_dir / "runtime_performance_tradeoff.csv",
        "warnings": out_dir / "warnings.md",
    }
    if ctx.resume and out_path.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        resumed = {
            "resumed_from_existing": True,
            "primary_output": str(out_path),
            "statistics_dir": str(out_dir),
            "progress_json": str(progress_path),
        }
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using complete existing output {out_path}")
        print(f"{stage}: complete existing output found; skipping statistics: {out_path}", flush=True)
        return resumed
    if ctx.overwrite:
        for path in (progress_path, *outputs.values()):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(out_path, resume=ctx.resume, overwrite=ctx.overwrite)
    from .statistics import bootstrap_ci_mean, friedman_wilcoxon_holm, nemenyi_cd

    total_steps = 10
    completed_steps = 0
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "statistics_dir": str(out_dir),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "remaining_steps": total_steps - completed_steps,
            "progress_percent": 0.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(f"{stage}: starting/resuming statistics and sensitivity analysis; output={out_dir}", flush=True)
    _log(ctx, stage, f"started statistics and sensitivity analysis: output={out_dir}")

    def mark_step(step: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        message = _stage_step_message(stage, completed_steps, total_steps, step=step)
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running" if completed_steps < total_steps else "complete",
                "run_id": ctx.run_id,
                "statistics_dir": str(out_dir),
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "remaining_steps": max(0, total_steps - completed_steps),
                "progress_percent": 100.0 * completed_steps / total_steps,
                "current_step": step,
            },
        )

    dataset_summary = read_csv(ctx.output_dir / "q1_metrics" / "dataset_metric_summary.csv")
    mark_step("load_dataset_metric_summary")
    rank_rows, rank_cells = compute_rank_based_summary(dataset_summary)
    mark_step("rank_based_summary")
    bootstrap_rows = []
    for row in rank_rows:
        ranks = rank_cells[row["method"]]["ranks"]
        ci_low, ci_high = bootstrap_ci_mean(ranks, n_resamples=int(ctx.config.get("statistics", {}).get("bootstrap_resamples", 1000)))
        bootstrap_rows.append({"method": row["method"], "mean_rank": row["overall_rank_mean"], "ci_low": ci_low, "ci_high": ci_high})
    mark_step("bootstrap_confidence_intervals")
    matrix, methods, blocks = _rank_matrix_from_dataset_summary(dataset_summary)
    friedman_rows, pairwise_rows, warnings = friedman_wilcoxon_holm(matrix, methods, blocks, family="q1_workflow_rank")
    cd = nemenyi_cd(len(methods), len(blocks))
    mark_step("friedman_wilcoxon_nemenyi")
    agg_cfg = ctx.config.get("aggregation", {}) if isinstance(ctx.config.get("aggregation"), dict) else {}
    alpha_values = agg_cfg.get("alpha_sensitivity", [0.5, 0.6, 0.7, 0.8, 0.9])
    alpha_rows = []
    for alpha in alpha_values:
        rows = compute_value_based_summary(dataset_summary, alpha=float(alpha), metric_weights=agg_cfg.get("metric_weights") or DEFAULT_METRIC_WEIGHTS)
        for row in rows:
            alpha_rows.append({"alpha": alpha, **row})
    mark_step("alpha_sensitivity")
    metric_weight_rows = []
    for scheme_name, weights in (agg_cfg.get("metric_weight_schemes") or {"default": agg_cfg.get("metric_weights") or DEFAULT_METRIC_WEIGHTS}).items():
        rows = compute_value_based_summary(dataset_summary, metric_weights=weights)
        metric_weight_rows.append({"scheme": scheme_name, "top_method": rows[0]["method"] if rows else "", "top_score": rows[0]["value_generalization_score"] if rows else float("nan")})
    mark_step("metric_weight_sensitivity")
    dataset_weight_rows = []
    datasets = sorted({row["dataset"] for row in dataset_summary})
    for target in datasets:
        weights = {dataset: (2.0 if dataset == target else 1.0) for dataset in datasets}
        rows = compute_value_based_summary(dataset_summary, dataset_weights=weights)
        dataset_weight_rows.append({"scheme": f"upweight_{target}", "top_method": rows[0]["method"] if rows else "", "top_score": rows[0]["value_generalization_score"] if rows else float("nan")})
    mark_step("dataset_weight_sensitivity")
    rank_stability_rows = []
    value_rows = compute_value_based_summary(dataset_summary)
    value_rank = {row["method"]: row["final_rank"] for row in value_rows}
    for row in rank_rows:
        rank_stability_rows.append({"method": row["method"], "rank_based_rank": row["final_rank"], "value_based_rank": value_rank.get(row["method"], ""), "absolute_rank_shift": abs(float(row["final_rank"]) - float(value_rank.get(row["method"], row["final_rank"])))})
    mark_step("rank_stability")
    runtime_rows = read_csv(ctx.output_dir / "q1_metrics" / "method_summary.csv")
    mark_step("runtime_tradeoff")
    write_csv(out_path, bootstrap_rows)
    write_csv(out_dir / "friedman_results.csv", friedman_rows)
    write_csv(out_dir / "nemenyi_cd_data.csv", [{"num_methods": len(methods), "num_blocks": len(blocks), "critical_difference": cd}])
    write_csv(out_dir / "pairwise_wilcoxon_holm.csv", pairwise_rows)
    write_csv(out_dir / "alpha_sensitivity.csv", alpha_rows)
    write_csv(out_dir / "metric_weight_sensitivity.csv", metric_weight_rows)
    write_csv(out_dir / "dataset_weight_sensitivity.csv", dataset_weight_rows)
    write_csv(out_dir / "rank_stability.csv", rank_stability_rows)
    write_csv(out_dir / "runtime_performance_tradeoff.csv", runtime_rows)
    write_text(out_dir / "warnings.md", "\n".join(f"- {warning}" for warning in warnings) + ("\n" if warnings else "No statistical assumption warnings.\n"))
    mark_step("write_statistics_outputs")
    payload = {"statistics_dir": str(out_dir), "progress_json": str(progress_path), "warnings": warnings}
    write_stage_summary(ctx, stage, payload)
    _log(ctx, stage, f"wrote statistics outputs: {out_dir}")
    return payload


def _copy_table_with_latex(src: Path, csv_dst: Path, tex_dst: Path) -> None:
    rows = read_csv(src) if src.exists() else []
    write_csv(csv_dst, rows)
    write_latex_table(tex_dst, rows)


def stage09_export_paper_assets(ctx: Q1RunContext) -> Dict[str, Any]:
    stage = "09_export_paper_assets"
    asset_marker = ctx.output_dir / "summaries" / "09_export_paper_assets_assets.json"
    progress_path = ctx.output_dir / "summaries" / "09_export_paper_assets.progress.json"
    if ctx.resume and asset_marker.exists() and _summary_is_complete(ctx, stage) and not ctx.overwrite:
        resumed = {
            "resumed_from_existing": True,
            "primary_output": str(asset_marker),
            "asset_manifest": str(asset_marker),
            "progress_json": str(progress_path),
        }
        write_stage_summary(ctx, stage, resumed, status="skipped_existing")
        _log(ctx, stage, f"resume: using complete existing asset marker {asset_marker}")
        print(f"{stage}: complete existing paper assets found; skipping export: {asset_marker}", flush=True)
        return resumed
    if ctx.overwrite:
        for path in (asset_marker, progress_path):
            if path.exists():
                path.unlink()
    else:
        ensure_stage_writable(asset_marker, resume=ctx.resume, overwrite=ctx.overwrite)
    paper1_assets = ctx.output_dir / "paper1_generalized" / "assets"
    paper2_assets = ctx.output_dir / "paper2_dataset_specific" / "assets"
    q1_metrics = ctx.output_dir / "q1_metrics"
    stats_dir = ctx.output_dir / "statistics"
    table_s1_csv = ctx.output_dir / "tables" / "Table_S1_ParameterSpace.csv"
    table_s1_tex = ctx.output_dir / "tables" / "Table_S1_ParameterSpace.tex"
    paper1_tables = {
        "Table_01_Datasets": ctx.output_dir / "reproducibility" / "dataset_summary.csv",
        "Table_02_MethodsCompared": q1_metrics / "method_summary.csv",
        "Table_03_MainResults": ctx.output_dir / "paper1_generalized" / "method_rank_summary.csv",
        "Table_04_Statistics": stats_dir / "friedman_results.csv",
        "Table_S1_ParameterSpace": table_s1_csv,
        "Table_S2_FullOperatorRanking": q1_metrics / "method_value_summary.csv",
        "Table_S3_FullLODOResults": ctx.output_dir / "paper1_generalized" / "lodo_results_per_dataset.csv",
    }
    paper2_tables = {
        "Table_01_Datasets": ctx.output_dir / "reproducibility" / "dataset_summary.csv",
        "Table_02_WorkflowsCompared": q1_metrics / "method_summary.csv",
        "Table_03_TransferMatrix": ctx.output_dir / "paper2_dataset_specific" / "transfer_matrix_rank_score.csv",
        "Table_04_GeneralizationGap": ctx.output_dir / "paper2_dataset_specific" / "generalization_gap_summary.csv",
        "Table_05_Statistics": stats_dir / "pairwise_wilcoxon_holm.csv",
        "Table_S1_DatasetSpecificPipelines": ctx.output_dir / "paper2_dataset_specific" / "dataset_specific_best_workflows.json",
        "Table_S2_FullTransferResults": ctx.output_dir / "paper2_dataset_specific" / "transfer_matrix_value_score.csv",
    }
    figure_tasks = [
        ("Figure_01_Workflow", paper1_assets / "Figure_01_Workflow"),
        ("Figure_02_LODOGeneralization", paper1_assets / "Figure_02_LODOGeneralization"),
        ("Figure_03_MethodRankComparison", paper1_assets / "Figure_03_MethodRankComparison"),
        ("Figure_04_DatasetwisePerformance", paper1_assets / "Figure_04_DatasetwisePerformance"),
        ("Figure_05_RuntimeComplexityTradeoff", paper1_assets / "Figure_05_RuntimeComplexityTradeoff"),
        ("Figure_06_RepresentativeFocusCurves", paper1_assets / "Figure_06_RepresentativeFocusCurves"),
        ("Figure_01_Workflow", paper2_assets / "Figure_01_Workflow"),
        ("Figure_02_TransferMatrix", paper2_assets / "Figure_02_TransferMatrix"),
        ("Figure_03_GeneralizedVsSpecific", paper2_assets / "Figure_03_GeneralizedVsSpecific"),
        ("Figure_04_OperatorFrequencyByDataset", paper2_assets / "Figure_04_OperatorFrequencyByDataset"),
        ("Figure_05_OverfittingGap", paper2_assets / "Figure_05_OverfittingGap"),
        ("Figure_06_RuntimeComplexityTradeoff", paper2_assets / "Figure_06_RuntimeComplexityTradeoff"),
    ]
    total_steps = 1 + len(paper1_tables) + len(paper2_tables) + len(figure_tasks) + 1
    completed_steps = 0
    _write_progress(
        progress_path,
        {
            "stage": stage,
            "status": "running",
            "run_id": ctx.run_id,
            "asset_manifest": str(asset_marker),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "remaining_steps": total_steps - completed_steps,
            "progress_percent": 0.0,
            "resume_mode": bool(ctx.resume),
        },
    )
    print(f"{stage}: starting/resuming paper asset export; asset_marker={asset_marker}", flush=True)
    _log(ctx, stage, f"started paper asset export: asset_marker={asset_marker}")

    def mark_step(step: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        message = _stage_step_message(stage, completed_steps, total_steps, step=step)
        _log(ctx, stage, message)
        print(message, flush=True)
        _write_progress(
            progress_path,
            {
                "stage": stage,
                "status": "running" if completed_steps < total_steps else "complete",
                "run_id": ctx.run_id,
                "asset_manifest": str(asset_marker),
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "remaining_steps": max(0, total_steps - completed_steps),
                "progress_percent": 100.0 * completed_steps / total_steps,
                "current_step": step,
            },
        )

    parameter_rows = parameter_space_rows()
    write_csv(table_s1_csv, parameter_rows)
    write_latex_table(table_s1_tex, parameter_rows[:40])
    mark_step("parameter_space_table")
    for name, src in paper1_tables.items():
        if src.suffix == ".json":
            rows = read_json(src)
            if isinstance(rows, dict):
                rows = [rows]
            write_csv(paper1_assets / f"{name}.csv", rows)
            write_latex_table(paper1_assets / f"{name}.tex", rows)
        else:
            _copy_table_with_latex(src, paper1_assets / f"{name}.csv", paper1_assets / f"{name}.tex")
        mark_step(f"paper1_table_{name}")
    for name, src in paper2_tables.items():
        if src.suffix == ".json":
            rows = read_json(src)
            if isinstance(rows, dict):
                rows = [rows]
            write_csv(paper2_assets / f"{name}.csv", rows)
            write_latex_table(paper2_assets / f"{name}.tex", rows)
        else:
            _copy_table_with_latex(src, paper2_assets / f"{name}.csv", paper2_assets / f"{name}.tex")
        mark_step(f"paper2_table_{name}")
    method_summary = read_csv(q1_metrics / "method_summary.csv")
    write_simple_bar_figure(paper1_assets / "Figure_01_Workflow", method_summary, x_key="method", y_key="rank_generalization_score", title="Paper 1 workflow comparison", ylabel="Rank score (lower is better)")
    mark_step("paper1_figure_Figure_01_Workflow")
    write_simple_bar_figure(paper1_assets / "Figure_02_LODOGeneralization", read_csv(ctx.output_dir / "paper1_generalized" / "method_rank_summary.csv"), x_key="method", y_key="rank_generalization_score", title="LODO generalization", ylabel="Rank score")
    mark_step("paper1_figure_Figure_02_LODOGeneralization")
    write_simple_bar_figure(paper1_assets / "Figure_03_MethodRankComparison", method_summary, x_key="method", y_key="rank_final_rank", title="Method rank comparison", ylabel="Final rank")
    mark_step("paper1_figure_Figure_03_MethodRankComparison")
    write_simple_bar_figure(paper1_assets / "Figure_04_DatasetwisePerformance", read_csv(ctx.output_dir / "paper1_generalized" / "lodo_results_per_dataset.csv"), x_key="dataset", y_key="absolute_peak_localization_error_mean", title="Dataset-wise performance", ylabel="Peak error")
    mark_step("paper1_figure_Figure_04_DatasetwisePerformance")
    write_simple_bar_figure(paper1_assets / "Figure_05_RuntimeComplexityTradeoff", method_summary, x_key="method", y_key="mean_execution_time", title="Runtime tradeoff", ylabel="Seconds/image")
    mark_step("paper1_figure_Figure_05_RuntimeComplexityTradeoff")
    write_focus_curve_figure(paper1_assets / "Figure_06_RepresentativeFocusCurves", read_csv(ctx.output_dir / "q1_curves" / "preprocessed_focus_curves.csv"), title="Representative normalized focus curves")
    mark_step("paper1_figure_Figure_06_RepresentativeFocusCurves")
    write_simple_bar_figure(paper2_assets / "Figure_01_Workflow", method_summary, x_key="method", y_key="rank_generalization_score", title="Paper 2 workflow comparison", ylabel="Rank score")
    mark_step("paper2_figure_Figure_01_Workflow")
    write_heatmap_figure(paper2_assets / "Figure_02_TransferMatrix", read_csv(ctx.output_dir / "paper2_dataset_specific" / "transfer_matrix_rank_score.csv"), title="Transfer matrix rank score")
    mark_step("paper2_figure_Figure_02_TransferMatrix")
    write_simple_bar_figure(paper2_assets / "Figure_03_GeneralizedVsSpecific", read_csv(ctx.output_dir / "paper2_dataset_specific" / "generalization_gap_summary.csv"), x_key="dataset", y_key="specific_minus_generalized_rank_score", title="Generalized vs dataset-specific", ylabel="Specific - generalized")
    mark_step("paper2_figure_Figure_03_GeneralizedVsSpecific")
    write_simple_bar_figure(paper2_assets / "Figure_04_OperatorFrequencyByDataset", read_csv(ctx.output_dir / "paper2_dataset_specific" / "operator_frequency_by_dataset.csv"), x_key="operator_or_workflow", y_key="count", title="Operator frequency by dataset", ylabel="Count")
    mark_step("paper2_figure_Figure_04_OperatorFrequencyByDataset")
    write_simple_bar_figure(paper2_assets / "Figure_05_OverfittingGap", read_csv(ctx.output_dir / "paper2_dataset_specific" / "overfitting_summary.csv"), x_key="workflow", y_key="overfitting_gap", title="Overfitting gap", ylabel="Other - own score")
    mark_step("paper2_figure_Figure_05_OverfittingGap")
    write_simple_bar_figure(paper2_assets / "Figure_06_RuntimeComplexityTradeoff", read_csv(ctx.output_dir / "paper2_dataset_specific" / "transfer_matrix_runtime.csv"), x_key="dataset", y_key="generalized", title="Runtime complexity tradeoff", ylabel="Seconds/image")
    mark_step("paper2_figure_Figure_06_RuntimeComplexityTradeoff")
    manifest = {"paper1_assets": str(paper1_assets), "paper2_assets": str(paper2_assets), "parameter_space_csv": str(table_s1_csv), "progress_json": str(progress_path)}
    write_json(asset_marker, manifest)
    mark_step("write_asset_manifest")
    write_stage_summary(ctx, stage, manifest)
    _log(ctx, stage, f"wrote paper assets: {paper1_assets}, {paper2_assets}")
    return manifest


def export_reproducibility(ctx: Q1RunContext) -> Dict[str, Any]:
    repro_dir = ctx.output_dir / "reproducibility"
    manifest_path = _manifest_path(ctx)
    env = environment_metadata()
    source_hash = source_tree_hash()
    write_json(repro_dir / "environment.json", env)
    write_json(repro_dir / "source_tree_hash.json", source_hash)
    failures = 0
    metrics_path = ctx.output_dir / "q1_metrics" / "per_stack_metrics.csv"
    if metrics_path.exists():
        rows = read_csv(metrics_path)
        failures = int(sum(1 for row in rows if str(row.get("failure", "0")) not in {"0", "0.0", ""}))
    run_summary = {
        "run_id": ctx.run_id,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "config_hash": stable_hash_file(ctx.config_path),
        "manifest_hash": stable_hash_file(manifest_path) if manifest_path.exists() else "",
        "source_tree_hash": source_hash["source_tree_sha256"],
        "random_seeds": {"seed": ctx.config.get("seed", ctx.config.get("random_seed", ""))},
        "dataset_paths": ctx.config.get("data", {}),
        "num_failures": failures,
        "output_dir": str(ctx.output_dir),
    }
    write_json(repro_dir / "run_summary.json", run_summary)
    write_text(
        repro_dir / "README.md",
        "\n".join(
            [
                "# Q1 Reproducibility Package",
                "",
                f"- Run ID: `{ctx.run_id}`",
                f"- Config hash: `{run_summary['config_hash']}`",
                f"- Manifest hash: `{run_summary['manifest_hash']}`",
                f"- Source tree hash: `{run_summary['source_tree_hash']}`",
                f"- Output directory: `{ctx.output_dir}`",
                "",
            ]
        ),
    )
    return run_summary


def run_all_stages(ctx: Q1RunContext) -> List[Dict[str, Any]]:
    stage_functions = [
        ("01_build_stacks_and_manifest", stage01_build_stacks_and_manifest),
        ("02_build_reference_labels", stage02_build_reference_labels),
        ("03_run_preprocessed_focus_curves", stage03_run_preprocessed_focus_curves),
        ("04_evaluate_workflows_q1_metrics", stage04_evaluate_workflows_q1_metrics),
        ("05_generalized_workflow_search_lodo", stage05_generalized_workflow_search_lodo),
        ("06_dataset_specific_workflow_search", stage06_dataset_specific_workflow_search),
        ("07_compare_generalized_vs_specific", stage07_compare_generalized_vs_specific),
        ("08_statistics_and_sensitivity", stage08_statistics_and_sensitivity),
        ("09_export_paper_assets", stage09_export_paper_assets),
    ]
    results: List[Dict[str, Any]] = []
    for stage_name, stage_func in stage_functions:
        start_time = datetime.now(timezone.utc)
        message = f"{stage_name}: started at {start_time.isoformat()}"
        _log(ctx, stage_name, message)
        print(message, flush=True)
        result = stage_func(ctx)
        results.append(result)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        done_message = f"{stage_name}: completed in {elapsed:.2f}s"
        _log(ctx, stage_name, done_message)
        print(done_message, flush=True)
    export_reproducibility(ctx)
    return results

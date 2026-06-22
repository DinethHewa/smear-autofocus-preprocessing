from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import math

import pandas as pd
import yaml


def _to_python(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_python(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def safe_yaml_dump(obj: Any) -> str:
    cleaned = _to_python(obj)
    return yaml.safe_dump(cleaned, sort_keys=False)


def dump_config_used(cfg: Dict[str, Any], run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    out_path = run_path / "config_used.yaml"
    out_path.write_text(safe_yaml_dump(cfg), encoding="utf-8")
    return str(out_path)


def format_run_summary_line(
    run_id: str,
    generalization_score: float,
    stability_ratio: float,
    ms_per_image: float,
    num_failed: int,
    num_total: int,
) -> str:
    return (
        f"RUN_SUMMARY: run_id={run_id}, "
        f"generalization_score={generalization_score:.6f}, "
        f"stability_ratio={stability_ratio:.3f}, "
        f"ms_per_image={ms_per_image:.3f}, "
        f"failures={num_failed}/{num_total}"
    )


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _to_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _parse_failure_counts(df: pd.DataFrame) -> Tuple[int, int]:
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


def _compute_ms_per_image(df: pd.DataFrame, total: int) -> float:
    if "runtime_sec" in df.columns:
        runtimes = pd.to_numeric(df["runtime_sec"], errors="coerce").fillna(0.0)
        if total > 0:
            return float(1000.0 * runtimes.sum() / total)
    if "ms_per_image" in df.columns:
        vals = pd.to_numeric(df["ms_per_image"], errors="coerce")
        if vals.notna().any():
            return float(vals.mean())
    return float("nan")


def _load_summary_metrics(run_dir: Path) -> Tuple[float, float]:
    eval_summary = run_dir / "eval_summary.json"
    if eval_summary.exists():
        data = _read_json(eval_summary) or {}
        score = _to_float_or_nan(data.get("generalization_score", float("nan")))
        ratio = _to_float_or_nan(data.get("stability_ratio", float("nan")))
        return score, ratio
    for name in ("run_summary.json", "eval_summary.json", "metrics.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = _read_json(path) or {}
        score = data.get("generalization_score", data.get("mean_score", float("nan")))
        ratio = data.get("global_stability_ratio", data.get("stability_ratio", float("nan")))
        return float(score), float(ratio)

    run_summaries = sorted(run_dir.glob("run_summary_*.json"))
    if run_summaries:
        data = _read_json(run_summaries[0]) or {}
        score = data.get("generalization_score", data.get("mean_score", float("nan")))
        ratio = data.get("global_stability_ratio", data.get("stability_ratio", float("nan")))
        return float(score), float(ratio)

    best_score_path = run_dir / "best_score.json"
    if best_score_path.exists():
        data = _read_json(best_score_path) or {}
        score = data.get("best_score", float("nan"))
        return float(score), float("nan")

    best_params_path = run_dir / "best_params.json"
    if best_params_path.exists():
        data = _read_json(best_params_path) or {}
        score = data.get("best_value", float("nan"))
        return float(score), float("nan")

    return float("nan"), float("nan")


def build_run_summary_line(run_dir: str | Path, run_id: str) -> str:
    run_path = Path(run_dir)
    eval_summary = run_path / "eval_summary.json"
    if eval_summary.exists():
        data = _read_json(eval_summary) or {}
        generalization_score = _to_float_or_nan(data.get("generalization_score", float("nan")))
        stability_ratio = _to_float_or_nan(data.get("stability_ratio", float("nan")))
        ms_per_image = _to_float_or_nan(data.get("ms_per_image", float("nan")))
        num_failed = int(data.get("failures_failed", 0))
        num_total = int(data.get("failures_total", 0))
    else:
        generalization_score, stability_ratio = _load_summary_metrics(run_path)
        num_failed = 0
        num_total = 0
        ms_per_image = float("nan")

        for name in ("per_stack_results_eval.csv", "eval_results_eval.csv", "per_stack_results.csv"):
            path = run_path / name
            if not path.exists():
                continue
            df = _read_csv(path)
            if df is None:
                continue
            num_failed, num_total = _parse_failure_counts(df)
            ms_per_image = _compute_ms_per_image(df, num_total)
            break

        if num_total == 0:
            failures_path = run_path / "failures_eval.csv"
            if not failures_path.exists():
                failures_path = run_path / "failures.csv"
            if failures_path.exists():
                df_fail = _read_csv(failures_path)
                if df_fail is not None:
                    num_failed = int(len(df_fail))
                    num_total = 0

    if math.isnan(generalization_score):
        generalization_score = float("nan")
    if math.isnan(stability_ratio):
        stability_ratio = float("nan")
    if math.isnan(ms_per_image):
        ms_per_image = float("nan")

    return format_run_summary_line(
        run_id=run_id,
        generalization_score=generalization_score,
        stability_ratio=stability_ratio,
        ms_per_image=ms_per_image,
        num_failed=num_failed,
        num_total=num_total,
    )


def write_run_summary(run_dir: str | Path, run_id: str, summary: Dict[str, Any]) -> str:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    out_path = run_path / "run_summary.txt"

    if "line" in summary:
        line = str(summary["line"])
    else:
        line = format_run_summary_line(
            run_id=run_id,
            generalization_score=float(summary.get("generalization_score", float("nan"))),
            stability_ratio=float(summary.get("stability_ratio", float("nan"))),
            ms_per_image=float(summary.get("ms_per_image", float("nan"))),
            num_failed=int(summary.get("num_failed", 0)),
            num_total=int(summary.get("num_total", 0)),
        )

    with out_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return str(out_path)

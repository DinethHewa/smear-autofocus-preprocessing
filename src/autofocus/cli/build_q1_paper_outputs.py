from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SVG_NS = "http://www.w3.org/2000/svg"
EPS = 1e-8


@dataclass(frozen=True)
class ResolvedRun:
    artifact_dir: Path
    gp_dir: Path
    per_op_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication-ready figures and tables from a saved autofocus_preprocess artifact run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Artifact run directory that contains both gp/ and optuna_per_op/ outputs.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output package directory. Defaults to <repo>/paper_outputs/<run_id>.",
    )
    return parser.parse_args()


def _resolve_run(run_dir: str | Path) -> ResolvedRun:
    base = Path(run_dir).expanduser().resolve()
    if (base / "gp").exists() and (base / "optuna_per_op").exists():
        return ResolvedRun(artifact_dir=base, gp_dir=base / "gp", per_op_dir=base / "optuna_per_op")
    if base.name == "gp" and (base.parent / "optuna_per_op").exists():
        artifact_dir = base.parent
        return ResolvedRun(artifact_dir=artifact_dir, gp_dir=base, per_op_dir=artifact_dir / "optuna_per_op")
    raise ValueError(f"Could not resolve artifact run with gp/ and optuna_per_op/: {base}")


def _default_outdir(resolved: ResolvedRun) -> Path:
    return REPO_ROOT / "paper_outputs" / resolved.artifact_dir.name


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if fieldnames is None:
        fieldnames = list(rows_list[0].keys()) if rows_list else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows_list:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    as_float = _to_float(value, default=float(default))
    if math.isnan(as_float):
        return default
    return int(round(as_float))


def _short_json(data: Any) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, sort_keys=True, ensure_ascii=True)


def _compact_params(data: Mapping[str, Any]) -> str:
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            value_text = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}")
    return "; ".join(parts)


def _objective_mode(cfg: Mapping[str, Any]) -> str:
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
    return "reference" if has_refs else "unsupervised"


def _dataset_weights(metrics_cfg: Mapping[str, Any]) -> Dict[str, float]:
    raw = metrics_cfg.get("dataset_weights", {}) if isinstance(metrics_cfg, dict) else {}
    weights: Dict[str, float] = {}
    if not isinstance(raw, dict):
        return weights
    for key, value in raw.items():
        numeric = _to_float(value)
        if not math.isnan(numeric):
            weights[str(key)] = numeric
    return weights


def _compute_dataset_summary(
    rows: Sequence[Mapping[str, Any]],
    failure_penalty: float,
    stability_eps: float = EPS,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for dataset in sorted(grouped):
        subset = grouped[dataset]
        values = np.asarray([_to_float(row.get("total_fitness")) for row in subset], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            mean_fitness = float(np.mean(finite))
            std_fitness = float(np.std(finite))
        else:
            mean_fitness = float(failure_penalty)
            std_fitness = float(failure_penalty)
        hard_failure_rate = float(np.mean([_to_float(row.get("is_failure"), 1.0) for row in subset])) if subset else 1.0
        stability_ratio = float(abs(mean_fitness) / (std_fitness + stability_eps))
        summaries.append(
            {
                "dataset": dataset,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "dataset_score": float(mean_fitness + std_fitness),
                "HardFailureRate": hard_failure_rate,
                "StabilityRatio": stability_ratio,
                "n_stacks": len(subset),
            }
        )
    return summaries


def _compute_generalization(
    dataset_summary: Sequence[Mapping[str, Any]],
    metrics_cfg: Mapping[str, Any],
    objective_mode: str,
    hard_failure_rate: float,
) -> Dict[str, float]:
    if not dataset_summary:
        failure_penalty = _to_float(metrics_cfg.get("failure_penalty"), 1e6)
        return {
            "generalization_score": failure_penalty,
            "weighted_mean": failure_penalty,
            "weighted_std": failure_penalty,
            "stability_ratio": 0.0,
            "hard_failure_rate": hard_failure_rate,
        }

    weights_lookup = _dataset_weights(metrics_cfg)
    weights = []
    values = []
    for row in dataset_summary:
        dataset = str(row.get("dataset"))
        values.append(_to_float(row.get("dataset_score")))
        weights.append(weights_lookup.get(dataset, 1.0))
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()

    weighted_mean = float(np.sum(weights_arr * values_arr))
    weighted_std = float(np.sqrt(np.sum(weights_arr * np.square(values_arr - weighted_mean))))
    stability_ratio = float(abs(weighted_mean) / (weighted_std + _to_float(metrics_cfg.get("stability_eps"), EPS)))
    score = float(weighted_mean + weighted_std)

    stability_threshold = _to_float(metrics_cfg.get("stability_threshold"), 1e4)
    stability_penalty_weight = _to_float(metrics_cfg.get("stability_penalty_weight"), 1e6)
    failure_rate_weight = _to_float(metrics_cfg.get("failure_rate_weight"), 1e3)
    stability_penalty = 0.0
    if objective_mode != "unsupervised" and stability_ratio < stability_threshold:
        stability_penalty = (stability_threshold - stability_ratio) * stability_penalty_weight
    failure_rate_penalty = hard_failure_rate * failure_rate_weight
    return {
        "generalization_score": float(score + stability_penalty + failure_rate_penalty),
        "weighted_mean": weighted_mean,
        "weighted_std": weighted_std,
        "stability_ratio": stability_ratio,
        "hard_failure_rate": hard_failure_rate,
    }


def _compute_ms_per_image(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return float("nan")
    runtimes = np.asarray([_to_float(row.get("runtime_sec")) for row in rows], dtype=float)
    if np.isfinite(runtimes).any():
        return float(1000.0 * np.nansum(runtimes) / max(len(rows), 1))
    ms = np.asarray([_to_float(row.get("ms_per_image")) for row in rows], dtype=float)
    if np.isfinite(ms).any():
        return float(np.nanmean(ms))
    return float("nan")


def _method_summary(
    method: str,
    detail: str,
    rows: Sequence[Mapping[str, Any]],
    metrics_cfg: Mapping[str, Any],
    objective_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    failure_penalty = _to_float(metrics_cfg.get("failure_penalty"), 1e6)
    dataset_summary = _compute_dataset_summary(rows, failure_penalty=failure_penalty)
    hard_failure_rate = float(np.mean([_to_float(row.get("is_failure"), 1.0) for row in rows])) if rows else 1.0
    aggregate = _compute_generalization(dataset_summary, metrics_cfg=metrics_cfg, objective_mode=objective_mode, hard_failure_rate=hard_failure_rate)
    failures_failed = int(sum(1 for row in rows if _to_float(row.get("is_failure")) != 0.0))
    failures_total = len(rows)
    summary_row = {
        "method": method,
        "baseline_detail": detail,
        "generalization_score": aggregate["generalization_score"],
        "weighted_mean": aggregate["weighted_mean"],
        "weighted_std": aggregate["weighted_std"],
        "stability_ratio": aggregate["stability_ratio"],
        "hard_failure_rate": aggregate["hard_failure_rate"],
        "ms_per_image": _compute_ms_per_image(rows),
        "failures_failed": failures_failed,
        "failures_total": failures_total,
    }
    return summary_row, dataset_summary


def _method_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stack_id"]))
        lookup[key] = dict(row)
    return lookup


def _normalize_per_stack_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "dataset": str(row.get("dataset", "")),
                "group_id": str(row.get("group_id", "")),
                "stack_id": str(row.get("stack_id", "")),
                "stack_index": _to_int(row.get("stack_index"), 0),
                "total_fitness": _to_float(row.get("total_fitness")),
                "is_failure": _to_float(row.get("is_failure"), 1.0),
                "runtime_sec": _to_float(row.get("runtime_sec")),
                "pred_peak": _to_float(row.get("pred_peak")),
                "penalty_nonfinite": _to_float(row.get("penalty_nonfinite")),
                "penalty_flat": _to_float(row.get("penalty_flat")),
                "penalty_multi_peak": _to_float(row.get("penalty_multi_peak")),
            }
        )
    return out


def _oracle_mapping(per_op_dir: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    per_stack_by_op: Dict[str, List[Dict[str, Any]]] = {}
    dataset_best: Dict[str, Tuple[float, str]] = {}

    for op_dir in sorted(per_op_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        per_stack_path = op_dir / "per_stack_results_eval.csv"
        if not per_stack_path.exists():
            continue
        rows = _normalize_per_stack_rows(_read_csv(per_stack_path))
        per_stack_by_op[op_dir.name] = rows
        dataset_summary = _compute_dataset_summary(rows, failure_penalty=1e6)
        for entry in dataset_summary:
            dataset = str(entry["dataset"])
            candidate = (float(entry["dataset_score"]), op_dir.name)
            current = dataset_best.get(dataset)
            if current is None or candidate[0] < current[0]:
                dataset_best[dataset] = candidate

    mapping = {dataset: choice[1] for dataset, choice in sorted(dataset_best.items())}
    return mapping, per_stack_by_op


def _assemble_oracle_rows(
    composite_rows: Sequence[Mapping[str, Any]],
    per_stack_by_op: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset_oracle_ops: Mapping[str, str],
) -> List[Dict[str, Any]]:
    per_op_lookup = {op_name: _method_lookup(rows) for op_name, rows in per_stack_by_op.items()}
    assembled: List[Dict[str, Any]] = []
    for row in composite_rows:
        dataset = str(row["dataset"])
        key = (dataset, str(row["stack_id"]))
        op_name = dataset_oracle_ops[dataset]
        selected = per_op_lookup[op_name].get(key)
        if selected is None:
            raise KeyError(f"Missing oracle row for dataset={dataset}, stack_id={row['stack_id']}, op={op_name}")
        assembled.append(dict(selected))
    return assembled


def _build_selected_per_stack(
    composite_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    fixed_op_name: str,
    oracle_ops: Mapping[str, str],
) -> List[Dict[str, Any]]:
    fixed_lookup = _method_lookup(fixed_rows)
    oracle_lookup = _method_lookup(oracle_rows)
    selected: List[Dict[str, Any]] = []
    for row in composite_rows:
        key = (str(row["dataset"]), str(row["stack_id"]))
        fixed_row = fixed_lookup[key]
        oracle_row = oracle_lookup[key]
        composite_value = _to_float(row["total_fitness"])
        fixed_value = _to_float(fixed_row["total_fitness"])
        oracle_value = _to_float(oracle_row["total_fitness"])
        selected.append(
            {
                "dataset": key[0],
                "stack_id": key[1],
                "stack_index": row["stack_index"],
                "fixed_single_op": fixed_op_name,
                "oracle_single_op": oracle_ops[key[0]],
                "composite_total_fitness": composite_value,
                "fixed_single_total_fitness": fixed_value,
                "oracle_single_total_fitness": oracle_value,
                "delta_fixed_minus_composite": fixed_value - composite_value,
                "delta_oracle_minus_composite": oracle_value - composite_value,
                "composite_beats_fixed": int(composite_value < fixed_value),
                "composite_beats_oracle": int(composite_value < oracle_value),
            }
        )
    return selected


def _rank_single_ops(
    per_op_summary_rows: Sequence[Mapping[str, str]],
    composite_score: float,
    composite_ms: float,
    oracle_ops: Mapping[str, str],
    fixed_op_name: str,
) -> List[Dict[str, Any]]:
    oracle_sets: Dict[str, List[str]] = {}
    for dataset, op_name in oracle_ops.items():
        oracle_sets.setdefault(op_name, []).append(dataset)

    ranked: List[Dict[str, Any]] = []
    for raw in per_op_summary_rows:
        ranked.append(
            {
                "op_name": str(raw.get("op_name", "")),
                "best_value": _to_float(raw.get("best_value")),
                "ms_per_image": _to_float(raw.get("ms_per_image")),
                "failures_failed": _to_int(raw.get("failures_failed")),
                "failures_total": _to_int(raw.get("failures_total")),
                "objective_mode": str(raw.get("objective_mode", "")),
                "best_params_json": str(raw.get("best_params_json", "")),
                "delta_vs_composite": _to_float(raw.get("best_value")) - composite_score,
                "runtime_ratio_vs_composite": _to_float(raw.get("ms_per_image")) / composite_ms if composite_ms and not math.isnan(composite_ms) else float("nan"),
                "is_fixed_single": int(str(raw.get("op_name", "")) == fixed_op_name),
                "oracle_datasets": ",".join(sorted(oracle_sets.get(str(raw.get("op_name", "")), []))),
            }
        )
    ranked.sort(key=lambda item: (item["best_value"], item["op_name"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _pipeline_rows(best_pipeline: Mapping[str, Any], tuned_params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    steps = best_pipeline.get("steps", [])
    rows: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        op_name = str(step.get("name", ""))
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        tuned_match = op_name in tuned_params and _short_json(params) == _short_json(tuned_params[op_name])
        rows.append(
            {
                "step_index": idx,
                "category": str(step.get("category", "")),
                "op_name": op_name,
                "is_identity": int(op_name == "identity"),
                "enabled": int(bool(step.get("enabled", True))),
                "params_json": _short_json(params),
                "params_compact": _compact_params(params),
                "tuned_params_used": int(tuned_match),
            }
        )
    return rows


def _convergence_rows(progress_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    best_so_far = float("inf")
    for raw in progress_rows:
        generation = _to_int(raw.get("generation"))
        best_score = _to_float(raw.get("best_score"))
        mean_score = _to_float(raw.get("mean_score"))
        std_score = _to_float(raw.get("std_score"))
        evaluated = _to_int(raw.get("evaluated"))
        cache_hits = _to_int(raw.get("cache_hits"))
        best_so_far = min(best_so_far, best_score)
        rows.append(
            {
                "generation": generation,
                "best_score": best_score,
                "mean_score": mean_score,
                "std_score": std_score,
                "evaluated": evaluated,
                "cache_hits": cache_hits,
                "best_score_cummin": best_so_far,
            }
        )
    return rows


def _convergence_summary(progress_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not progress_rows:
        return {
            "generations_total": 0,
            "generation_of_best": 0,
            "initial_best_score": float("nan"),
            "final_best_score": float("nan"),
            "absolute_improvement": float("nan"),
            "relative_improvement_pct": float("nan"),
            "mean_evaluated_per_generation": float("nan"),
            "mean_cache_hits_per_generation": float("nan"),
        }
    first = progress_rows[0]
    best = min(progress_rows, key=lambda row: _to_float(row["best_score"]))
    last = progress_rows[-1]
    initial = _to_float(first["best_score"])
    final = _to_float(last["best_score"])
    best_score = _to_float(best["best_score"])
    absolute_improvement = initial - best_score
    denom = abs(initial) if abs(initial) > EPS else float("nan")
    relative_improvement_pct = 100.0 * absolute_improvement / denom if not math.isnan(denom) else float("nan")
    return {
        "generations_total": len(progress_rows),
        "generation_of_best": _to_int(best["generation"]),
        "initial_best_score": initial,
        "final_best_score": final,
        "best_score": best_score,
        "absolute_improvement": absolute_improvement,
        "relative_improvement_pct": relative_improvement_pct,
        "mean_evaluated_per_generation": float(np.mean([_to_float(row["evaluated"]) for row in progress_rows])),
        "mean_cache_hits_per_generation": float(np.mean([_to_float(row["cache_hits"]) for row in progress_rows])),
    }


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" + "r" * max(len(columns) - 1, 0)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\hline"]
    headers = " & ".join(_latex_escape(label) for _, label in columns) + r" \\"
    lines.append(headers)
    lines.append(r"\hline")
    for row in rows:
        rendered = " & ".join(_latex_escape(_format_cell(row.get(key, ""))) for key, _ in columns) + r" \\"
        lines.append(rendered)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.elements: List[str] = []

    def add(self, element: str) -> None:
        self.elements.append(element)

    def rect(self, x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", stroke_width: float = 1.0, rx: float = 0.0) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float = 1.0, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr} />'
        )

    def circle(self, cx: float, cy: float, radius: float, fill: str, stroke: str = "none") -> None:
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" />')

    def polyline(self, points: Sequence[Tuple[float, float]], stroke: str, stroke_width: float = 2.0, fill: str = "none") -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(
            f'<polyline points="{encoded}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" stroke-linejoin="round" stroke-linecap="round" />'
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        size: int = 14,
        fill: str = "#111827",
        anchor: str = "start",
        weight: str = "400",
    ) -> None:
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
            f'font-family="Helvetica, Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}">{escape(content)}</text>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = [
            f'<svg xmlns="{SVG_NS}" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="#f8fafc" />',
            *self.elements,
            "</svg>",
        ]
        path.write_text("\n".join(doc) + "\n", encoding="utf-8")


def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if not math.isfinite(value):
        return range_min
    if math.isclose(domain_min, domain_max):
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _tick_values(low: float, high: float, count: int = 5) -> List[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if math.isclose(low, high):
        return [low]
    ticks = np.linspace(low, high, num=max(count, 2))
    return [float(tick) for tick in ticks]


def _save_main_comparison(summary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    colors = {
        "composite": "#0f766e",
        "fixed_single": "#c2410c",
        "oracle_single": "#4d7c0f",
    }
    labels = {
        "composite": "Composite GP",
        "fixed_single": "Fixed single",
        "oracle_single": "Dataset oracle single",
    }
    width, height = 960, 480
    left, top, chart_w, chart_h = 280, 80, 620, 300
    canvas = SvgCanvas(width, height)
    canvas.text(40, 40, "Main method comparison", size=24, weight="700")
    canvas.text(40, 62, "Generalization score on the held-out evaluation split (lower is better)", size=14, fill="#475569")

    values = [_to_float(row["generalization_score"]) for row in summary_rows]
    domain_min = min(values + [0.0]) * 1.10
    domain_max = max(values + [0.0]) * 1.05
    domain_max = max(domain_max, 0.0)
    zero_x = _scale_linear(0.0, domain_min, domain_max, left, left + chart_w)

    for tick in _tick_values(domain_min, domain_max, count=6):
        x = _scale_linear(tick, domain_min, domain_max, left, left + chart_w)
        canvas.line(x, top, x, top + chart_h, "#dbe4ee", 1.0)
        canvas.text(x, top + chart_h + 28, f"{tick:.2f}", size=11, fill="#334155", anchor="middle")
    canvas.line(zero_x, top, zero_x, top + chart_h, "#475569", 1.5)

    bar_h = 52
    bar_gap = 34
    for idx, row in enumerate(summary_rows):
        y = top + idx * (bar_h + bar_gap)
        method = str(row["method"])
        score = _to_float(row["generalization_score"])
        bar_x = _scale_linear(score, domain_min, domain_max, left, left + chart_w)
        x = min(bar_x, zero_x)
        bar_w = abs(zero_x - bar_x)
        canvas.rect(x, y, bar_w, bar_h, fill=colors.get(method, "#64748b"), rx=8.0)
        label = labels.get(method, method)
        detail = str(row.get("baseline_detail", "") or "")
        if detail and method != "composite":
            label = f"{label} ({detail})"
        canvas.text(40, y + 30, label, size=15, weight="600")
        canvas.text(40, y + 50, f"runtime={_to_float(row['ms_per_image']):.2f} ms/image, failures={_to_int(row['failures_failed'])}/{_to_int(row['failures_total'])}", size=12, fill="#475569")
        anchor = "end" if score <= 0 else "start"
        text_x = x - 10 if score <= 0 else x + bar_w + 10
        canvas.text(text_x, y + 32, f"{score:.3f}", size=13, fill="#111827", anchor=anchor, weight="600")
    canvas.save(path)


def _save_datasetwise_chart(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    colors = {"composite": "#0f766e", "fixed_single": "#c2410c", "oracle_single": "#4d7c0f"}
    width, height = 1080, 620
    left, top, chart_w, chart_h = 330, 90, 690, 430
    canvas = SvgCanvas(width, height)
    canvas.text(40, 40, "Dataset-wise comparison", size=24, weight="700")
    canvas.text(40, 62, "Each row compares the final GP composite against the strongest fixed single baseline and a dataset oracle single baseline", size=14, fill="#475569")

    values: List[float] = []
    for row in rows:
        values.extend(
            [
                _to_float(row["composite_dataset_score"]),
                _to_float(row["fixed_single_dataset_score"]),
                _to_float(row["oracle_single_dataset_score"]),
            ]
        )
    domain_min = min(values + [0.0]) * 1.10
    domain_max = max(values + [0.0]) * 1.05
    domain_max = max(domain_max, 0.0)
    zero_x = _scale_linear(0.0, domain_min, domain_max, left, left + chart_w)

    for tick in _tick_values(domain_min, domain_max, count=6):
        x = _scale_linear(tick, domain_min, domain_max, left, left + chart_w)
        canvas.line(x, top, x, top + chart_h, "#dbe4ee", 1.0)
        canvas.text(x, top + chart_h + 28, f"{tick:.2f}", size=11, fill="#334155", anchor="middle")
    canvas.line(zero_x, top, zero_x, top + chart_h, "#475569", 1.5)

    group_h = chart_h / max(len(rows), 1)
    bar_h = 18
    methods = [
        ("composite", "composite_dataset_score", "Composite GP"),
        ("fixed_single", "fixed_single_dataset_score", "Fixed single"),
        ("oracle_single", "oracle_single_dataset_score", "Oracle single"),
    ]
    for idx, row in enumerate(rows):
        group_y = top + idx * group_h + 10
        dataset = str(row["dataset"])
        canvas.text(40, group_y + 28, dataset, size=16, weight="700")
        canvas.text(40, group_y + 48, f"fixed={row['fixed_single_op']} | oracle={row['oracle_single_op']}", size=12, fill="#475569")
        for method_idx, (method, key, label) in enumerate(methods):
            y = group_y + method_idx * (bar_h + 8)
            score = _to_float(row[key])
            bar_x = _scale_linear(score, domain_min, domain_max, left, left + chart_w)
            x = min(bar_x, zero_x)
            bar_w = abs(zero_x - bar_x)
            canvas.rect(x, y, bar_w, bar_h, fill=colors[method], rx=4.0)
            canvas.text(left - 12, y + 14, label, size=11, fill="#334155", anchor="end")
            anchor = "end" if score <= 0 else "start"
            text_x = x - 8 if score <= 0 else x + bar_w + 8
            canvas.text(text_x, y + 14, f"{score:.3f}", size=11, anchor=anchor)
    canvas.save(path)


def _save_convergence_chart(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    width, height = 1080, 560
    left, top, chart_w, chart_h = 90, 80, 920, 380
    canvas = SvgCanvas(width, height)
    canvas.text(40, 40, "GP convergence", size=24, weight="700")
    canvas.text(40, 62, "Best and population mean scores across generations (lower is better)", size=14, fill="#475569")
    if not rows:
        canvas.text(40, 120, "No gp_progress.csv data found.", size=16, fill="#991b1b")
        canvas.save(path)
        return

    x_min = min(_to_float(row["generation"]) for row in rows)
    x_max = max(_to_float(row["generation"]) for row in rows)
    all_y = [_to_float(row["best_score"]) for row in rows] + [_to_float(row["mean_score"]) for row in rows]
    y_min = min(all_y) * 1.05
    y_max = max(all_y) * 1.05

    for tick in _tick_values(y_min, y_max, count=6):
        y = _scale_linear(tick, y_min, y_max, top + chart_h, top)
        canvas.line(left, y, left + chart_w, y, "#dbe4ee", 1.0)
        canvas.text(left - 12, y + 4, f"{tick:.2f}", size=11, fill="#334155", anchor="end")
    for tick in _tick_values(x_min, x_max, count=6):
        x = _scale_linear(tick, x_min, x_max, left, left + chart_w)
        canvas.line(x, top, x, top + chart_h, "#eef2f7", 1.0)
        canvas.text(x, top + chart_h + 26, f"{int(round(tick))}", size=11, fill="#334155", anchor="middle")

    best_points = []
    mean_points = []
    for row in rows:
        x = _scale_linear(_to_float(row["generation"]), x_min, x_max, left, left + chart_w)
        best_points.append((x, _scale_linear(_to_float(row["best_score"]), y_min, y_max, top + chart_h, top)))
        mean_points.append((x, _scale_linear(_to_float(row["mean_score"]), y_min, y_max, top + chart_h, top)))
    canvas.polyline(mean_points, stroke="#94a3b8", stroke_width=2.0)
    canvas.polyline(best_points, stroke="#0f766e", stroke_width=3.0)
    best_row = min(rows, key=lambda row: _to_float(row["best_score"]))
    best_x = _scale_linear(_to_float(best_row["generation"]), x_min, x_max, left, left + chart_w)
    best_y = _scale_linear(_to_float(best_row["best_score"]), y_min, y_max, top + chart_h, top)
    canvas.circle(best_x, best_y, 5.0, "#0f766e")
    canvas.text(best_x + 10, best_y - 10, f"best @ gen {int(best_row['generation'])}: {_to_float(best_row['best_score']):.3f}", size=12, fill="#0f172a")
    canvas.line(left, top + chart_h, left + chart_w, top + chart_h, "#475569", 1.5)
    canvas.line(left, top, left, top + chart_h, "#475569", 1.5)
    canvas.text(48, top + chart_h / 2, "score", size=12, fill="#334155")
    canvas.text(left + chart_w / 2, top + chart_h + 54, "generation", size=12, fill="#334155", anchor="middle")
    canvas.rect(780, 96, 210, 52, fill="#ffffff", stroke="#cbd5e1", stroke_width=1.0, rx=8.0)
    canvas.line(796, 114, 836, 114, "#0f766e", 3.0)
    canvas.text(846, 118, "best score", size=12)
    canvas.line(796, 132, 836, 132, "#94a3b8", 2.0)
    canvas.text(846, 136, "mean score", size=12)
    canvas.save(path)


def _save_runtime_frontier(rows: Sequence[Mapping[str, Any]], composite_row: Mapping[str, Any], path: Path) -> None:
    width, height = 1080, 600
    left, top, chart_w, chart_h = 90, 80, 920, 400
    canvas = SvgCanvas(width, height)
    canvas.text(40, 40, "Accuracy-runtime frontier", size=24, weight="700")
    canvas.text(40, 62, "Single-operator baselines plus the final GP composite. Lower y is better.", size=14, fill="#475569")
    values_x = [_to_float(row["ms_per_image"]) for row in rows] + [_to_float(composite_row["ms_per_image"])]
    values_y = [_to_float(row["best_value"]) for row in rows] + [_to_float(composite_row["generalization_score"])]
    x_min, x_max = 0.0, max(values_x) * 1.10
    y_min, y_max = min(values_y) * 1.10, max(values_y) * 1.05
    y_max = max(y_max, 0.0)

    for tick in _tick_values(x_min, x_max, count=6):
        x = _scale_linear(tick, x_min, x_max, left, left + chart_w)
        canvas.line(x, top, x, top + chart_h, "#eef2f7", 1.0)
        canvas.text(x, top + chart_h + 26, f"{tick:.0f}", size=11, fill="#334155", anchor="middle")
    for tick in _tick_values(y_min, y_max, count=6):
        y = _scale_linear(tick, y_min, y_max, top + chart_h, top)
        canvas.line(left, y, left + chart_w, y, "#dbe4ee", 1.0)
        canvas.text(left - 12, y + 4, f"{tick:.2f}", size=11, fill="#334155", anchor="end")

    for row in rows:
        x = _scale_linear(_to_float(row["ms_per_image"]), x_min, x_max, left, left + chart_w)
        y = _scale_linear(_to_float(row["best_value"]), y_min, y_max, top + chart_h, top)
        fill = "#64748b"
        radius = 5.0
        if _to_int(row.get("is_fixed_single")) == 1:
            fill = "#c2410c"
            radius = 6.5
        elif str(row.get("oracle_datasets", "")):
            fill = "#4d7c0f"
        canvas.circle(x, y, radius, fill)
        should_label = radius > 6.0 or row["rank"] <= 5 or _to_float(row["best_value"]) <= 0.15
        if should_label:
            canvas.text(x + 8, y - 8, str(row["op_name"]), size=11, fill="#0f172a")

    gp_x = _scale_linear(_to_float(composite_row["ms_per_image"]), x_min, x_max, left, left + chart_w)
    gp_y = _scale_linear(_to_float(composite_row["generalization_score"]), y_min, y_max, top + chart_h, top)
    canvas.rect(gp_x - 7, gp_y - 7, 14, 14, fill="#0f766e", rx=2.0)
    canvas.text(gp_x + 12, gp_y - 10, "Composite GP", size=12, fill="#0f172a", weight="600")
    canvas.line(left, top + chart_h, left + chart_w, top + chart_h, "#475569", 1.5)
    canvas.line(left, top, left, top + chart_h, "#475569", 1.5)
    canvas.text(left + chart_w / 2, top + chart_h + 54, "runtime (ms/image)", size=12, fill="#334155", anchor="middle")
    canvas.text(54, top + chart_h / 2, "generalization score", size=12, fill="#334155")
    canvas.save(path)


def _save_pipeline_workflow(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    width, height = 1260, 340
    canvas = SvgCanvas(width, height)
    canvas.text(40, 40, "Final preprocessing workflow", size=24, weight="700")
    canvas.text(40, 62, "The best GP-discovered pipeline reuses tuned single-operator parameters where available", size=14, fill="#475569")
    if not rows:
        canvas.text(40, 120, "No best_pipeline.json data found.", size=16, fill="#991b1b")
        canvas.save(path)
        return
    left = 40
    top = 110
    box_w = 125
    box_h = 96
    gap = 10
    for idx, row in enumerate(rows):
        x = left + idx * (box_w + gap)
        op_name = str(row["op_name"])
        fill = "#e2e8f0" if op_name == "identity" else "#d1fae5"
        stroke = "#94a3b8" if op_name == "identity" else "#0f766e"
        canvas.rect(x, top, box_w, box_h, fill=fill, stroke=stroke, stroke_width=1.5, rx=12.0)
        canvas.text(x + 12, top + 22, str(row["category"]), size=12, fill="#475569", weight="600")
        canvas.text(x + 12, top + 48, op_name, size=15, fill="#111827", weight="700")
        params_line = str(row.get("params_compact", "") or "default/identity")
        if len(params_line) > 30:
            params_line = params_line[:27] + "..."
        canvas.text(x + 12, top + 72, params_line, size=11, fill="#334155")
        tune_text = "tuned" if _to_int(row.get("tuned_params_used")) == 1 else "fixed"
        canvas.text(x + 12, top + 90, tune_text, size=11, fill="#0f766e" if tune_text == "tuned" else "#64748b", weight="600")
        if idx < len(rows) - 1:
            arrow_x = x + box_w
            canvas.line(arrow_x + 4, top + box_h / 2, arrow_x + gap - 2, top + box_h / 2, "#475569", 2.0)
            canvas.line(arrow_x + gap - 8, top + box_h / 2 - 6, arrow_x + gap - 2, top + box_h / 2, "#475569", 2.0)
            canvas.line(arrow_x + gap - 8, top + box_h / 2 + 6, arrow_x + gap - 2, top + box_h / 2, "#475569", 2.0)
    canvas.save(path)


def _build_readme(
    resolved: ResolvedRun,
    outdir: Path,
    main_summary: Sequence[Mapping[str, Any]],
    convergence_summary: Mapping[str, Any],
    dataset_oracle_ops: Mapping[str, str],
    fixed_single: str,
) -> str:
    by_method = {str(row["method"]): row for row in main_summary}
    composite = by_method["composite"]
    fixed = by_method["fixed_single"]
    oracle = by_method["oracle_single"]
    return "\n".join(
        [
            "# Q1 Paper Output Package",
            "",
            f"- Source artifact run: `{resolved.artifact_dir}`",
            f"- Output package: `{outdir}`",
            "",
            "## Script Audit",
            "- `run_optuna_per_op.py`: tunes each operator independently and writes `optuna_per_op/per_op_summary.csv` plus per-op evaluation artifacts.",
            "- `run_gp.py` and `run_full.py`: run the composite GP search, persist `gp_progress.csv`, `best_pipeline.json`, and the final evaluation CSV/JSON files.",
            "- `make_report.py`: creates a compact run report for one directory, but it is not a publication package.",
            "- `make_tables.py`: currently averages every `per_stack_results*.csv` it finds, so it is too coarse for manuscript tables.",
            "- `make_ablation_table.py`: concatenates `per_op_summary.csv` files across runs, which is useful for audits but not for one finalized study package.",
            "",
            "## Main Findings",
            f"- Final composite GP score: `{_to_float(composite['generalization_score']):.4f}`",
            f"- Best fixed single-op baseline: `{fixed_single}` with score `{_to_float(fixed['generalization_score']):.4f}`",
            f"- Dataset-oracle single baseline score: `{_to_float(oracle['generalization_score']):.4f}`",
            f"- Composite beats the fixed single baseline by `{_to_float(fixed['generalization_score']) - _to_float(composite['generalization_score']):.4f}` score units.",
            f"- GP reached its best score at generation `{_to_int(convergence_summary['generation_of_best'])}` out of `{_to_int(convergence_summary['generations_total'])}` generations.",
            f"- Dataset oracle operator map: `{json.dumps(dataset_oracle_ops, sort_keys=True)}`",
            "",
            "## Output Layout",
            "- `main/`: canonical intermediate summaries for the selected run.",
            "- `tables/`: manuscript-facing CSV and LaTeX tables.",
            "- `figures/`: SVG publication figures.",
            "",
        ]
    ) + "\n"


def build_q1_paper_outputs(run_dir: str | Path, outdir: str | Path | None = None) -> Dict[str, Any]:
    resolved = _resolve_run(run_dir)
    package_dir = Path(outdir).expanduser().resolve() if outdir else _default_outdir(resolved)
    main_dir = package_dir / "main"
    tables_dir = package_dir / "tables"
    figures_dir = package_dir / "figures"
    _write_json(package_dir / "_build_state.json", {"status": "started", "source_run": str(resolved.artifact_dir)})

    cfg = _load_yaml(resolved.artifact_dir / "config_used.yaml")
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics"), dict) else {}
    objective_mode = _objective_mode(cfg)

    gp_per_stack = _normalize_per_stack_rows(_read_csv(resolved.gp_dir / "per_stack_results_eval.csv"))
    gp_progress = _convergence_rows(_read_csv(resolved.gp_dir / "gp_progress.csv"))
    gp_best_pipeline = _load_json(resolved.gp_dir / "best_pipeline.json")
    tuned_params = _load_json(resolved.gp_dir / "tuned_params_used.json")
    run_metadata = _load_json(resolved.artifact_dir / "run_metadata.json")
    per_op_summary_raw = _read_csv(resolved.per_op_dir / "per_op_summary.csv")

    if not per_op_summary_raw:
        raise FileNotFoundError(f"Missing per_op_summary.csv under {resolved.per_op_dir}")

    sorted_per_op = sorted(per_op_summary_raw, key=lambda row: (_to_float(row.get("best_value")), str(row.get("op_name", ""))))
    fixed_single = str(sorted_per_op[0]["op_name"])
    fixed_rows = _normalize_per_stack_rows(_read_csv(resolved.per_op_dir / fixed_single / "per_stack_results_eval.csv"))
    dataset_oracle_ops, per_stack_by_op = _oracle_mapping(resolved.per_op_dir)
    oracle_rows = _assemble_oracle_rows(gp_per_stack, per_stack_by_op, dataset_oracle_ops)

    composite_summary, composite_dataset = _method_summary("composite", "gp_composite", gp_per_stack, metrics_cfg, objective_mode)
    fixed_summary, fixed_dataset = _method_summary("fixed_single", fixed_single, fixed_rows, metrics_cfg, objective_mode)
    oracle_summary, oracle_dataset = _method_summary("oracle_single", "dataset_best_single_op", oracle_rows, metrics_cfg, objective_mode)
    main_summary = [composite_summary, fixed_summary, oracle_summary]

    composite_score = _to_float(composite_summary["generalization_score"])
    composite_ms = _to_float(composite_summary["ms_per_image"])
    for row in main_summary:
        row["delta_vs_composite"] = _to_float(row["generalization_score"]) - composite_score
        row["runtime_ratio_vs_composite"] = _to_float(row["ms_per_image"]) / composite_ms if composite_ms and not math.isnan(composite_ms) else float("nan")

    ranked_single_ops = _rank_single_ops(sorted_per_op, composite_score, composite_ms, dataset_oracle_ops, fixed_single)
    pipeline_rows = _pipeline_rows(gp_best_pipeline, tuned_params)
    convergence_summary = _convergence_summary(gp_progress)

    composite_dataset_map = {str(row["dataset"]): row for row in composite_dataset}
    fixed_dataset_map = {str(row["dataset"]): row for row in fixed_dataset}
    oracle_dataset_map = {str(row["dataset"]): row for row in oracle_dataset}
    datasetwide_rows: List[Dict[str, Any]] = []
    for dataset in sorted(composite_dataset_map):
        comp = composite_dataset_map[dataset]
        fix = fixed_dataset_map[dataset]
        ora = oracle_dataset_map[dataset]
        datasetwide_rows.append(
            {
                "dataset": dataset,
                "n_stacks": comp["n_stacks"],
                "fixed_single_op": fixed_single,
                "oracle_single_op": dataset_oracle_ops[dataset],
                "composite_dataset_score": comp["dataset_score"],
                "fixed_single_dataset_score": fix["dataset_score"],
                "oracle_single_dataset_score": ora["dataset_score"],
                "composite_mean_fitness": comp["mean_fitness"],
                "fixed_single_mean_fitness": fix["mean_fitness"],
                "oracle_single_mean_fitness": ora["mean_fitness"],
                "delta_fixed_minus_composite": fix["dataset_score"] - comp["dataset_score"],
                "delta_oracle_minus_composite": ora["dataset_score"] - comp["dataset_score"],
            }
        )

    selected_per_stack = _build_selected_per_stack(gp_per_stack, fixed_rows, oracle_rows, fixed_single, dataset_oracle_ops)
    win_loss_rows = [
        {
            "comparison": "composite_vs_fixed_single",
            "pct_composite_better": 100.0 * float(np.mean([row["composite_beats_fixed"] for row in selected_per_stack])),
            "pct_composite_worse": 100.0 * (1.0 - float(np.mean([row["composite_beats_fixed"] for row in selected_per_stack]))),
        },
        {
            "comparison": "composite_vs_oracle_single",
            "pct_composite_better": 100.0 * float(np.mean([row["composite_beats_oracle"] for row in selected_per_stack])),
            "pct_composite_worse": 100.0 * (1.0 - float(np.mean([row["composite_beats_oracle"] for row in selected_per_stack]))),
        },
    ]

    _write_csv(main_dir / "paper_summary_main.csv", main_summary)
    _write_csv(main_dir / "paper_summary_datasetwise.csv", datasetwide_rows)
    _write_csv(main_dir / "paper_summary_single_ops.csv", ranked_single_ops)
    _write_csv(main_dir / "paper_summary_pipeline.csv", pipeline_rows)
    _write_csv(main_dir / "paper_summary_convergence.csv", gp_progress)
    _write_csv(main_dir / "paper_summary_perstack_selected.csv", selected_per_stack)
    _write_csv(main_dir / "paper_summary_win_loss.csv", win_loss_rows)

    table_main_path = tables_dir / "Table_01_MainComparison.csv"
    table_dataset_path = tables_dir / "Table_02_DatasetwiseComparison.csv"
    table_pipeline_path = tables_dir / "Table_03_FinalPipeline.csv"
    table_single_path = tables_dir / "Table_04_SingleOperatorRanking.csv"
    table_search_path = tables_dir / "Table_05_SearchRuntimeAndConvergence.csv"

    _write_csv(table_main_path, main_summary)
    _write_csv(table_dataset_path, datasetwide_rows)
    _write_csv(table_pipeline_path, pipeline_rows)
    _write_csv(table_single_path, ranked_single_ops)
    _write_csv(table_search_path, [convergence_summary])

    _write_latex_table(
        table_main_path.with_suffix(".tex"),
        main_summary,
        [
            ("method", "Method"),
            ("baseline_detail", "Detail"),
            ("generalization_score", "Generalization"),
            ("stability_ratio", "Stability"),
            ("ms_per_image", "ms/image"),
            ("failures_failed", "Failures"),
        ],
    )
    _write_latex_table(
        table_dataset_path.with_suffix(".tex"),
        datasetwide_rows,
        [
            ("dataset", "Dataset"),
            ("fixed_single_op", "Fixed single"),
            ("oracle_single_op", "Oracle single"),
            ("composite_dataset_score", "Composite"),
            ("fixed_single_dataset_score", "Fixed"),
            ("oracle_single_dataset_score", "Oracle"),
        ],
    )
    _write_latex_table(
        table_pipeline_path.with_suffix(".tex"),
        pipeline_rows,
        [
            ("step_index", "Step"),
            ("category", "Category"),
            ("op_name", "Operator"),
            ("params_compact", "Parameters"),
            ("tuned_params_used", "Tuned"),
        ],
    )
    _write_latex_table(
        table_single_path.with_suffix(".tex"),
        ranked_single_ops,
        [
            ("rank", "Rank"),
            ("op_name", "Operator"),
            ("best_value", "Score"),
            ("ms_per_image", "ms/image"),
            ("oracle_datasets", "Oracle datasets"),
        ],
    )
    _write_latex_table(
        table_search_path.with_suffix(".tex"),
        [convergence_summary],
        [
            ("generations_total", "Generations"),
            ("generation_of_best", "Best gen"),
            ("initial_best_score", "Initial"),
            ("best_score", "Best"),
            ("relative_improvement_pct", "Improvement %"),
        ],
    )

    fig_main = figures_dir / "Figure_01_MainComparison.svg"
    fig_dataset = figures_dir / "Figure_02_DatasetwiseComparison.svg"
    fig_convergence = figures_dir / "Figure_03_GPConvergence.svg"
    fig_frontier = figures_dir / "Figure_04_RuntimeScoreFrontier.svg"
    fig_pipeline = figures_dir / "Figure_05_FinalPipelineWorkflow.svg"
    _save_main_comparison(main_summary, fig_main)
    _save_datasetwise_chart(datasetwide_rows, fig_dataset)
    _save_convergence_chart(gp_progress, fig_convergence)
    _save_runtime_frontier(ranked_single_ops, composite_summary, fig_frontier)
    _save_pipeline_workflow(pipeline_rows, fig_pipeline)

    readme = _build_readme(resolved, package_dir, main_summary, convergence_summary, dataset_oracle_ops, fixed_single)
    _write_text(package_dir / "README.md", readme)

    manifest = {
        "source_run": str(resolved.artifact_dir),
        "output_dir": str(package_dir),
        "objective_mode": objective_mode,
        "fixed_single_op": fixed_single,
        "dataset_oracle_ops": dataset_oracle_ops,
        "run_metadata": {
            "config_hash": run_metadata.get("config_hash"),
            "git_hash": run_metadata.get("git_hash"),
            "opencv_available": run_metadata.get("opencv_available"),
            "os": run_metadata.get("os"),
        },
        "main_findings": {
            "composite_generalization_score": composite_summary["generalization_score"],
            "fixed_single_generalization_score": fixed_summary["generalization_score"],
            "oracle_single_generalization_score": oracle_summary["generalization_score"],
            "best_generation": convergence_summary["generation_of_best"],
        },
        "tables": [
            str(table_main_path),
            str(table_dataset_path),
            str(table_pipeline_path),
            str(table_single_path),
            str(table_search_path),
        ],
        "figures": [
            str(fig_main),
            str(fig_dataset),
            str(fig_convergence),
            str(fig_frontier),
            str(fig_pipeline),
        ],
    }
    _write_json(main_dir / "paper_summary_manifest.json", manifest)
    _write_json(package_dir / "_build_state.json", {"status": "completed", "source_run": str(resolved.artifact_dir), "output_dir": str(package_dir)})
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_q1_paper_outputs(args.run_dir, args.outdir)
    print(f"Q1_PAPER_OUTPUTS_DONE: {manifest['output_dir']}")


if __name__ == "__main__":
    main()

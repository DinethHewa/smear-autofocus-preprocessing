from __future__ import annotations

import csv
import copy
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..preprocess.ops import is_op_available
from ..preprocess.param_spaces import DEFAULT_PARAMS
from ..preprocess.pipeline import Pipeline
from .aggregation import DEFAULT_METRIC_WEIGHTS, compute_rank_based_summary, compute_value_based_summary, dataset_metric_summary_from_per_stack
from .workflows import evaluate_methods, pipeline_from_steps


@dataclass(frozen=True)
class Q1GPResult:
    best_genome: List[str]
    best_score: float
    best_pipeline: Pipeline
    best_pipeline_json: str
    progress_csv: Path
    best_by_generation_csv: Path
    candidate_summary_csv: Path
    state_json: Path


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonify_rng_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonify_rng_state(item) for item in value]
    return value


def _tupleize_rng_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize_rng_state(item) for item in value)
    return value


def _available_gp_ops(config: Mapping[str, Any], opt_cfg: Mapping[str, Any] | None = None) -> List[str]:
    opt_cfg = opt_cfg or {}
    workflows = config.get("workflows", {}) if isinstance(config.get("workflows"), dict) else {}
    if opt_cfg.get("operator_allowlist"):
        configured = [str(name) for name in opt_cfg.get("operator_allowlist", []) or []]
    else:
        configured = [str(name) for name in workflows.get("single_ops", []) or []]
    excluded = {str(name) for name in opt_cfg.get("operator_exclude", []) or []}
    ops = ["identity", *configured]
    out: List[str] = []
    for op_name in ops:
        if op_name in excluded:
            continue
        if op_name in out:
            continue
        if is_op_available(op_name):
            out.append(op_name)
    if not out:
        raise ValueError("No available preprocessing operators for Q1 GP search")
    return out


def _focus_backend_uses_cuda(config: Mapping[str, Any]) -> bool:
    focus_cfg = config.get("focus_measure", {}) if isinstance(config.get("focus_measure"), dict) else {}
    backend = str(focus_cfg.get("backend", focus_cfg.get("curve_backend", "cpu"))).strip().lower()
    return backend in {"cuda", "torch_cuda", "torch_cuda_auto"}


def _normalize_genome(genome: Sequence[str], *, max_length: int) -> List[str]:
    clean = [str(op) for op in genome if str(op)]
    if not clean:
        clean = ["identity"]
    clean = clean[: max(1, int(max_length))]
    if all(op == "identity" for op in clean):
        return ["identity"]
    return [op for op in clean if op != "identity"] or ["identity"]


def genome_to_q1_pipeline(genome: Sequence[str], *, max_length: int) -> Pipeline:
    steps = []
    for idx, op_name in enumerate(_normalize_genome(genome, max_length=max_length), start=1):
        steps.append(
            {
                "name": op_name,
                "enabled": True,
                "category": f"gp_step_{idx}",
                "params": dict(DEFAULT_PARAMS.get(op_name, {})),
            }
        )
    return pipeline_from_steps(steps)


def method_entry_from_pipeline(method: str, pipeline: Pipeline, *, method_type: str = "q1_gp_workflow") -> Dict[str, Any]:
    pipeline_json = json.dumps(pipeline.to_dict(), sort_keys=True, default=str)
    return {
        "method": method,
        "method_type": method_type,
        "pipeline": pipeline,
        "pipeline_json": pipeline_json,
    }


def pipeline_from_pipeline_json(pipeline_json: str) -> Pipeline:
    payload = json.loads(str(pipeline_json))
    return Pipeline.from_dict(payload).resolved_defaults()


def _candidate_id(pipeline: Pipeline) -> str:
    return f"gp::{pipeline.fingerprint()[:16]}"


def _random_genome(rng: random.Random, ops: Sequence[str], max_length: int) -> List[str]:
    length = rng.randint(1, max(1, int(max_length)))
    return _normalize_genome([rng.choice(list(ops)) for _ in range(length)], max_length=max_length)


def _mutate(rng: random.Random, genome: Sequence[str], ops: Sequence[str], max_length: int, mutation_rate: float) -> List[str]:
    child = list(genome)
    if not child:
        child = ["identity"]
    for idx in range(len(child)):
        if rng.random() < mutation_rate:
            child[idx] = rng.choice(list(ops))
    if rng.random() < mutation_rate and len(child) < max_length:
        child.insert(rng.randint(0, len(child)), rng.choice(list(ops)))
    if rng.random() < mutation_rate and len(child) > 1:
        del child[rng.randrange(len(child))]
    return _normalize_genome(child, max_length=max_length)


def _crossover(rng: random.Random, a: Sequence[str], b: Sequence[str], max_length: int) -> tuple[List[str], List[str]]:
    if len(a) <= 1 or len(b) <= 1:
        return list(a), list(b)
    ia = rng.randint(1, len(a) - 1)
    ib = rng.randint(1, len(b) - 1)
    child_a = list(a[:ia]) + list(b[ib:])
    child_b = list(b[:ib]) + list(a[ia:])
    return _normalize_genome(child_a, max_length=max_length), _normalize_genome(child_b, max_length=max_length)


def _candidate_summary_fieldnames(example: Mapping[str, Any]) -> List[str]:
    keys = ["candidate_id", "pipeline_fingerprint", "genome_json", "pipeline_json"]
    for key in example:
        if key not in keys:
            keys.append(key)
    return keys


def _load_candidate_cache(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_csv(path):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id:
            cache.setdefault(candidate_id, []).append(dict(row))
    return cache


def _remap_summary_method(rows: Sequence[Mapping[str, Any]], method: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        mapped = dict(row)
        mapped["method"] = method
        out.append(mapped)
    return out


def _filter_manifest_for_search(
    manifest_df: pd.DataFrame,
    *,
    datasets: Sequence[str] | None,
    outer_split: Sequence[str] | None,
    inner_split: Sequence[str] | None,
) -> pd.DataFrame:
    df = manifest_df.copy()
    if datasets is not None:
        df = df[df["dataset"].isin(list(datasets))].copy()
    if outer_split is not None:
        df = df[df["outer_split"].isin(list(outer_split))].copy()
    if inner_split is not None:
        df = df[df["inner_split"].isin(list(inner_split))].copy()
    if df.empty:
        raise ValueError("No manifest rows left after applying Q1 GP search filters")
    return df.reset_index(drop=True)


def _sample_manifest_for_search(manifest_df: pd.DataFrame, *, max_stacks_per_dataset: int | None, seed: int) -> pd.DataFrame:
    if max_stacks_per_dataset is None or int(max_stacks_per_dataset) <= 0:
        return manifest_df.reset_index(drop=True)
    max_count = int(max_stacks_per_dataset)
    sampled: List[pd.DataFrame] = []
    for offset, (dataset, subset) in enumerate(sorted(manifest_df.groupby("dataset"), key=lambda item: str(item[0]))):
        if len(subset) <= max_count:
            sampled.append(subset)
            continue
        if "group_id" in subset.columns:
            group_ids = sorted(subset["group_id"].astype(str).unique().tolist())
            rng = np.random.default_rng(int(seed) + offset)
            rng.shuffle(group_ids)
            selected_groups: list[str] = []
            total_rows = 0
            for group_id in group_ids:
                selected_groups.append(group_id)
                total_rows += int((subset["group_id"].astype(str) == group_id).sum())
                if total_rows >= max_count:
                    break
            sampled_subset = subset[subset["group_id"].astype(str).isin(selected_groups)].head(max_count)
        else:
            sampled_subset = subset.sample(n=max_count, random_state=int(seed) + offset)
        sampled.append(sampled_subset)
    return pd.concat(sampled, ignore_index=True).sort_values(["dataset", "stack_id"]).reset_index(drop=True)


def _read_completed_stack_ids(path: Path) -> set[str]:
    return {str(row.get("stack_id", "")) for row in _read_csv(path) if row.get("stack_id")}


def _dedupe_rows_by_stack(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_stack: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        stack_id = str(row.get("stack_id", ""))
        if not stack_id:
            continue
        if stack_id not in by_stack:
            order.append(stack_id)
        by_stack[stack_id] = dict(row)
    return [by_stack[stack_id] for stack_id in order]


def _candidate_chunk_worker(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    chunk = pd.DataFrame(payload["records"])
    per_stack, _, _ = evaluate_methods(
        manifest_df=chunk,
        labels_df=payload["labels_df"],
        methods=[payload["method_entry"]],
        config=payload["config"],
        datasets=None,
        outer_split=None,
        inner_split=None,
        write_curves=False,
        compute_metrics=True,
    )
    return per_stack


def _evaluate_candidate_dataset_summary(
    *,
    manifest_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    method_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    candidate_metric_csv: Path,
    datasets: Sequence[str] | None,
    outer_split: Sequence[str] | None,
    inner_split: Sequence[str] | None,
    chunk_size: int,
    num_workers: int,
    resume: bool,
    progress_cb: Callable[[Mapping[str, Any]], None] | None,
    progress_base: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    eval_manifest = _filter_manifest_for_search(
        manifest_df,
        datasets=datasets,
        outer_split=outer_split,
        inner_split=inner_split,
    )
    if not resume and candidate_metric_csv.exists():
        candidate_metric_csv.unlink()
    completed_stack_ids = _read_completed_stack_ids(candidate_metric_csv) if resume and candidate_metric_csv.exists() else set()
    total = int(len(eval_manifest))
    completed = len(completed_stack_ids)
    chunk_size = max(1, int(chunk_size))
    if progress_cb is not None:
        progress_cb(
            {
                **dict(progress_base),
                "event": "candidate_started",
                "completed_stack_method_pairs": completed,
                "total_stack_method_pairs": total,
                "progress_percent": 100.0 * completed / total if total else 100.0,
                "candidate_metric_csv": str(candidate_metric_csv),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    pending = eval_manifest[~eval_manifest["stack_id"].astype(str).isin(completed_stack_ids)].copy()
    chunks = [pending.iloc[start : start + chunk_size].copy() for start in range(0, len(pending), chunk_size)]

    def handle_result(chunk: pd.DataFrame, per_stack: Sequence[Mapping[str, Any]]) -> None:
        nonlocal completed
        if per_stack:
            _append_rows(candidate_metric_csv, per_stack, list(per_stack[0].keys()))
        for row in per_stack:
            completed_stack_ids.add(str(row["stack_id"]))
        completed = len(completed_stack_ids)
        if progress_cb is not None:
            progress_cb(
                {
                    **dict(progress_base),
                    "event": "candidate_chunk",
                    "completed_stack_method_pairs": completed,
                    "total_stack_method_pairs": total,
                    "progress_percent": 100.0 * completed / total if total else 100.0,
                    "current_dataset": str(chunk.iloc[-1].get("dataset", "")) if not chunk.empty else "",
                    "candidate_metric_csv": str(candidate_metric_csv),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

    if int(num_workers) > 1 and len(chunks) > 1:
        payloads = [
            {
                "records": chunk.to_dict("records"),
                "labels_df": labels_df,
                "method_entry": method_entry,
                "config": config,
            }
            for chunk in chunks
        ]
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            for chunk, per_stack in zip(chunks, executor.map(_candidate_chunk_worker, payloads)):
                handle_result(chunk, per_stack)
    else:
        for chunk in chunks:
            per_stack = _candidate_chunk_worker(
                {
                    "records": chunk.to_dict("records"),
                    "labels_df": labels_df,
                    "method_entry": method_entry,
                    "config": config,
                }
            )
            handle_result(chunk, per_stack)
    if len(completed_stack_ids) < total:
        raise RuntimeError(
            f"Q1 GP candidate incomplete: {len(completed_stack_ids)}/{total} stack-method rows. "
            f"Resume from {candidate_metric_csv} with --resume."
        )
    metric_rows = _dedupe_rows_by_stack(_read_csv(candidate_metric_csv))
    if len(metric_rows) < total:
        raise RuntimeError(
            f"Q1 GP candidate checkpoint has {len(metric_rows)}/{total} unique stack rows: {candidate_metric_csv}"
        )
    return dataset_metric_summary_from_per_stack(metric_rows)


def _score_generation(
    dataset_summary_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    agg_cfg = config.get("aggregation", {}) if isinstance(config.get("aggregation"), dict) else {}
    alpha = float(agg_cfg.get("alpha", 0.7))
    metric_weights = dict(agg_cfg.get("metric_weights") or DEFAULT_METRIC_WEIGHTS)
    rank_rows, _ = compute_rank_based_summary(dataset_summary_rows, alpha=alpha)
    value_rows = compute_value_based_summary(
        dataset_summary_rows,
        metric_weights=metric_weights,
        alpha=alpha,
        weighting_mode=str(agg_cfg.get("dataset_weighting", "equal_dataset")),
    )
    by_rank = {str(row["method"]): row for row in rank_rows}
    by_value = {str(row["method"]): row for row in value_rows}
    methods = set(by_rank) | set(by_value)
    scores: Dict[str, Dict[str, float]] = {}
    for method in methods:
        scores[method] = {
            "rank_score": float(by_rank.get(method, {}).get("rank_generalization_score", float("inf"))),
            "value_score": float(by_value.get(method, {}).get("value_generalization_score", float("inf"))),
            "rank": float(by_rank.get(method, {}).get("final_rank", float("inf"))),
        }
    return scores


def run_q1_gp_search(
    *,
    manifest_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: Mapping[str, Any],
    opt_cfg: Mapping[str, Any],
    search_dir: Path,
    search_name: str,
    seed: int,
    datasets: Sequence[str] | None,
    outer_split: Sequence[str] | None,
    inner_split: Sequence[str] | None,
    resume: bool,
    progress_cb: Callable[[Mapping[str, Any]], None] | None = None,
) -> Q1GPResult:
    search_dir.mkdir(parents=True, exist_ok=True)
    generations = int(opt_cfg.get("gp_generations", opt_cfg.get("generations", 10)))
    population_size = int(opt_cfg.get("population_size", 20))
    elite_size = int(opt_cfg.get("elite_size", 2))
    mutation_rate = float(opt_cfg.get("mutation_rate", 0.2))
    crossover_rate = float(opt_cfg.get("crossover_rate", 0.8))
    max_length = int(opt_cfg.get("max_pipeline_length", 5))
    output_cfg = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    candidate_chunk_size = int(
        opt_cfg.get(
            "candidate_chunk_stack_count",
            output_cfg.get("gp_candidate_chunk_stack_count", 64),
        )
        or 10
    )
    requested_num_workers = int(opt_cfg.get("candidate_num_workers", output_cfg.get("gp_candidate_num_workers", 1)) or 1)
    num_workers = requested_num_workers
    cuda_serial_policy = False
    if _focus_backend_uses_cuda(config) and num_workers > 1:
        # Forked Python workers can deadlock after Torch/CUDA runtime initialization.
        # Keep CUDA candidate evaluation serial and rely on batched GPU kernels inside
        # each focus-curve calculation instead of process-level parallelism.
        num_workers = 1
        cuda_serial_policy = True
    max_stacks_per_dataset_raw = opt_cfg.get("search_max_stacks_per_dataset", output_cfg.get("gp_search_max_stacks_per_dataset"))
    max_stacks_per_dataset = int(max_stacks_per_dataset_raw) if max_stacks_per_dataset_raw not in (None, "", 0, "0") else None
    search_compute_rrmse = bool(opt_cfg.get("search_compute_rrmse", True))
    search_config = copy.deepcopy(dict(config))
    if not search_compute_rrmse:
        stats_cfg = search_config.get("statistics", {}) if isinstance(search_config.get("statistics"), dict) else {}
        stats_cfg = dict(stats_cfg)
        stats_cfg["compute_rrmse"] = False
        search_config["statistics"] = stats_cfg
    if generations <= 0 or population_size <= 0:
        raise ValueError("Q1 GP generations and population_size must be positive")
    elite_size = max(1, min(elite_size, population_size))

    filtered_manifest = _filter_manifest_for_search(
        manifest_df,
        datasets=datasets,
        outer_split=outer_split,
        inner_split=inner_split,
    )
    search_manifest = _sample_manifest_for_search(
        filtered_manifest,
        max_stacks_per_dataset=max_stacks_per_dataset,
        seed=seed,
    )
    search_stack_counts = {str(dataset): int(len(subset)) for dataset, subset in search_manifest.groupby("dataset")}

    progress_csv = search_dir / "gp_progress.csv"
    best_csv = search_dir / "gp_best_by_generation.csv"
    candidate_summary_csv = search_dir / "gp_candidate_dataset_metrics.csv"
    candidate_metrics_dir = search_dir / "candidate_metrics"
    metadata_path = search_dir / "gp_search_metadata.json"
    state_json = search_dir / "gp_state.json"
    best_pipeline_path = search_dir / "best_pipeline.json"
    best_score_path = search_dir / "best_score.json"
    if not resume:
        for path in (progress_csv, best_csv, candidate_summary_csv, metadata_path, state_json, best_pipeline_path, best_score_path):
            if path.exists():
                path.unlink()
        if candidate_metrics_dir.exists():
            shutil.rmtree(candidate_metrics_dir)
    candidate_metrics_dir.mkdir(parents=True, exist_ok=True)
    search_metadata = {
        "search_name": search_name,
        "search_stack_counts": search_stack_counts,
        "search_max_stacks_per_dataset": max_stacks_per_dataset,
        "search_compute_rrmse": search_compute_rrmse,
        "candidate_chunk_stack_count": candidate_chunk_size,
        "candidate_num_workers": num_workers,
        "candidate_num_workers_requested": requested_num_workers,
        "cuda_serial_worker_policy": cuda_serial_policy,
    }
    if resume:
        if metadata_path.exists():
            previous_metadata = _read_json(metadata_path)
            comparable_keys = [
                "search_stack_counts",
                "search_max_stacks_per_dataset",
                "search_compute_rrmse",
            ]
            mismatches = {
                key: {"previous": previous_metadata.get(key), "current": search_metadata.get(key)}
                for key in comparable_keys
                if previous_metadata.get(key) != search_metadata.get(key)
            }
            if mismatches:
                raise RuntimeError(
                    f"Q1 GP resume metadata mismatch for {search_name}: {mismatches}. "
                    "Use --overwrite when changing GP search sample or metric budget."
                )
        elif candidate_summary_csv.exists() or any(candidate_metrics_dir.glob("*.csv")):
            raise RuntimeError(
                f"Q1 GP resume found old cache files without metadata in {search_dir}. "
                "Use --overwrite to restart with the current accelerated search budget."
            )
    _write_json(metadata_path, search_metadata)

    ops = _available_gp_ops(config, opt_cfg)
    rng = random.Random(int(seed))
    candidate_cache = _load_candidate_cache(candidate_summary_csv) if resume else {}
    state = _read_json(state_json) if resume else {}
    if state:
        previous_counts = state.get("search_stack_counts")
        if previous_counts and dict(previous_counts) != search_stack_counts:
            raise RuntimeError(
                f"Q1 GP resume metadata mismatch for {search_name}. Existing search_stack_counts={previous_counts}, "
                f"current={search_stack_counts}. Start a new run or use --overwrite for the changed search budget."
            )
        population = [list(genome) for genome in state.get("next_population", [])]
        completed_generation = int(state.get("completed_generation", 0))
        best_genome = list(state.get("best_genome", []))
        best_score = float(state.get("best_score", float("inf")))
        if isinstance(state.get("rng_state"), list):
            rng.setstate(_tupleize_rng_state(state["rng_state"]))
    else:
        population = [_random_genome(rng, ops, max_length) for _ in range(population_size)]
        completed_generation = 0
        best_genome = []
        best_score = float("inf")

    if not population:
        population = [_random_genome(rng, ops, max_length) for _ in range(population_size)]
    population = population[:population_size]
    while len(population) < population_size:
        population.append(_random_genome(rng, ops, max_length))

    progress_fieldnames = [
        "search_name",
        "generation",
        "generations",
        "best_score",
        "best_candidate_id",
        "best_genome_json",
        "best_pipeline_json",
        "mean_score",
        "std_score",
        "evaluated",
        "cache_hits",
        "timestamp_utc",
    ]
    best_fieldnames = ["search_name", "generation", "best_score", "genome_json", "pipeline_json", "candidate_id"]

    for generation in range(completed_generation + 1, generations + 1):
        generation_dataset_summaries: List[Dict[str, Any]] = []
        generation_candidates: Dict[str, Dict[str, Any]] = {}
        evaluated = 0
        cache_hits = 0
        for candidate_index, genome in enumerate(population, start=1):
            pipeline = genome_to_q1_pipeline(genome, max_length=max_length)
            cid = _candidate_id(pipeline)
            pipeline_json = json.dumps(pipeline.to_dict(), sort_keys=True, default=str)
            generation_candidates[cid] = {
                "genome": _normalize_genome(genome, max_length=max_length),
                "pipeline": pipeline,
                "pipeline_json": pipeline_json,
            }
            if cid in candidate_cache:
                cache_hits += 1
                generation_dataset_summaries.extend(_remap_summary_method(candidate_cache[cid], cid))
                if progress_cb is not None:
                    progress_cb(
                        {
                            "event": "candidate",
                            "search_name": search_name,
                            "generation": generation,
                            "generations": generations,
                            "candidate_index": candidate_index,
                            "population_size": population_size,
                            "candidate_id": cid,
                            "evaluated": evaluated,
                            "cache_hits": cache_hits,
                            "cached": True,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                )
                continue
            method_entry = method_entry_from_pipeline(cid, pipeline)
            dataset_summary = _evaluate_candidate_dataset_summary(
                manifest_df=search_manifest,
                labels_df=labels_df,
                method_entry=method_entry,
                config=search_config,
                candidate_metric_csv=candidate_metrics_dir / f"{cid}.csv",
                datasets=None,
                outer_split=None,
                inner_split=None,
                chunk_size=candidate_chunk_size,
                num_workers=num_workers,
                resume=resume,
                progress_cb=progress_cb,
                progress_base={
                    "search_name": search_name,
                    "generation": generation,
                    "generations": generations,
                    "candidate_index": candidate_index,
                    "population_size": population_size,
                    "candidate_id": cid,
                    "evaluated": evaluated,
                    "cache_hits": cache_hits,
                    "cached": False,
                    "search_stack_counts": search_stack_counts,
                    "search_compute_rrmse": search_compute_rrmse,
                },
            )
            rows_to_store = []
            for row in dataset_summary:
                stored = {
                    "candidate_id": cid,
                    "pipeline_fingerprint": pipeline.fingerprint(),
                    "genome_json": json.dumps(_normalize_genome(genome, max_length=max_length)),
                    "pipeline_json": pipeline_json,
                    **dict(row),
                    "method": cid,
                }
                rows_to_store.append(stored)
            if rows_to_store:
                _append_rows(candidate_summary_csv, rows_to_store, _candidate_summary_fieldnames(rows_to_store[0]))
            candidate_cache[cid] = rows_to_store
            generation_dataset_summaries.extend(_remap_summary_method(rows_to_store, cid))
            evaluated += 1
            if progress_cb is not None:
                progress_cb(
                    {
                        "event": "candidate",
                        "search_name": search_name,
                        "generation": generation,
                        "generations": generations,
                        "candidate_index": candidate_index,
                        "population_size": population_size,
                        "candidate_id": cid,
                        "evaluated": evaluated,
                        "cache_hits": cache_hits,
                        "cached": False,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

        scores = _score_generation(generation_dataset_summaries, config)
        scored_candidates = [
            (cid, values["rank_score"], values["value_score"])
            for cid, values in scores.items()
            if cid in generation_candidates
        ]
        if not scored_candidates:
            raise RuntimeError(f"Q1 GP generation {generation} produced no scored candidates")
        scored_candidates.sort(key=lambda item: (item[1], item[2], item[0]))
        gen_best_id, gen_best_score, _ = scored_candidates[0]
        gen_scores = np.asarray([score for _, score, _ in scored_candidates], dtype=float)
        gen_best = generation_candidates[gen_best_id]
        if gen_best_score < best_score:
            best_score = float(gen_best_score)
            best_genome = list(gen_best["genome"])
            best_pipeline = gen_best["pipeline"]
            best_pipeline_path.write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")
            best_score_path.write_text(json.dumps({"best_score": best_score, "candidate_id": gen_best_id}, indent=2), encoding="utf-8")
        else:
            best_pipeline = genome_to_q1_pipeline(best_genome, max_length=max_length) if best_genome else gen_best["pipeline"]

        progress_row = {
            "search_name": search_name,
            "generation": generation,
            "generations": generations,
            "best_score": float(gen_best_score),
            "best_candidate_id": gen_best_id,
            "best_genome_json": json.dumps(gen_best["genome"]),
            "best_pipeline_json": gen_best["pipeline_json"],
            "mean_score": float(np.nanmean(gen_scores)) if gen_scores.size else float("nan"),
            "std_score": float(np.nanstd(gen_scores)) if gen_scores.size else float("nan"),
            "evaluated": evaluated,
            "cache_hits": cache_hits,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_rows(progress_csv, [progress_row], progress_fieldnames)
        _append_rows(
            best_csv,
            [
                {
                    "search_name": search_name,
                    "generation": generation,
                    "best_score": float(best_score),
                    "genome_json": json.dumps(best_genome),
                    "pipeline_json": json.dumps(best_pipeline.to_dict(), sort_keys=True, default=str),
                    "candidate_id": _candidate_id(best_pipeline),
                }
            ],
            best_fieldnames,
        )
        if progress_cb is not None:
            progress_cb({"event": "generation", **progress_row})

        ranked = [generation_candidates[cid]["genome"] for cid, _, _ in scored_candidates]
        next_population = [list(genome) for genome in ranked[:elite_size]]
        while len(next_population) < population_size:
            parent_a = rng.choice(ranked[: max(elite_size, min(len(ranked), 5))])
            parent_b = rng.choice(ranked[: max(elite_size, min(len(ranked), 5))])
            child_a, child_b = list(parent_a), list(parent_b)
            if rng.random() < crossover_rate:
                child_a, child_b = _crossover(rng, parent_a, parent_b, max_length)
            child_a = _mutate(rng, child_a, ops, max_length, mutation_rate)
            next_population.append(child_a)
            if len(next_population) < population_size:
                child_b = _mutate(rng, child_b, ops, max_length, mutation_rate)
                next_population.append(child_b)
        population = next_population[:population_size]
        _write_json(
            state_json,
            {
                "search_name": search_name,
                "completed_generation": generation,
                "generations": generations,
                "population_size": population_size,
                "next_population": population,
                "best_genome": best_genome,
                "best_score": best_score,
                "best_pipeline_json": json.dumps(best_pipeline.to_dict(), sort_keys=True, default=str),
                "rng_state": _jsonify_rng_state(rng.getstate()),
                "search_stack_counts": search_stack_counts,
                "search_max_stacks_per_dataset": max_stacks_per_dataset,
                "search_compute_rrmse": search_compute_rrmse,
                "candidate_chunk_stack_count": candidate_chunk_size,
                "candidate_num_workers": num_workers,
                "candidate_num_workers_requested": requested_num_workers,
                "cuda_serial_worker_policy": cuda_serial_policy,
                "candidate_summary_csv": str(candidate_summary_csv),
                "progress_csv": str(progress_csv),
            },
        )

    if not best_genome:
        final_state = _read_json(state_json)
        best_genome = list(final_state.get("best_genome", []))
        best_score = float(final_state.get("best_score", float("inf")))
    best_pipeline = genome_to_q1_pipeline(best_genome, max_length=max_length)
    best_pipeline_json = json.dumps(best_pipeline.to_dict(), sort_keys=True, default=str)
    best_pipeline_path.write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")
    best_score_path.write_text(json.dumps({"best_score": best_score, "candidate_id": _candidate_id(best_pipeline)}, indent=2), encoding="utf-8")
    return Q1GPResult(
        best_genome=best_genome,
        best_score=float(best_score),
        best_pipeline=best_pipeline,
        best_pipeline_json=best_pipeline_json,
        progress_csv=progress_csv,
        best_by_generation_csv=best_csv,
        candidate_summary_csv=candidate_summary_csv,
        state_json=state_json,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import time
import numpy as np
import pandas as pd

from ..data.io import load_stack, load_scores_npz, select_measure_curve
from ..data.manifests import build_manifest_from_direct_inputs, load_manifest, save_manifest_snapshot, validate_manifest
from ..data.splits import assign_nested_splits, assert_no_group_overlap, assert_test_never_in_inner
from ..metrics import objectives
from ..metrics import unsupervised
from ..metrics.focus_measures import compute_focus_curve
from ..preprocess.pipeline import (
    Pipeline,
    PreprocessError,
    ShapeMismatchError,
    InvalidValueError,
    RangeError,
    OOMError,
    IOError as PipelineIOError,
    TimeoutError as PipelineTimeoutError,
    UnknownError,
)


FAILURE_CODES = {
    ShapeMismatchError: 'ShapeMismatchError',
    InvalidValueError: 'InvalidValueError',
    RangeError: 'RangeError',
    OOMError: 'OOMError',
    PipelineIOError: 'IOError',
    PipelineTimeoutError: 'TimeoutError',
    UnknownError: 'UnknownError',
}


@dataclass
class FailureRecord:
    dataset: str
    group_id: str
    op_name: str
    error_code: str
    message: str


class FailureRecorder:
    def __init__(self) -> None:
        self.records: List[FailureRecord] = []

    def add(self, record: FailureRecord) -> None:
        self.records.append(record)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])



def _error_code(exc: Exception) -> str:
    for exc_type, code in FAILURE_CODES.items():
        if isinstance(exc, exc_type):
            return code
    msg = str(exc).lower()
    if 'out of memory' in msg:
        return 'OOMError'
    return 'UnknownError'


def _op_name(exc: Exception) -> str:
    return getattr(exc, 'op_name', 'pipeline')


def _normalize_reference_config(eval_cfg: Dict[str, Any], mode: str) -> Tuple[List[str], List[str]]:
    if mode == 'cached_scores':
        cached_cfg = eval_cfg.get('cached_scores', {})
        ref_paths = cached_cfg.get('ref_paths', []) or []
        dataset_names = cached_cfg.get('dataset_names', []) or []
        return list(dataset_names), list(ref_paths)

    ref_cfg = eval_cfg.get('reference_paths', {}) or {}
    if isinstance(ref_cfg, dict):
        dataset_names = list(ref_cfg.keys())
        ref_paths = [ref_cfg[name] for name in dataset_names]
        return dataset_names, ref_paths
    if isinstance(ref_cfg, list):
        dataset_names = []
        ref_paths = []
        for item in ref_cfg:
            if not isinstance(item, dict):
                continue
            name = item.get('dataset') or item.get('name')
            path = item.get('path')
            if name and path:
                dataset_names.append(str(name))
                ref_paths.append(str(path))
        return dataset_names, ref_paths
    return [], []


def _build_reference_map(dataset_names: List[str], ref_paths: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    ref_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, path in zip(dataset_names, ref_paths):
        if not path:
            continue
        peaks, conf = objectives.load_reference_peaks(path)
        ref_map[str(name)] = (peaks, conf)
    return ref_map


_UNSUP_WARNED = False


def _resolve_reference_mode(
    evaluation_cfg: Dict[str, Any],
    dataset_names: List[str],
    ref_paths: List[str],
) -> Tuple[str, Dict[str, Tuple[np.ndarray, np.ndarray]], str | None]:
    global _UNSUP_WARNED
    use_reference = bool(ref_paths)
    missing = []
    for path in ref_paths:
        if path and not Path(path).exists():
            missing.append(path)
    if use_reference and missing:
        raise ValueError(f"Reference peaks required but missing: {missing}")
    if not use_reference:
        if not _UNSUP_WARNED:
            print("WARNING: Reference peaks not provided; using unsupervised objective: unimodal_peak")
            _UNSUP_WARNED = True
        return 'unsupervised', {}, 'unimodal_peak'
    return 'reference', _build_reference_map(dataset_names, ref_paths), None


def _reference_for_row(
    row: pd.Series,
    reference_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    ref_peak = None
    confidence = None
    for key in ('ref_peak', 'reference_peak', 'best_focus_index'):
        if key in row and not pd.isna(row[key]):
            ref_peak = float(row[key])
            break
    if 'confidence' in row and not pd.isna(row['confidence']):
        confidence = float(row['confidence'])

    if (ref_peak is None or confidence is None) and reference_map:
        dataset = str(row.get('dataset'))
        stack_index = row.get('stack_index', None)
        if dataset in reference_map and stack_index is not None:
            peaks, conf = reference_map[dataset]
            idx = int(stack_index)
            if 0 <= idx < len(peaks):
                if ref_peak is None:
                    ref_peak = float(peaks[idx])
                if confidence is None:
                    if idx < len(conf):
                        confidence = float(conf[idx])

    if ref_peak is None or not np.isfinite(ref_peak):
        ref_peak = float('nan')
    if confidence is None or not np.isfinite(confidence):
        confidence = 0.0
    return ref_peak, confidence


def _build_manifest_from_cached_scores(cached_cfg: Dict[str, Any]) -> pd.DataFrame:
    scores_paths = cached_cfg.get('scores_paths') or cached_cfg.get('scores_npz') or []
    dataset_names = cached_cfg.get('dataset_names', []) or []
    if len(scores_paths) != len(dataset_names):
        raise ValueError('cached_scores.scores_paths and cached_scores.dataset_names must be the same length')
    records = []
    for dataset, path in zip(dataset_names, scores_paths):
        scores, _ = load_scores_npz(path)
        for idx in range(scores.shape[0]):
            stack_id = f'{dataset}_{idx:06d}'
            records.append({
                'dataset': str(dataset),
                'stack_id': stack_id,
                'group_id': stack_id,
                'path': str(path),
                'stack_index': idx,
            })
    df = pd.DataFrame(records)
    validate_manifest(df)
    return df


def _load_cached_curves(eval_cfg: Dict[str, Any], focus_cfg: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], str]:
    cached_cfg = eval_cfg.get('cached_scores', {})
    scores_paths = cached_cfg.get('scores_paths') or cached_cfg.get('scores_npz') or []
    dataset_names = cached_cfg.get('dataset_names', []) or []
    if len(scores_paths) != len(dataset_names):
        raise ValueError('cached_scores.scores_paths and cached_scores.dataset_names must be the same length')
    target_name = cached_cfg.get('measure_name') or focus_cfg.get('name')
    target_index = cached_cfg.get('measure_index', None)

    curves_map: Dict[str, np.ndarray] = {}
    selected_name = str(target_name) if target_name else ''
    for dataset, path in zip(dataset_names, scores_paths):
        scores, measure_names = load_scores_npz(path)
        curves, used_name = select_measure_curve(scores, measure_names, target_name=target_name, target_index=target_index)
        curves_map[str(dataset)] = curves
        selected_name = used_name
    return curves_map, selected_name


def build_manifest_for_run(
    cfg: Dict[str, Any],
    run_dir: Path,
    smoke_rows: Optional[int] = None,
    manifest_override: Optional[str] = None,
) -> pd.DataFrame:
    eval_cfg = cfg.get('evaluation', {}) or {}
    eval_mode = eval_cfg.get('mode', 'raw')
    if manifest_override:
        df = load_manifest(manifest_override, smoke_rows=smoke_rows)
    elif eval_mode == 'cached_scores':
        df = _build_manifest_from_cached_scores(eval_cfg.get('cached_scores', {}))
    elif cfg.get('data_mode') == 'direct':
        direct_cfg = cfg.get('direct_inputs', {})
        datasets_cfg = direct_cfg.get('datasets', [])
        group_map_path = direct_cfg.get('group_map_path') or direct_cfg.get('group_map')
        df = build_manifest_from_direct_inputs(datasets_cfg, group_map_path=group_map_path)
    else:
        df = load_manifest(cfg['manifest_path'], smoke_rows=smoke_rows)

    if smoke_rows is not None:
        df = df.head(int(smoke_rows)).copy()

    splits_cfg = cfg.get('splits', {})
    outer_test_size = float(splits_cfg.get('outer_test_size', 0.2))
    inner_val_size = float(splits_cfg.get('inner_val_size', 0.2))
    seed = int(cfg.get('seed', 0))

    if 'outer_split' not in df.columns or 'inner_split' not in df.columns:
        df = assign_nested_splits(df, group_col='group_id', outer_test_size=outer_test_size, inner_val_size=inner_val_size, seed=seed)
    assert_no_group_overlap(df, 'group_id', 'outer_split')
    assert_no_group_overlap(df[df['outer_split'] == 'trainval'], 'group_id', 'inner_split')
    assert_test_never_in_inner(df)

    artifacts_cfg = cfg.get('artifacts', {})
    if artifacts_cfg.get('save_manifest_snapshot', False):
        save_manifest_snapshot(df, run_dir / 'manifest_used.csv')
    return df


def evaluate_pipeline(
    manifest_df: pd.DataFrame,
    pipeline: Pipeline,
    focus_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, float],
    artifacts_dir: Path,
    run_tag: str | None,
    failure_penalty: float,
    outer_split: Optional[Sequence[str]] = None,
    inner_split: Optional[Sequence[str]] = None,
    cache: Optional[Dict[Tuple[str, str, int | None], Dict[str, Any]]] = None,
    pipeline_fingerprint: Optional[str] = None,
    evaluation_cfg: Optional[Dict[str, Any]] = None,
    tuning_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    eval_cfg = evaluation_cfg or {}
    tuning_cfg = tuning_cfg or {}
    eval_mode = eval_cfg.get('mode', 'raw')
    obj_cfg = objectives.config_from_dict(metrics_cfg, failure_penalty=failure_penalty)
    objectives.set_objective_config(obj_cfg)

    df = manifest_df.copy()
    if outer_split is not None:
        df = df[df['outer_split'].isin(list(outer_split))].copy()
    if inner_split is not None:
        df = df[df['inner_split'].isin(list(inner_split))].copy()
    if df.empty:
        raise ValueError('No samples found for the selected split filters')

    dataset_names, ref_paths = _normalize_reference_config(eval_cfg, eval_mode)
    objective_mode, reference_map, unsup_name = _resolve_reference_mode(
        eval_cfg, dataset_names, ref_paths
    )

    cached_curves = {}
    cached_measure_name = ''
    if eval_mode == 'cached_scores':
        cached_curves, cached_measure_name = _load_cached_curves(eval_cfg, focus_cfg)

    results = []
    failures = FailureRecorder()

    for row in df.to_dict('records'):
        dataset = str(row['dataset'])
        group_id = str(row['group_id'])
        stack_id = str(row['stack_id'])
        path = row['path']
        stack_index = row.get('stack_index', None)
        ref_peak, confidence = _reference_for_row(row, reference_map)

        cache_key = None
        if cache is not None and pipeline_fingerprint is not None:
            cache_key = (pipeline_fingerprint, group_id, stack_index)
            if cache_key in cache:
                cached = cache[cache_key].copy()
                cached['stack_id'] = stack_id
                cached['dataset'] = dataset
                cached['group_id'] = group_id
                cached['stack_index'] = stack_index
                cached['ref_peak'] = ref_peak
                cached['confidence'] = confidence
                results.append(cached)
                continue
        start = time.perf_counter()
        try:
            if eval_mode == 'cached_scores':
                if dataset not in cached_curves:
                    raise ValueError(f'Cached curves missing dataset: {dataset}')
                curves = cached_curves[dataset]
                if stack_index is None:
                    raise ValueError('stack_index required for cached_scores mode')
                curve = curves[int(stack_index)]
            else:
                stack = load_stack(path, stack_index=stack_index)
                processed = np.stack([pipeline.apply(stack[i]) for i in range(stack.shape[0])], axis=0)
                curve = compute_focus_curve(processed, name=focus_cfg['name'], **focus_cfg.get('params', {}))
            if objective_mode == 'reference':
                metrics = objectives.stack_fitness(curve, ref_peak, confidence)
            else:
                score, details = unsupervised.unimodal_peak_details(curve)
                is_failure = 1.0 if (not np.isfinite(score) or score >= obj_cfg.failure_penalty) else 0.0
                metrics = {
                    'pred_peak': float(details.get('pred_peak', float('nan'))),
                    'peak_error': float('nan'),
                    'penalty_nonfinite': float(details.get('nonfinite', 0.0)),
                    'penalty_flat': float(details.get('flat_penalty', float('nan'))),
                    'penalty_rough': float('nan'),
                    'penalty_multi_peak': float(details.get('multi_peak_penalty', float('nan'))),
                    'peak_gap': float('nan'),
                    'total_fitness': float(score),
                    'is_failure': float(is_failure),
                }
            metrics['stack_id'] = stack_id
            metrics['dataset'] = dataset
            metrics['group_id'] = group_id
            metrics['stack_index'] = stack_index
            metrics['ref_peak'] = ref_peak
            metrics['confidence'] = confidence
            metrics['runtime_sec'] = float(time.perf_counter() - start)
            metrics['objective_mode'] = objective_mode
            metrics['unsupervised_objective'] = unsup_name
            results.append(metrics)
            if metrics.get('penalty_nonfinite', 0.0) > 0.0:
                failures.add(FailureRecord(dataset, group_id, 'curve', 'InvalidValueError', 'Nonfinite curve'))
            if cache_key is not None:
                cache[cache_key] = metrics.copy()
        except Exception as exc:
            code = _error_code(exc)
            failures.add(FailureRecord(dataset, group_id, _op_name(exc), code, str(exc)))
            record = {
                'stack_id': stack_id,
                'dataset': dataset,
                'group_id': group_id,
                'stack_index': stack_index,
                'ref_peak': ref_peak,
                'confidence': confidence,
                'pred_peak': np.nan,
                'peak_error': np.nan,
                'penalty_nonfinite': 1.0,
                'penalty_flat': np.nan,
                'penalty_rough': np.nan,
                'penalty_multi_peak': np.nan,
                'peak_gap': np.nan,
                'total_fitness': float(obj_cfg.failure_penalty),
                'is_failure': 1.0,
                'runtime_sec': float(time.perf_counter() - start),
                'objective_mode': objective_mode,
                'unsupervised_objective': unsup_name,
            }
            results.append(record)
            if cache_key is not None:
                cache[cache_key] = record.copy()

    results_df = pd.DataFrame(results)
    required_cols = [
        'dataset',
        'group_id',
        'stack_id',
        'stack_index',
        'ref_peak',
        'confidence',
        'pred_peak',
        'peak_error',
        'penalty_nonfinite',
        'penalty_flat',
        'penalty_rough',
        'penalty_multi_peak',
        'peak_gap',
        'total_fitness',
        'is_failure',
    ]
    ordered = [c for c in required_cols if c in results_df.columns]
    extra = [c for c in results_df.columns if c not in ordered]
    results_df = results_df[ordered + extra]

    dataset_rows = []
    for dataset, group in results_df.groupby('dataset'):
        summary = objectives.aggregate_dataset_fitness(group)
        summary['dataset'] = dataset
        dataset_rows.append(summary)
    dataset_summary_df = pd.DataFrame(dataset_rows)

    dataset_weights = {}
    if isinstance(metrics_cfg, dict):
        for key, value in (metrics_cfg.get('dataset_weights', {}) or {}).items():
            try:
                dataset_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    gen = objectives.generalization_score(dataset_rows, dataset_weights)
    stability_ratio = gen['stability_ratio']
    stability_penalty = 0.0
    if objective_mode != 'unsupervised' and stability_ratio < obj_cfg.stability_threshold:
        stability_penalty = (obj_cfg.stability_threshold - stability_ratio) * obj_cfg.stability_penalty_weight
    hard_failure_rate = float(results_df['is_failure'].mean()) if len(results_df) else 1.0
    failure_rate_penalty = hard_failure_rate * obj_cfg.failure_rate_weight
    generalization_score = gen['generalization_score'] + stability_penalty + failure_rate_penalty

    summary = {
        'generalization_score': float(generalization_score),
        'weighted_mean': float(gen['weighted_mean']),
        'weighted_std': float(gen['weighted_std']),
        'global_stability_ratio': float(stability_ratio),
        'hard_failure_rate': float(hard_failure_rate),
        'stability_penalty': float(stability_penalty),
        'failure_rate_penalty': float(failure_rate_penalty),
        'num_samples': int(len(results_df)),
        'cached_measure_name': cached_measure_name,
        'objective_mode': objective_mode,
        'unsupervised_objective': unsup_name,
    }

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'_{run_tag}' if run_tag else ''
    results_df.to_csv(out_dir / f'per_stack_results{tag}.csv', index=False)
    dataset_summary_df.to_csv(out_dir / f'dataset_summary{tag}.csv', index=False)
    failures_df = failures.to_frame()
    failures_df.to_csv(out_dir / f'failures{tag}.csv', index=False)
    if not failures_df.empty:
        totals = results_df.groupby('dataset').size().to_dict()
        grouped = failures_df.groupby(['dataset', 'op_name']).size().reset_index(name='fail_count')
        grouped['failure_rate'] = grouped['dataset'].map(totals)
        grouped['failure_rate'] = grouped['fail_count'] / grouped['failure_rate']
        grouped.to_csv(out_dir / f'failure_rates{tag}.csv', index=False)

    run_summary = {
        'generalization_score': summary['generalization_score'],
        'weighted_mean': summary['weighted_mean'],
        'weighted_std': summary['weighted_std'],
        'global_stability_ratio': summary['global_stability_ratio'],
        'hard_failure_rate': summary['hard_failure_rate'],
        'cached_measure_name': cached_measure_name,
    }
    metadata_path = out_dir / 'run_metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        run_summary['config_hash'] = metadata.get('config_hash')
        backend_info = {}
        for key in ('opencv_available', 'unavailable_ops', 'backend_info'):
            if key in metadata:
                backend_info[key] = metadata[key]
        run_summary['backend_info'] = backend_info
    if pipeline_fingerprint:
        run_summary['pipeline_fingerprint'] = pipeline_fingerprint
    (out_dir / f'run_summary{tag}.json').write_text(json.dumps(run_summary, indent=2), encoding='utf-8')

    return summary, results_df

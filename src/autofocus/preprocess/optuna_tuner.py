from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import time

import optuna
import pandas as pd

from .param_spaces import PARAM_SPACES, DEFAULT_PARAMS
from .pipeline import Pipeline, PipelineStep
from ..eval.evaluate import evaluate_pipeline


class _PrefixedTrial:
    def __init__(self, trial: optuna.Trial, prefix: str) -> None:
        self._trial = trial
        self._prefix = prefix

    def _name(self, name: str) -> str:
        return f'{self._prefix}__{name}'

    def suggest_int(self, name: str, *args, **kwargs):
        return self._trial.suggest_int(self._name(name), *args, **kwargs)

    def suggest_float(self, name: str, *args, **kwargs):
        return self._trial.suggest_float(self._name(name), *args, **kwargs)

    def suggest_categorical(self, name: str, *args, **kwargs):
        return self._trial.suggest_categorical(self._name(name), *args, **kwargs)


def collect_tunable_params(tuning_cfg: Dict[str, Any], pipeline_steps: List[PipelineStep]) -> List[Tuple[str, str]]:
    tunables: List[Tuple[str, str]] = []
    allow_empty = bool(tuning_cfg.get('allow_empty_search_space', False))
    for step in pipeline_steps:
        if not step.enabled:
            continue
        params = step.params or {}
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if value is None:
                if step.name not in PARAM_SPACES:
                    raise ValueError(f"Step '{step.name}' has null param '{key}' but no PARAM_SPACES entry.")
                tunables.append((step.name, key))
    if not tunables and not allow_empty:
        raise ValueError(
            "No tunable parameters found. Enable steps and set params to null, and ensure PARAM_SPACES defines them."
        )
    return tunables


class OptunaTuner:
    def __init__(self, pipeline: Pipeline, focus_cfg: Dict[str, Any], metrics_cfg: Dict[str, Any],
                 n_trials: int, timeout_seconds: int, direction: str = 'minimize', seed: int = 0,
                 evaluation_cfg: Dict[str, Any] | None = None, tuning_cfg: Dict[str, Any] | None = None) -> None:
        self.pipeline = pipeline
        self.focus_cfg = focus_cfg
        self.metrics_cfg = metrics_cfg
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.direction = direction
        self.seed = seed
        self.evaluation_cfg = evaluation_cfg or {}
        self.tuning_cfg = tuning_cfg or {}

    def _suggest_params(self, trial: optuna.Trial, tunable_map: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        params = {}
        for step in self.pipeline.steps:
            if not step.enabled:
                continue
            space = PARAM_SPACES.get(step.name)
            if space is not None:
                tunables = tunable_map.get(step.name, [])
                if tunables:
                    suggested = space(_PrefixedTrial(trial, step.name))
                    selected = {}
                    for name in tunables:
                        if name not in suggested:
                            raise ValueError(f"PARAM_SPACES for '{step.name}' missing key '{name}'")
                        selected[name] = suggested[name]
                    params[step.name] = selected
                else:
                    params[step.name] = {}
            else:
                params[step.name] = DEFAULT_PARAMS.get(step.name, {})
        return params

    def _build_pipeline(self, params: Dict[str, Dict[str, Any]]) -> Pipeline:
        steps = []
        for step in self.pipeline.steps:
            base_params = step.params or {}
            if step.name in params:
                new_params = dict(base_params)
                new_params.update(params[step.name])
            else:
                new_params = base_params
            steps.append(PipelineStep(name=step.name, enabled=step.enabled, params=new_params, category=step.category))
        return Pipeline(steps=steps)

    def run(self, manifest_df: pd.DataFrame, artifacts_dir: Path, trial_results_path: str | Path | None = None) -> Tuple[Pipeline, Dict[str, Any]]:
        tunables = collect_tunable_params(self.tuning_cfg, self.pipeline.steps)
        tunable_map: Dict[str, List[str]] = {}
        for step_name, param_name in tunables:
            tunable_map.setdefault(step_name, []).append(param_name)

        writer = None
        trial_file = None
        if trial_results_path is not None:
            path = Path(trial_results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            trial_file = path.open('a', newline='', encoding='utf-8')
            writer = csv.DictWriter(trial_file, fieldnames=['trial_number', 'value', 'duration_sec', 'params_json'])
            if write_header:
                writer.writeheader()

        def objective(trial: optuna.Trial) -> float:
            start = time.perf_counter()
            params = self._suggest_params(trial, tunable_map)
            trial_pipeline = self._build_pipeline(params)
            summary, _ = evaluate_pipeline(
                manifest_df,
                trial_pipeline,
                focus_cfg=self.focus_cfg,
                metrics_cfg=self.metrics_cfg,
                outer_split=['trainval'],
                inner_split=['train', 'val'],
                artifacts_dir=artifacts_dir,
                run_tag=f'optuna_trial_{trial.number}',
                failure_penalty=1e6,
                evaluation_cfg=self.evaluation_cfg,
                tuning_cfg=self.tuning_cfg,
            )
            value = float(summary['generalization_score'])
            duration = float(time.perf_counter() - start)
            if writer is not None:
                writer.writerow({
                    'trial_number': trial.number,
                    'value': value,
                    'duration_sec': duration,
                    'params_json': json.dumps(params),
                })
            return value

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)
        if trial_file is not None:
            trial_file.flush()
            trial_file.close()

        best_params = study.best_params
        by_step: Dict[str, Dict[str, Any]] = {}
        for key, value in best_params.items():
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                by_step.setdefault(step_name, {})[param_name] = value
        best_pipeline = self._build_pipeline(by_step)
        return best_pipeline, {
            'best_params': best_params,
            'best_params_by_step': by_step,
            'best_value': study.best_value,
        }

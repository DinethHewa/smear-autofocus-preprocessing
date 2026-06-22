from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import copy
import csv
import json
import time


from .param_spaces import PARAM_SPACES, DEFAULT_PARAMS, param_space_keys
from .pipeline import Pipeline, PipelineStep
from ..preprocess.ops import is_op_available
from ..eval.evaluate import build_manifest_for_run, evaluate_pipeline
from ..eval.summary import write_eval_summary
from ..utils.run_artifacts import dump_config_used, format_run_summary_line, write_run_summary


def build_per_op_pipeline(cfg: Dict[str, Any], op_name: str) -> Pipeline:
    if op_name not in PARAM_SPACES:
        raise ValueError(f"Unknown operator for tuning: {op_name}")
    if not is_op_available(op_name):
        raise ValueError(f"Operator '{op_name}' is unavailable in this environment")

    tunable_keys = param_space_keys(op_name)
    if not tunable_keys:
        raise ValueError(f"Operator '{op_name}' has no tunable parameters defined in PARAM_SPACES")

    base = Pipeline.from_dict(cfg["pipeline"])
    steps = []
    found = False

    for step in base.steps:
        is_baseline = step.name == "identity" or bool(step.baseline)
        if step.name == op_name:
            found = True
            params = dict(DEFAULT_PARAMS.get(step.name, {}))
            params.update(step.params or {})
            for key in tunable_keys:
                params[key] = None
            steps.append(PipelineStep(
                name=step.name,
                enabled=True,
                params=params,
                category=step.category,
                baseline=step.baseline,
            ))
        elif is_baseline:
            steps.append(PipelineStep(
                name=step.name,
                enabled=True,
                params=step.params or {},
                category=step.category,
                baseline=step.baseline,
            ))
        else:
            steps.append(PipelineStep(
                name=step.name,
                enabled=False,
                params=step.params or {},
                category=step.category,
                baseline=step.baseline,
            ))

    if not found:
        params = dict(DEFAULT_PARAMS.get(op_name, {}))
        for key in tunable_keys:
            params[key] = None
        steps.append(PipelineStep(
            name=op_name,
            enabled=True,
            params=params,
            category=None,
            baseline=False,
        ))

    return Pipeline(steps=steps)


def _inject_params(pipeline: Pipeline, op_name: str, params: Dict[str, Any]) -> Pipeline:
    steps = []
    for step in pipeline.steps:
        if step.name == op_name:
            new_params = dict(step.params or {})
            new_params.update(params)
            steps.append(PipelineStep(
                name=step.name,
                enabled=step.enabled,
                params=new_params,
                category=step.category,
                baseline=step.baseline,
            ))
        else:
            steps.append(step)
    return Pipeline(steps=steps)


def tune_single_op(cfg: Dict[str, Any], op_name: str, run_dir: str | Path,
                   n_trials: int, smoke: bool) -> Dict[str, Any]:
    if op_name not in PARAM_SPACES:
        raise ValueError(f"Unknown operator for tuning: {op_name}")
    if not is_op_available(op_name):
        raise ValueError(f"Operator '{op_name}' is unavailable in this environment")

    tunable_keys = param_space_keys(op_name)
    if not tunable_keys:
        raise ValueError(f"Operator '{op_name}' has no tunable parameters defined in PARAM_SPACES")

    per_op_cfg = copy.deepcopy(cfg)
    pipeline = build_per_op_pipeline(per_op_cfg, op_name)
    per_op_cfg["pipeline"] = pipeline.to_dict()

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    dump_config_used(per_op_cfg, run_path)

    smoke_rows = per_op_cfg.get("smoke_rows", 10) if smoke else None
    manifest_df = build_manifest_for_run(per_op_cfg, run_path, smoke_rows=smoke_rows, manifest_override=None)

    trial_results_path = run_path / "trial_results.csv"
    trial_file = trial_results_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(trial_file, fieldnames=["trial_number", "value", "duration_sec", "params_json"])
    writer.writeheader()

    space_fn = PARAM_SPACES[op_name]
    best_value = float("inf")
    best_params: Dict[str, Any] = {}

    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        if not params:
            raise ValueError(f"Operator '{op_name}' produced empty parameter space")
        start = time.perf_counter()
        trial_pipeline = _inject_params(pipeline, op_name, params)
        summary, _ = evaluate_pipeline(
            manifest_df,
            trial_pipeline,
            focus_cfg=per_op_cfg["focus_measure"],
            metrics_cfg=per_op_cfg["metrics"],
            outer_split=["trainval"],
            inner_split=["train", "val"],
            artifacts_dir=run_path,
            run_tag=f"trial_{trial.number}",
            failure_penalty=float(per_op_cfg["metrics"].get("failure_penalty", 1e6)),
            evaluation_cfg=per_op_cfg.get("evaluation", {}),
            pipeline_fingerprint=trial_pipeline.fingerprint(),
        )
        value = float(summary["generalization_score"])
        duration = float(time.perf_counter() - start)
        writer.writerow({
            "trial_number": trial.number,
            "value": value,
            "duration_sec": duration,
            "params_json": json.dumps(params),
        })
        trial_file.flush()
        trial.set_user_attr("params_dict", params)
        return value

    sampler = optuna.samplers.TPESampler(seed=int(per_op_cfg.get("seed", 0)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials))

    if study.best_trial is not None:
        best_value = float(study.best_value)
        best_params = dict(study.best_trial.user_attrs.get("params_dict", {}))

    trial_file.close()

    best_pipeline = _inject_params(pipeline, op_name, best_params)
    (run_path / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (run_path / "best_pipeline.json").write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")

    best_summary, _ = evaluate_pipeline(
        manifest_df,
        best_pipeline,
        focus_cfg=per_op_cfg["focus_measure"],
        metrics_cfg=per_op_cfg["metrics"],
        outer_split=["trainval"],
        inner_split=["train", "val"],
        artifacts_dir=run_path,
        run_tag="eval",
        failure_penalty=float(per_op_cfg["metrics"].get("failure_penalty", 1e6)),
        evaluation_cfg=per_op_cfg.get("evaluation", {}),
        pipeline_fingerprint=best_pipeline.fingerprint(),
    )

    eval_summary = write_eval_summary(
        run_path,
        run_path.name,
        best_summary.get("objective_mode"),
        best_summary.get("unsupervised_objective"),
    )
    if eval_summary.get("generalization_score") is None:
        eval_summary["generalization_score"] = float(best_value)
        (run_path / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    def _to_float(value: Any) -> float:
        try:
            if value is None:
                return float("nan")
            return float(value)
        except Exception:
            return float("nan")

    line = format_run_summary_line(
        run_id=run_path.name,
        generalization_score=_to_float(eval_summary.get("generalization_score", best_value)),
        stability_ratio=_to_float(eval_summary.get("stability_ratio", float("nan"))),
        ms_per_image=_to_float(eval_summary.get("ms_per_image", float("nan"))),
        num_failed=int(eval_summary.get("failures_failed", 0)),
        num_total=int(eval_summary.get("failures_total", 0)),
    )
    write_run_summary(run_path, run_path.name, {"line": line})

    return {
        "best_value": best_value,
        "best_params": best_params,
        "objective_mode": eval_summary.get("objective_mode"),
        "ms_per_image": eval_summary.get("ms_per_image"),
        "failures_failed": eval_summary.get("failures_failed", 0),
        "failures_total": eval_summary.get("failures_total", 0),
        "run_dir": str(run_path),
        "trial_results": str(trial_results_path),
    }

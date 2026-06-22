from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import numbers


def _require_key(cfg: Dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    value = cfg[key]
    if not isinstance(value, expected):
        exp = expected if isinstance(expected, tuple) else (expected,)
        exp_names = ", ".join(t.__name__ for t in exp)
        raise ValueError(f"Config key '{key}' must be {exp_names}")
    return value


def _require_numeric(cfg: Dict[str, Any], key: str) -> float:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    value = cfg[key]
    if not isinstance(value, numbers.Real):
        raise ValueError(f"Config key '{key}' must be a number")
    return float(value)


def _require_list(value: Any, key: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Config key '{key}' must be a list")
    return value


def _require_dict(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a dict")
    return value


def _require_non_empty_list(value: Any, key: str) -> list:
    lst = _require_list(value, key)
    if not lst:
        raise ValueError(f"Config key '{key}' must be a non-empty list")
    return lst


def _require_list_of_dicts(value: Any, key: str, required_keys: Iterable[str]) -> list:
    lst = _require_non_empty_list(value, key)
    for idx, item in enumerate(lst):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{idx}] must be a dict")
        for req in required_keys:
            if req not in item:
                raise ValueError(f"{key}[{idx}] missing required key: {req}")
            if not isinstance(item[req], str):
                raise ValueError(f"{key}[{idx}].{req} must be a string")
    return lst


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    _require_key(cfg, "seed", int)
    _require_key(cfg, "artifacts_dir", str)
    data_mode = _require_key(cfg, "data_mode", str)
    if data_mode not in {"direct", "manifest"}:
        raise ValueError("Config key 'data_mode' must be one of {'direct','manifest'}")

    splits = _require_key(cfg, "splits", dict)
    _require_numeric(splits, "outer_test_size")
    _require_numeric(splits, "inner_val_size")

    pipeline = _require_key(cfg, "pipeline", dict)
    steps = _require_key(pipeline, "steps", list)
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"pipeline.steps[{idx}] must be a dict")
        if "name" not in step or not isinstance(step["name"], str):
            raise ValueError(f"pipeline.steps[{idx}].name must be a string")
        if "enabled" not in step or not isinstance(step["enabled"], bool):
            raise ValueError(f"pipeline.steps[{idx}].enabled must be a bool")
        if "params" not in step or not isinstance(step["params"], dict):
            raise ValueError(f"pipeline.steps[{idx}].params must be a dict")
        if "baseline" in step and not isinstance(step["baseline"], bool):
            raise ValueError(f"pipeline.steps[{idx}].baseline must be a bool if provided")

    focus = _require_key(cfg, "focus_measure", dict)
    if "name" not in focus or not isinstance(focus["name"], str):
        raise ValueError("focus_measure.name must be a string")
    params = focus.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("focus_measure.params must be a dict if provided")
    focus["params"] = params

    metrics = _require_key(cfg, "metrics", dict)
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict")

    evaluation = _require_key(cfg, "evaluation", dict)
    mode = _require_key(evaluation, "mode", str)
    if mode not in {"raw", "cached_scores"}:
        raise ValueError("evaluation.mode must be one of {'raw','cached_scores'}")

    if mode == "cached_scores":
        cached = _require_dict(evaluation.get("cached_scores", {}), "evaluation.cached_scores")
        if "scores_paths" not in cached and "scores_npz" in cached:
            cached["scores_paths"] = cached.get("scores_npz")
        scores_paths = _require_non_empty_list(cached.get("scores_paths"), "evaluation.cached_scores.scores_paths")
        dataset_names = _require_non_empty_list(cached.get("dataset_names"), "evaluation.cached_scores.dataset_names")
        ref_paths = _require_non_empty_list(cached.get("ref_paths"), "evaluation.cached_scores.ref_paths")
        measure_name = cached.get("measure_name")
        if not isinstance(measure_name, str) or not measure_name:
            raise ValueError("evaluation.cached_scores.measure_name must be a non-empty string")
        if len(scores_paths) != len(dataset_names):
            raise ValueError("cached_scores.scores_paths must match dataset_names length")
        if len(ref_paths) != len(dataset_names):
            raise ValueError("cached_scores.ref_paths must match dataset_names length")
    else:
        ref_paths = evaluation.get("reference_paths", None)
        if ref_paths is not None and not isinstance(ref_paths, (dict, list)):
            raise ValueError("evaluation.reference_paths must be a dict or list when provided")

    if data_mode == "direct":
        direct = _require_dict(cfg.get("direct_inputs", {}), "direct_inputs")
        _require_list_of_dicts(direct.get("datasets"), "direct_inputs.datasets", ("name", "path"))
        group_map_path = direct.get("group_map_path") or direct.get("group_map")
        if group_map_path is not None:
            if not isinstance(group_map_path, str):
                raise ValueError("direct_inputs.group_map_path must be a string if provided")
            if not Path(group_map_path).exists():
                raise ValueError(f"direct_inputs.group_map_path does not exist: {group_map_path}")
    else:
        manifest_path = _require_key(cfg, "manifest_path", str)
        if not Path(manifest_path).exists():
            raise ValueError(f"manifest_path does not exist: {manifest_path}")

    tuning = cfg.get("tuning", None)
    if tuning is not None:
        tuning = _require_dict(tuning, "tuning")
        if "mode" in tuning:
            if not isinstance(tuning["mode"], str):
                raise ValueError("tuning.mode must be a string")
            if tuning["mode"] not in {"joint", "per_step"}:
                raise ValueError("tuning.mode must be one of {'joint','per_step'}")
        if "steps" in tuning:
            steps = tuning["steps"]
            if not isinstance(steps, list):
                raise ValueError("tuning.steps must be a list of strings")
            for idx, name in enumerate(steps):
                if not isinstance(name, str) or not name:
                    raise ValueError(f"tuning.steps[{idx}] must be a non-empty string")
        if "allow_empty_search_space" in tuning and not isinstance(tuning["allow_empty_search_space"], bool):
            raise ValueError("tuning.allow_empty_search_space must be a bool")
        if "objective_mode" in tuning and not isinstance(tuning["objective_mode"], str):
            raise ValueError("tuning.objective_mode must be a string")
        if "unsupervised_objective" in tuning and not isinstance(tuning["unsupervised_objective"], str):
            raise ValueError("tuning.unsupervised_objective must be a string")
        cfg["tuning"] = tuning

    artifacts = cfg.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a dict when provided")
    if "save_manifest_snapshot" not in artifacts:
        artifacts["save_manifest_snapshot"] = True
    cfg["artifacts"] = artifacts

    gp_cfg = cfg.get("gp", None)
    if gp_cfg is not None:
        gp_cfg = _require_dict(gp_cfg, "gp")
        if "use_tuned_params" in gp_cfg and not isinstance(gp_cfg["use_tuned_params"], bool):
            raise ValueError("gp.use_tuned_params must be a bool")
        if "strict_tuned_params" in gp_cfg and not isinstance(gp_cfg["strict_tuned_params"], bool):
            raise ValueError("gp.strict_tuned_params must be a bool")
        tuned_dir = gp_cfg.get("tuned_params_dir")
        if tuned_dir is not None:
            if not isinstance(tuned_dir, str):
                raise ValueError("gp.tuned_params_dir must be a string if provided")
            if tuned_dir and not Path(tuned_dir).exists():
                raise ValueError(f"gp.tuned_params_dir does not exist: {tuned_dir}")
        cfg["gp"] = gp_cfg

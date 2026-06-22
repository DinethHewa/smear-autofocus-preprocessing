from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_per_op_root(per_op_run: Path) -> Path:
    if (per_op_run / "optuna_per_op").exists():
        return per_op_run / "optuna_per_op"
    if per_op_run.name == "optuna_per_op":
        return per_op_run
    raise ValueError(f"Cannot find optuna_per_op under: {per_op_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge GP pipeline with per-operator tuned params.")
    parser.add_argument("--gp-run", required=True, help="Path to GP run directory (contains best_pipeline.json).")
    parser.add_argument("--per-op-run", required=True, help="Path to per-op run directory (contains optuna_per_op).")
    parser.add_argument("--pipeline-out", default=None, help="Output JSON path for tuned pipeline.")
    parser.add_argument("--config-in", default="configs/default.yaml", help="Base config to update.")
    parser.add_argument("--config-out", default=None, help="Output YAML path for tuned config.")
    args = parser.parse_args()

    gp_run = Path(args.gp_run)
    per_op_run = Path(args.per_op_run)

    gp_pipeline_path = gp_run / "best_pipeline.json"
    if not gp_pipeline_path.exists():
        raise FileNotFoundError(f"Missing best_pipeline.json in: {gp_run}")

    per_op_root = _resolve_per_op_root(per_op_run)

    per_op_params: Dict[str, Dict[str, Any]] = {}
    for op_dir in per_op_root.iterdir():
        if not op_dir.is_dir():
            continue
        best_params = op_dir / "best_params.json"
        if best_params.exists():
            per_op_params[op_dir.name] = _load_json(best_params)

    pipeline = _load_json(gp_pipeline_path)
    steps = pipeline.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("Invalid pipeline JSON: missing steps list.")

    applied = []
    skipped = []
    for step in steps:
        name = step.get("name")
        if not isinstance(name, str):
            continue
        params = step.get("params") or {}
        if name in per_op_params:
            merged = dict(params)
            merged.update(per_op_params[name])
            step["params"] = merged
            applied.append(name)
        else:
            skipped.append(name)

    pipeline_out = Path(args.pipeline_out) if args.pipeline_out else (gp_run / "best_pipeline_tuned.json")
    pipeline_out.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")

    config_in = Path(args.config_in)
    if not config_in.exists():
        raise FileNotFoundError(f"Config not found: {config_in}")

    config_out = Path(args.config_out) if args.config_out else (gp_run / "config_tuned.yaml")
    cfg = _load_yaml(config_in)
    cfg["pipeline"] = {"steps": steps}
    config_out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Wrote tuned pipeline: {pipeline_out}")
    print(f"Wrote tuned config: {config_out}")
    print(f"Applied params for ops: {applied}")
    print(f"Missing per-op params for ops: {skipped}")


if __name__ == "__main__":
    main()

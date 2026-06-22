from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


def resolve_per_op_root(per_op_run: str | Path) -> Path:
    base = Path(per_op_run)
    if (base / "optuna_per_op").exists():
        return base / "optuna_per_op"
    if base.name == "optuna_per_op":
        return base
    raise ValueError(f"Cannot find optuna_per_op under: {base}")


def load_tuned_params(per_op_run: str | Path, strict: bool = True) -> Dict[str, Dict[str, Any]]:
    root = resolve_per_op_root(per_op_run)
    params: Dict[str, Dict[str, Any]] = {}
    missing = []
    for op_dir in sorted(root.iterdir()):
        if not op_dir.is_dir():
            continue
        best_params_path = op_dir / "best_params.json"
        if not best_params_path.exists():
            missing.append(op_dir.name)
            continue
        try:
            data = json.loads(best_params_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {best_params_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"best_params.json must contain a JSON object: {best_params_path}")
        params[op_dir.name] = data

    if strict and missing:
        raise ValueError(f"Missing best_params.json for ops: {missing}")
    return params

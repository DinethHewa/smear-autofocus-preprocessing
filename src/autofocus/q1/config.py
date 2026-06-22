from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_outputs"


@dataclass(frozen=True)
class Q1RunContext:
    config_path: Path
    config: Dict[str, Any]
    run_id: str
    output_dir: Path
    logs_dir: Path
    smoke: bool
    resume: bool
    overwrite: bool


def load_q1_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Q1 config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Q1 config must be a mapping: {config_path}")
    return cfg


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_text(text: str) -> str:
    return stable_hash_bytes(text.encode("utf-8"))


def stable_hash_file(path: str | Path) -> str:
    return stable_hash_bytes(Path(path).read_bytes())


def stable_hash_mapping(mapping: Mapping[str, Any]) -> str:
    raw = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str)
    return stable_hash_text(raw)


def _default_run_id(cfg: Mapping[str, Any], config_path: Path, smoke: bool) -> str:
    env_run_id = os.environ.get("Q1_RUN_ID")
    if env_run_id:
        return env_run_id
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    if run_cfg.get("run_id"):
        return str(run_cfg["run_id"])
    name = str(run_cfg.get("name") or ("q1_smoke" if smoke else "q1_run"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = stable_hash_text(str(config_path) + json.dumps(cfg, sort_keys=True, default=str))[:8]
    return f"{stamp}_{name}_{suffix}"


def make_run_context(
    config_path: str | Path,
    *,
    smoke: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> Q1RunContext:
    resolved_config_path = Path(config_path).expanduser().resolve()
    cfg = load_q1_config(resolved_config_path)
    run_id = _default_run_id(cfg, resolved_config_path, smoke)
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    output_root = Path(run_cfg.get("output_root") or DEFAULT_OUTPUT_ROOT).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_dir = output_root / run_id
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return Q1RunContext(
        config_path=resolved_config_path,
        config=cfg,
        run_id=run_id,
        output_dir=output_dir,
        logs_dir=logs_dir,
        smoke=smoke,
        resume=resume,
        overwrite=overwrite,
    )


def ensure_stage_writable(path: Path, *, resume: bool, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite or resume:
        return
    raise FileExistsError(
        f"Refusing to overwrite existing output without --overwrite or --resume: {path}"
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage_summary(
    ctx: Q1RunContext,
    stage_name: str,
    payload: Mapping[str, Any],
    *,
    status: str = "complete",
) -> Path:
    summary = {
        "stage": stage_name,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "config_path": str(ctx.config_path),
        "output_dir": str(ctx.output_dir),
        **dict(payload),
    }
    path = ctx.output_dir / "summaries" / f"{stage_name}.json"
    write_json(path, summary)
    return path


def source_tree_hash(root: Path = REPO_ROOT, suffixes: Iterable[str] = (".py", ".yaml", ".yml", ".toml")) -> Dict[str, Any]:
    files = []
    digest = hashlib.sha256()
    skip_dirs = {".git", ".pytest_cache", "__pycache__", "paper_outputs", "artifacts"}
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in skip_dirs]
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if path.suffix not in suffixes:
                continue
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            file_hash = stable_hash_bytes(data)
            files.append({"path": rel, "sha256": file_hash})
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\0")
    return {"source_tree_sha256": digest.hexdigest(), "files": files}


def environment_metadata() -> Dict[str, Any]:
    packages: Dict[str, str] = {}
    try:
        import importlib.metadata as importlib_metadata

        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name")
            version = dist.version
            if name:
                packages[str(name)] = str(version)
    except Exception:
        packages = {}
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "packages": dict(sorted(packages.items())),
    }


def display_category(category: str | None) -> str:
    if not category:
        return "Unspecified"
    return str(category).replace("_", " ").title()

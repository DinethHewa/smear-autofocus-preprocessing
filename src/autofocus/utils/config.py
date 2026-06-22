from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return data or {}


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return load_yaml(path)


def hash_config(cfg: Dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

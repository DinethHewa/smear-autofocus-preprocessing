from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

from importlib import metadata as importlib_metadata

from .config import hash_config


def _git_hash(root: Path) -> str:
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(root), stderr=subprocess.DEVNULL
        )
        return result.decode('utf-8').strip()
    except Exception:
        return 'unknown'


def _collect_packages() -> Dict[str, str]:
    packages = {}
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get('Name')
            if name:
                packages[name] = dist.version
    except Exception:
        pass
    return packages


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def init_run(artifacts_dir: str | Path, config: Dict[str, Any], seed: int, extra_metadata: Dict[str, Any] | None = None) -> Tuple[str, Path]:
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'run_id': run_id,
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'seed': seed,
        'config_hash': hash_config(config),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'os': os.name,
        'git_hash': _git_hash(Path.cwd()),
        'packages': _collect_packages(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(run_dir / 'run_metadata.json', metadata)
    return run_id, run_dir

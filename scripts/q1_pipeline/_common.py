from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from autofocus.q1.config import make_run_context
from autofocus.q1.stages import export_reproducibility


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to Q1 YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run in reduced-size smoke mode.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing stage outputs when present.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing stage outputs.")
    parser.add_argument("--run-id", default="", help="Run ID to resume or share across stage scripts.")
    return parser


def run_stage(stage_func: Callable[[Any], Any], description: str) -> None:
    parser = build_parser(description)
    args = parser.parse_args()
    if args.run_id:
        os.environ["Q1_RUN_ID"] = args.run_id
    ctx = make_run_context(
        args.config,
        smoke=args.smoke,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    result = stage_func(ctx)
    try:
        export_reproducibility(ctx)
    except FileNotFoundError:
        pass
    print(json.dumps({"run_id": ctx.run_id, "output_dir": str(ctx.output_dir), "result": result}, indent=2, default=str))

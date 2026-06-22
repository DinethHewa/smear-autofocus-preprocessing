from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the Q1 pipeline with optional backend flags.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-gpu", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--backend", choices=["auto", "pandas", "polars"], default="auto")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--disable-parallel", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["AUTOFOCUS_USE_GPU"] = args.use_gpu
    env["AUTOFOCUS_DATAFRAME_BACKEND"] = args.backend
    env["AUTOFOCUS_NUM_WORKERS"] = str(args.num_workers)
    env["AUTOFOCUS_DISABLE_PARALLEL"] = "1" if args.disable_parallel else "0"

    script = Path(__file__).resolve().parents[1] / "scripts" / "q1_pipeline" / "10_run_full_q1_pipeline.py"
    command = [sys.executable, str(script), "--config", args.config]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.smoke:
        command.append("--smoke")
    if args.resume:
        command.append("--resume")
    if args.overwrite:
        command.append("--overwrite")
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()

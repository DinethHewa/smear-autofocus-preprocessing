from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--all-ops", action="store_true", default=False)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from ..utils.config import load_config
    cfg = load_config(args.config)
    from ..utils.config_validate import validate_config
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    seed = int(cfg.get("seed", 0))
    from ..utils.seed import set_global_seed
    set_global_seed(seed)
    from ..utils.logging import init_run
    from ..preprocess.ops import backend_info
    artifacts_root = args.out if args.out else cfg.get("artifacts_dir", "artifacts")
    run_id, run_dir = init_run(artifacts_root, cfg, seed, extra_metadata=backend_info())

    from ..utils.run_artifacts import dump_config_used
    dump_config_used(cfg, run_dir)

    from ..eval.evaluate import build_manifest_for_run
    smoke_rows = cfg.get("smoke_rows", 10) if args.smoke else None
    build_manifest_for_run(cfg, run_dir, smoke_rows=smoke_rows, manifest_override=None)

    from ..preprocess.param_spaces import PARAM_SPACES
    from ..preprocess.ops import is_op_available
    if args.ops:
        ops = [str(op) for op in args.ops]
    else:
        ops = sorted(PARAM_SPACES.keys())

    if args.all_ops:
        ops = sorted(PARAM_SPACES.keys())

    if args.ops:
        unavailable = [op for op in ops if not is_op_available(op)]
        if unavailable:
            raise ValueError(f"Requested operators unavailable in this environment: {unavailable}")
    else:
        skipped = [op for op in ops if not is_op_available(op)]
        if skipped:
            print(f"Warning: skipping unavailable operators: {skipped}")
        ops = [op for op in ops if is_op_available(op)]

    if not ops:
        raise ValueError("No operators selected for per-operator tuning.")

    n_trials = args.n_trials
    if n_trials is None:
        n_trials = int(cfg.get("optuna", {}).get("n_trials", 50))

    per_op_root = Path(run_dir) / "optuna_per_op"
    per_op_root.mkdir(parents=True, exist_ok=True)

    from ..preprocess.optuna_single_op import tune_single_op

    summary_rows = []
    for op_name in ops:
        op_dir = per_op_root / op_name
        result = tune_single_op(cfg, op_name, op_dir, n_trials=n_trials, smoke=args.smoke)
        summary_rows.append({
            "op_name": op_name,
            "best_value": float(result.get("best_value", float("nan"))),
            "best_params_json": json.dumps(result.get("best_params", {})),
            "objective_mode": result.get("objective_mode"),
            "ms_per_image": result.get("ms_per_image"),
            "failures_failed": result.get("failures_failed", 0),
            "failures_total": result.get("failures_total", 0),
        })

    import pandas as pd

    summary_path = per_op_root / "per_op_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"PER_OP_TUNING_DONE: parent_run_id={run_id}, ops={len(summary_rows)}, summary_csv={summary_path}")


if __name__ == "__main__":
    main()

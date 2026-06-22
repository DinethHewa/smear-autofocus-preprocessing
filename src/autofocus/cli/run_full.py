from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..eval.evaluate import build_manifest_for_run, evaluate_pipeline
from ..eval.summary import write_eval_summary
from ..explain.report import build_report
from ..gp.genome import DEFAULT_CATEGORIES, DEFAULT_OPS_BY_CATEGORY, filter_ops_by_availability
from ..gp.preprocess_gp import GPConfig, GPOptimizer
from ..preprocess.optuna_single_op import tune_single_op
from ..preprocess.ops import backend_info, is_op_available
from ..preprocess.param_spaces import PARAM_SPACES
from ..preprocess.tuned_params import load_tuned_params, resolve_per_op_root
from ..utils.config import load_config
from ..utils.config_validate import validate_config
from ..utils.logging import init_run
from ..utils.run_artifacts import dump_config_used, format_run_summary_line, write_run_summary
from ..utils.seed import set_global_seed


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary_row(op_name: str, op_dir: Path) -> Dict[str, Any]:
    best_params = _load_json(op_dir / "best_params.json") if op_dir.exists() else {}
    eval_summary = _load_json(op_dir / "eval_summary.json") if op_dir.exists() else {}
    return {
        "op_name": op_name,
        "best_value": eval_summary.get("generalization_score", float("nan")),
        "best_params_json": json.dumps(best_params),
        "objective_mode": eval_summary.get("objective_mode"),
        "ms_per_image": eval_summary.get("ms_per_image"),
        "failures_failed": eval_summary.get("failures_failed", 0),
        "failures_total": eval_summary.get("failures_total", 0),
    }


def _resolve_ops(ops: List[str] | None, all_ops: bool) -> List[str]:
    if ops:
        return [str(op) for op in ops]
    if all_ops:
        return sorted(PARAM_SPACES.keys())
    return sorted(PARAM_SPACES.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="Parent run directory. Required for resume.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run directory.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--all-ops", action="store_true", default=False)
    parser.add_argument("--per-op-run", default=None, help="Reuse an existing per-op run directory.")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            if args.resume:
                raise ValueError(f"run-dir does not exist for resume: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and not (run_dir / "run_metadata.json").exists():
            print("Warning: run_metadata.json missing; resume continues without parent metadata.")
        run_id = run_dir.name
    else:
        run_id, run_dir = init_run(cfg.get("artifacts_dir", "artifacts"), cfg, seed, extra_metadata=backend_info())

    dump_config_used(cfg, run_dir)

    smoke_rows = cfg.get("smoke_rows", 10) if args.smoke else None
    manifest_df = build_manifest_for_run(cfg, run_dir, smoke_rows=smoke_rows, manifest_override=None)

    ops = _resolve_ops(args.ops, args.all_ops)
    if args.ops:
        unavailable = [op for op in ops if not is_op_available(op)]
        if unavailable:
            raise ValueError(f"Requested operators unavailable in this environment: {unavailable}")
    else:
        ops = [op for op in ops if is_op_available(op)]

    # Per-operator tuning.
    if args.per_op_run:
        per_op_root = resolve_per_op_root(args.per_op_run)
    else:
        per_op_root = run_dir / "optuna_per_op"
        per_op_root.mkdir(parents=True, exist_ok=True)
        n_trials = args.n_trials if args.n_trials is not None else int(cfg.get("optuna", {}).get("n_trials", 50))
        for op_name in ops:
            op_dir = per_op_root / op_name
            best_params_path = op_dir / "best_params.json"
            if args.resume and best_params_path.exists():
                print(f"PER_OP_RESUME: skipping {op_name} (best_params.json exists)")
                continue
            print(f"PER_OP_TUNE: {op_name}")
            tune_single_op(cfg, op_name, op_dir, n_trials=n_trials, smoke=args.smoke)

        summary_rows = [_summary_row(op_name, per_op_root / op_name) for op_name in ops]
        pd.DataFrame(summary_rows).to_csv(per_op_root / "per_op_summary.csv", index=False)

    # GP search using tuned params.
    gp_dir = run_dir / "gp"
    gp_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (gp_dir / "best_pipeline.json").exists():
        print("GP_RESUME: skipping GP search (best_pipeline.json exists)")
    else:
        gp_cfg = cfg.get("gp", {})
        config = GPConfig(
            generations=int(gp_cfg.get("generations", 10)),
            population_size=int(gp_cfg.get("population_size", 20)),
            elite_size=int(gp_cfg.get("elite_size", 2)),
            mutation_rate=float(gp_cfg.get("mutation_rate", 0.2)),
            crossover_rate=float(gp_cfg.get("crossover_rate", 0.8)),
            cache=bool(gp_cfg.get("cache", True)),
        )
        gp_log_path = gp_dir / "gp_log.txt"
        gp_log_file = gp_log_path.open("w", encoding="utf-8")

        def log(msg: str) -> None:
            print(msg)
            gp_log_file.write(msg + "\n")
            gp_log_file.flush()

        log(
            "GP_START: "
            f"generations={config.generations}, "
            f"population={config.population_size}, "
            f"elite={config.elite_size}, "
            f"mutation={config.mutation_rate}, "
            f"crossover={config.crossover_rate}"
        )

        params_lookup = load_tuned_params(per_op_root, strict=True)
        (gp_dir / "tuned_params_used.json").write_text(
            json.dumps(params_lookup, indent=2), encoding="utf-8"
        )
        log(f"GP_TUNED_PARAMS: enabled=True, per_op_run={per_op_root}, ops={len(params_lookup)}")

        ops_by_category = filter_ops_by_availability(DEFAULT_OPS_BY_CATEGORY)
        optimizer = GPOptimizer(
            categories=gp_cfg.get("categories", DEFAULT_CATEGORIES),
            ops_by_category=ops_by_category,
            seed=seed,
            config=config,
            focus_cfg=cfg["focus_measure"],
            metrics_cfg=cfg["metrics"],
            evaluation_cfg=cfg.get("evaluation", {}),
            tuning_cfg=cfg.get("tuning", {}),
            params_lookup=params_lookup,
            strict_params=True,
        )
        progress_rows: List[Dict[str, Any]] = []
        best_rows: List[Dict[str, Any]] = []

        def on_progress(info: dict) -> None:
            progress_rows.append(info)
            best_genome = info.get("best_genome") or []
            if best_genome:
                best_pipeline_gen = optimizer.build_pipeline(best_genome)
                best_rows.append({
                    "generation": info.get("generation"),
                    "best_score": info.get("best_score"),
                    "genome_json": json.dumps(best_genome),
                    "pipeline_json": json.dumps(best_pipeline_gen.to_dict()),
                })
            log(
                "GP_GEN: "
                f"{info.get('generation')}/{config.generations} "
                f"best={info.get('best_score'):.6f} "
                f"mean={info.get('mean_score'):.6f} "
                f"std={info.get('std_score'):.6f} "
                f"evals={info.get('evaluated')} "
                f"cache_hits={info.get('cache_hits')}"
            )

        best_genome, best_score = optimizer.run(manifest_df, gp_dir, progress_cb=on_progress)
        best_pipeline = optimizer.build_pipeline(best_genome)

        (gp_dir / "best_pipeline.json").write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")
        (gp_dir / "best_score.json").write_text(json.dumps({"best_score": best_score}, indent=2), encoding="utf-8")

        if progress_rows:
            pd.DataFrame(progress_rows).to_csv(gp_dir / "gp_progress.csv", index=False)
        if best_rows:
            pd.DataFrame(best_rows).to_csv(gp_dir / "gp_best_by_generation.csv", index=False)

        summary, _ = evaluate_pipeline(
            manifest_df,
            best_pipeline,
            focus_cfg=cfg["focus_measure"],
            metrics_cfg=cfg["metrics"],
            outer_split=["trainval"],
            inner_split=["train", "val"],
            artifacts_dir=gp_dir,
            run_tag="eval",
            failure_penalty=float(cfg.get("metrics", {}).get("failure_penalty", 1e6)),
            evaluation_cfg=cfg.get("evaluation", {}),
            pipeline_fingerprint=best_pipeline.fingerprint(),
            tuning_cfg=cfg.get("tuning", {}),
        )
        eval_summary = write_eval_summary(
            gp_dir,
            run_id,
            summary.get("objective_mode"),
            summary.get("unsupervised_objective"),
        )
        eval_summary["generalization_score"] = float(best_score)
        (gp_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

        def _to_float(value: Any) -> float:
            try:
                if value is None:
                    return float("nan")
                return float(value)
            except Exception:
                return float("nan")

        line = format_run_summary_line(
            run_id=run_id,
            generalization_score=_to_float(eval_summary.get("generalization_score", best_score)),
            stability_ratio=_to_float(eval_summary.get("stability_ratio", float("nan"))),
            ms_per_image=_to_float(eval_summary.get("ms_per_image", float("nan"))),
            num_failed=int(eval_summary.get("failures_failed", 0)),
            num_total=int(eval_summary.get("failures_total", 0)),
        )
        log(line)
        write_run_summary(gp_dir, run_id, {"line": line})
        gp_log_file.close()

    # Report generation.
    if not args.skip_report:
        report_path = build_report(gp_dir)
        print(f"REPORT_DONE: {report_path}")

    print(f"FULL_RUN_DONE: parent_run={run_dir}, gp_run={gp_dir}, per_op_run={per_op_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--per-op-run', default=None, help='Per-op Optuna run directory containing optuna_per_op.')
    args = parser.parse_args()

    from ..utils.config import load_config
    cfg = load_config(args.config)
    from ..utils.config_validate import validate_config
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    smoke_rows = cfg.get('smoke_rows') if args.smoke else None

    seed = int(cfg.get('seed', 0))
    from ..utils.seed import set_global_seed
    set_global_seed(seed)
    from ..utils.logging import init_run
    from ..preprocess.ops import backend_info
    run_id, run_dir = init_run(cfg.get('artifacts_dir', 'artifacts'), cfg, seed, extra_metadata=backend_info())
    from ..utils.run_artifacts import dump_config_used, format_run_summary_line, write_run_summary
    dump_config_used(cfg, run_dir)

    from ..eval.evaluate import build_manifest_for_run
    from ..gp.genome import DEFAULT_CATEGORIES, DEFAULT_OPS_BY_CATEGORY
    from ..gp.preprocess_gp import GPConfig, GPOptimizer
    manifest_df = build_manifest_for_run(cfg, run_dir, smoke_rows=smoke_rows, manifest_override=args.manifest)

    if cfg.get('evaluation', {}).get('mode', 'raw') == 'cached_scores':
        raise ValueError('GP search requires evaluation.mode=raw (cached_scores is eval-only).')

    gp_cfg = cfg.get('gp', {})
    if not gp_cfg:
        raise ValueError("Missing gp configuration block in config.")
    config = GPConfig(
        generations=int(gp_cfg.get('generations', 400)),
        population_size=int(gp_cfg.get('population_size', 50)),
        elite_size=int(gp_cfg.get('elite_size', 4)),
        mutation_rate=float(gp_cfg.get('mutation_rate', 0.2)),
        crossover_rate=float(gp_cfg.get('crossover_rate', 0.8)),
        cache=bool(gp_cfg.get('cache', True)),
    )
    gp_log_path = Path(run_dir) / "gp_log.txt"
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
    from ..preprocess.tuned_params import load_tuned_params
    from ..preprocess.param_spaces import PARAM_SPACES
    from ..preprocess.ops import is_op_available
    from ..gp.genome import filter_ops_by_availability

    use_tuned = bool(gp_cfg.get('use_tuned_params', False) or args.per_op_run)
    params_lookup = {}
    if use_tuned:
        per_op_run = args.per_op_run or gp_cfg.get('tuned_params_dir')
        if not per_op_run:
            raise ValueError("GP tuned params enabled but no per-op run directory provided.")
        strict_params = bool(gp_cfg.get('strict_tuned_params', True))
        params_lookup = load_tuned_params(per_op_run, strict=strict_params)
        # Validate that tunable operators present in GP search have tuned params.
        missing = []
        filtered_ops = filter_ops_by_availability(DEFAULT_OPS_BY_CATEGORY)
        for ops in filtered_ops.values():
            for op_name in ops:
                if not is_op_available(op_name):
                    continue
                if op_name in PARAM_SPACES and op_name not in params_lookup:
                    missing.append(op_name)
        if strict_params and missing:
            raise ValueError(f"Missing tuned params for GP operators: {sorted(set(missing))}")
        (Path(run_dir) / "tuned_params_used.json").write_text(
            json.dumps(params_lookup, indent=2), encoding="utf-8"
        )
        log(f"GP_TUNED_PARAMS: enabled=True, per_op_run={per_op_run}, ops={len(params_lookup)}")
    else:
        strict_params = False
        log("GP_TUNED_PARAMS: enabled=False (using default operator parameters)")

    optimizer = GPOptimizer(
        categories=gp_cfg.get('categories', DEFAULT_CATEGORIES),
        ops_by_category=DEFAULT_OPS_BY_CATEGORY,
        seed=seed,
        config=config,
        focus_cfg=cfg['focus_measure'],
        metrics_cfg=cfg['metrics'],
        evaluation_cfg=cfg.get('evaluation', {}),
        tuning_cfg=cfg.get('tuning', {}),
        params_lookup=params_lookup,
        strict_params=strict_params,
    )
    progress_rows = []
    best_rows = []

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

    best_genome, best_score = optimizer.run(manifest_df, run_dir, progress_cb=on_progress)
    best_pipeline = optimizer.build_pipeline(best_genome)

    (run_dir / 'best_pipeline.json').write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding='utf-8')
    (run_dir / 'best_score.json').write_text(json.dumps({'best_score': best_score}, indent=2), encoding='utf-8')

    if progress_rows:
        import pandas as pd
        pd.DataFrame(progress_rows).to_csv(Path(run_dir) / "gp_progress.csv", index=False)
    if best_rows:
        import pandas as pd
        pd.DataFrame(best_rows).to_csv(Path(run_dir) / "gp_best_by_generation.csv", index=False)

    # Evaluate the best pipeline once to produce per-stack artifacts for summaries.
    from ..eval.evaluate import evaluate_pipeline
    summary, _ = evaluate_pipeline(
        manifest_df,
        best_pipeline,
        focus_cfg=cfg['focus_measure'],
        metrics_cfg=cfg['metrics'],
        outer_split=['trainval'],
        inner_split=['train', 'val'],
        artifacts_dir=run_dir,
        run_tag='eval',
        failure_penalty=1e6,
        evaluation_cfg=cfg.get('evaluation', {}),
        pipeline_fingerprint=best_pipeline.fingerprint(),
        tuning_cfg=cfg.get('tuning', {}),
    )
    from ..eval.summary import write_eval_summary
    eval_summary = write_eval_summary(
        run_dir,
        run_id,
        summary.get("objective_mode"),
        summary.get("unsupervised_objective"),
    )
    eval_summary["generalization_score"] = float(best_score)
    (run_dir / 'eval_summary.json').write_text(json.dumps(eval_summary, indent=2), encoding='utf-8')
    def _to_float(value):
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
    write_run_summary(run_dir, run_id, {"line": line})
    gp_log_file.close()


if __name__ == '__main__':
    main()

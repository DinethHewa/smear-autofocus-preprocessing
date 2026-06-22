from __future__ import annotations

import argparse
import sys
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--inner-split', default=None)
    parser.add_argument('--smoke', action='store_true')
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
    from ..utils.run_artifacts import dump_config_used, build_run_summary_line, write_run_summary
    dump_config_used(cfg, run_dir)

    from ..eval.evaluate import build_manifest_for_run, evaluate_pipeline
    from ..preprocess.pipeline import Pipeline
    manifest_df = build_manifest_for_run(cfg, run_dir, smoke_rows=smoke_rows, manifest_override=args.manifest)

    outer_split = [args.split]
    inner_split = [args.inner_split] if args.inner_split else None
    available_outer = sorted(manifest_df['outer_split'].unique().tolist())
    if args.split not in available_outer:
        if args.smoke:
            outer_split = available_outer
            print(f'Warning: outer split \"{args.split}\" not found in smoke data; using {outer_split}')
        else:
            raise ValueError(f'Outer split \"{args.split}\" not found in manifest. Available: {available_outer}')

    pipeline = Pipeline.from_dict(cfg['pipeline']).resolved_defaults()
    summary, _ = evaluate_pipeline(
        manifest_df,
        pipeline,
        focus_cfg=cfg['focus_measure'],
        metrics_cfg=cfg['metrics'],
        outer_split=outer_split,
        inner_split=inner_split,
        artifacts_dir=run_dir,
        run_tag=None,
        failure_penalty=1e6,
        evaluation_cfg=cfg.get('evaluation', {}),
        pipeline_fingerprint=pipeline.fingerprint(),
        tuning_cfg=cfg.get('tuning', {}),
    )

    from ..eval.summary import write_eval_summary
    eval_summary = write_eval_summary(
        run_dir,
        run_id,
        summary.get("objective_mode"),
        summary.get("unsupervised_objective"),
    )
    line = build_run_summary_line(run_dir, run_id)
    print(line)
    write_run_summary(run_dir, run_id, {"line": line})


if __name__ == '__main__':
    main()

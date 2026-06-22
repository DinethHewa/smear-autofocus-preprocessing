from __future__ import annotations

import argparse
import csv
import json
import sys


def _resolve_per_step_names(tuning_cfg: dict) -> list[str]:
    names = tuning_cfg.get("steps") or []
    if names:
        return [str(name) for name in names if isinstance(name, str) and name]
    from ..preprocess.param_spaces import PARAM_SPACES
    from ..preprocess.ops import OP_REGISTRY, is_op_available
    return [name for name in sorted(PARAM_SPACES.keys()) if name in OP_REGISTRY and is_op_available(name)]


def _build_single_step_pipeline(step_name: str, base_pipeline) -> "Pipeline":
    from ..preprocess.param_spaces import DEFAULT_PARAMS, param_space_keys
    from ..preprocess.pipeline import Pipeline, PipelineStep

    base_step = next((step for step in base_pipeline.steps if step.name == step_name), None)
    params = dict(DEFAULT_PARAMS.get(step_name, {}))
    if base_step is not None and isinstance(base_step.params, dict):
        params.update(base_step.params)
    for key in param_space_keys(step_name):
        params[key] = None
    category = base_step.category if base_step is not None else None
    return Pipeline(steps=[PipelineStep(name=step_name, enabled=True, params=params, category=category)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    from ..utils.config import load_config
    cfg = load_config(args.config)
    from ..utils.config_validate import validate_config
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    smoke_rows = cfg.get("smoke_rows") if args.smoke else None

    seed = int(cfg.get("seed", 0))
    from ..utils.seed import set_global_seed
    set_global_seed(seed)
    from ..utils.logging import init_run
    from ..preprocess.ops import backend_info
    run_id, run_dir = init_run(cfg.get("artifacts_dir", "artifacts"), cfg, seed, extra_metadata=backend_info())
    from ..utils.run_artifacts import dump_config_used, format_run_summary_line, write_run_summary
    dump_config_used(cfg, run_dir)

    from ..eval.evaluate import build_manifest_for_run
    from ..preprocess.pipeline import Pipeline
    from ..preprocess.optuna_tuner import OptunaTuner, collect_tunable_params
    manifest_df = build_manifest_for_run(cfg, run_dir, smoke_rows=smoke_rows, manifest_override=args.manifest)

    if cfg.get("evaluation", {}).get("mode", "raw") == "cached_scores":
        raise ValueError("Optuna tuning requires evaluation.mode=raw (cached_scores is eval-only).")

    pipeline = Pipeline.from_dict(cfg["pipeline"])
    opt_cfg = cfg["optuna"]
    if args.n_trials is not None:
        opt_cfg = dict(opt_cfg)
        opt_cfg["n_trials"] = int(args.n_trials)

    tuning_cfg = cfg.get("tuning", {})
    mode = tuning_cfg.get("mode", "joint")
    if mode not in {"joint", "per_step"}:
        raise ValueError(f"Unsupported tuning.mode: {mode}")

    if mode == "joint":
        try:
            collect_tunable_params(tuning_cfg, pipeline.steps)
        except ValueError as exc:
            raise ValueError(
                "No tunable parameters found. Enable steps and set params to null, "
                "or use run_optuna_per_op for per-operator tuning."
            ) from exc

    if mode == "per_step":
        from ..preprocess.param_spaces import param_space_keys
        from ..preprocess.ops import OP_REGISTRY
        from ..eval.summary import write_eval_summary

        per_step_root = run_dir / "per_step"
        per_step_root.mkdir(parents=True, exist_ok=True)
        step_names = _resolve_per_step_names(tuning_cfg)
        if not step_names:
            raise ValueError("No preprocessing activities found for per_step tuning.")

        per_step_best_params: dict[str, dict] = {}
        summary_rows = []
        best_overall_value = None
        best_overall = None

        root_trial_path = run_dir / "trial_results.csv"
        root_trial_file = root_trial_path.open("w", newline="", encoding="utf-8")
        root_writer = csv.DictWriter(
            root_trial_file,
            fieldnames=["step_name", "trial_number", "value", "duration_sec", "params_json"],
        )
        root_writer.writeheader()

        for step_name in step_names:
            if step_name not in OP_REGISTRY:
                summary_rows.append({
                    "step_name": step_name,
                    "status": "skipped_unknown_op",
                    "best_value": float("nan"),
                    "trial_results_csv": "",
                    "eval_summary_json": "",
                })
                continue
            keys = param_space_keys(step_name)
            if not keys:
                summary_rows.append({
                    "step_name": step_name,
                    "status": "skipped_no_tunables",
                    "best_value": float("nan"),
                    "trial_results_csv": "",
                    "eval_summary_json": "",
                })
                continue

            step_dir = per_step_root / step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            dump_config_used(cfg, step_dir)
            step_pipeline = _build_single_step_pipeline(step_name, pipeline)
            trial_results_path = step_dir / "trial_results.csv"
            tuner = OptunaTuner(
                step_pipeline,
                focus_cfg=cfg["focus_measure"],
                metrics_cfg=cfg["metrics"],
                n_trials=int(opt_cfg["n_trials"]),
                timeout_seconds=int(opt_cfg["timeout_seconds"]),
                direction=opt_cfg.get("direction", "minimize"),
                seed=seed,
                evaluation_cfg=cfg.get("evaluation", {}),
                tuning_cfg=tuning_cfg,
            )
            best_pipeline, best_info = tuner.run(manifest_df, step_dir, trial_results_path=trial_results_path)

            (step_dir / "best_params.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
            (step_dir / "best_pipeline.json").write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")
            study_meta = {
                "n_trials": int(opt_cfg["n_trials"]),
                "timeout_seconds": int(opt_cfg["timeout_seconds"]),
                "seed": seed,
                "step_name": step_name,
            }
            (step_dir / "study_metadata.json").write_text(json.dumps(study_meta, indent=2), encoding="utf-8")

            eval_summary = write_eval_summary(step_dir, step_name, None, None)
            eval_summary["generalization_score"] = float(best_info.get("best_value", float("nan")))
            (step_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

            per_step_best_params[step_name] = best_info.get("best_params_by_step", {})
            best_value = float(best_info.get("best_value", float("nan")))
            summary_rows.append({
                "step_name": step_name,
                "status": "ok",
                "best_value": best_value,
                "trial_results_csv": str(trial_results_path),
                "eval_summary_json": str(step_dir / "eval_summary.json"),
            })

            if trial_results_path.exists():
                with trial_results_path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        row["step_name"] = step_name
                        root_writer.writerow(row)

            if best_overall_value is None or best_value < best_overall_value:
                best_overall_value = best_value
                best_overall = {
                    "step_name": step_name,
                    "best_pipeline": best_pipeline,
                    "best_info": best_info,
                    "eval_summary": eval_summary,
                }

        root_trial_file.flush()
        root_trial_file.close()

        per_step_summary_path = run_dir / "per_step_summary.csv"
        with per_step_summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["step_name", "status", "best_value", "trial_results_csv", "eval_summary_json"],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        (run_dir / "per_step_best_params.json").write_text(
            json.dumps(per_step_best_params, indent=2),
            encoding="utf-8",
        )

        if best_overall is None:
            raise ValueError("Per-step tuning produced no valid results.")

        (run_dir / "best_params.json").write_text(json.dumps(best_overall["best_info"], indent=2), encoding="utf-8")
        (run_dir / "best_pipeline.json").write_text(
            json.dumps(best_overall["best_pipeline"].to_dict(), indent=2),
            encoding="utf-8",
        )
        study_meta = {
            "n_trials": int(opt_cfg["n_trials"]),
            "timeout_seconds": int(opt_cfg["timeout_seconds"]),
            "seed": seed,
            "mode": "per_step",
            "steps": step_names,
        }
        (run_dir / "study_metadata.json").write_text(json.dumps(study_meta, indent=2), encoding="utf-8")

        eval_summary = dict(best_overall["eval_summary"])
        eval_summary["generalization_score"] = float(best_overall_value)
        (run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

        def _to_float(value):
            try:
                if value is None:
                    return float("nan")
                return float(value)
            except Exception:
                return float("nan")

        line = format_run_summary_line(
            run_id=run_id,
            generalization_score=_to_float(eval_summary.get("generalization_score", best_overall_value)),
            stability_ratio=_to_float(eval_summary.get("stability_ratio", float("nan"))),
            ms_per_image=_to_float(eval_summary.get("ms_per_image", float("nan"))),
            num_failed=int(eval_summary.get("failures_failed", 0)),
            num_total=int(eval_summary.get("failures_total", 0)),
        )
        print(line)
        write_run_summary(run_dir, run_id, {"line": line})
        return

    trial_results_path = run_dir / "trial_results.csv"
    tuner = OptunaTuner(
        pipeline,
        focus_cfg=cfg["focus_measure"],
        metrics_cfg=cfg["metrics"],
        n_trials=int(opt_cfg["n_trials"]),
        timeout_seconds=int(opt_cfg["timeout_seconds"]),
        direction=opt_cfg.get("direction", "minimize"),
        seed=seed,
        evaluation_cfg=cfg.get("evaluation", {}),
        tuning_cfg=tuning_cfg,
    )
    best_pipeline, best_info = tuner.run(manifest_df, run_dir, trial_results_path=trial_results_path)

    (run_dir / "best_params.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    (run_dir / "best_pipeline.json").write_text(json.dumps(best_pipeline.to_dict(), indent=2), encoding="utf-8")
    study_meta = {
        "n_trials": int(opt_cfg["n_trials"]),
        "timeout_seconds": int(opt_cfg["timeout_seconds"]),
        "seed": seed,
        "mode": "joint",
    }
    (run_dir / "study_metadata.json").write_text(json.dumps(study_meta, indent=2), encoding="utf-8")
    from ..eval.summary import write_eval_summary
    eval_summary = write_eval_summary(run_dir, run_id, None, None)
    eval_summary["generalization_score"] = float(best_info.get("best_value", float("nan")))
    (run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    def _to_float(value):
        try:
            if value is None:
                return float("nan")
            return float(value)
        except Exception:
            return float("nan")

    line = format_run_summary_line(
        run_id=run_id,
        generalization_score=_to_float(eval_summary.get("generalization_score", best_info.get("best_value", float("nan")))),
        stability_ratio=_to_float(eval_summary.get("stability_ratio", float("nan"))),
        ms_per_image=_to_float(eval_summary.get("ms_per_image", float("nan"))),
        num_failed=int(eval_summary.get("failures_failed", 0)),
        num_total=int(eval_summary.get("failures_total", 0)),
    )
    print(line)
    write_run_summary(run_dir, run_id, {"line": line})


if __name__ == "__main__":
    main()

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofocus.preprocess.ops import OP_REGISTRY
from autofocus.preprocess.param_spaces import PARAM_SPACES, param_space_keys
from autofocus.preprocess.pipeline import Pipeline
from autofocus.metrics.focus_measures import FOCUS_MEASURES, compute_focus_curve
from autofocus.q1.aggregation import (
    compute_rank_based_summary,
    dataset_metric_summary_from_per_stack,
)
from autofocus.q1.config import make_run_context
from autofocus.q1.curves import normalize_focus_curve, predict_peak_index
from autofocus.q1.gp_search import run_q1_gp_search
from autofocus.q1.labels import build_manifest, build_reference_labels
from autofocus.q1.metrics import AUTOFOCUS_METRICS, METRIC_DIRECTION, compute_q1_metrics_for_stack
from autofocus.q1.stages import run_all_stages, stage01_build_stacks_and_manifest
from autofocus.q1.stages import _has_fresh_transfer_resume_evidence, _has_true_gp_resume_evidence
from autofocus.q1.statistics import bootstrap_ci_mean, friedman_wilcoxon_holm


class ProbeTrial:
    def suggest_int(self, name, low, high=None, **kwargs):
        return int(low)

    def suggest_float(self, name, low, high=None, **kwargs):
        return float(low)

    def suggest_categorical(self, name, choices, **kwargs):
        return list(choices)[0]


def _smoke_config(tmp_path: Path, run_id: str = "pytest_q1_smoke") -> Path:
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    cfg["run"]["output_root"] = str(tmp_path / "paper_outputs")
    cfg["run"]["run_id"] = run_id
    path = tmp_path / "q1_smoke.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _smoke_true_gp_config(tmp_path: Path, run_id: str = "pytest_q1_true_gp") -> Path:
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    cfg["run"]["output_root"] = str(tmp_path / "paper_outputs")
    cfg["run"]["run_id"] = run_id
    for section in ("generalized", "dataset_specific"):
        cfg["optimization"][section].update(
            {
                "optimizer": "genetic_programming",
                "gp_generations": 1,
                "population_size": 2,
                "elite_size": 1,
                "max_pipeline_length": 2,
                "mutation_rate": 0.2,
                "crossover_rate": 0.8,
            }
        )
    path = tmp_path / "q1_smoke_true_gp.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    cfg_path = _smoke_config(tmp_path_factory.mktemp("q1_smoke_run"))
    ctx = make_run_context(cfg_path, smoke=True, overwrite=True)
    run_all_stages(ctx)
    return ctx


def test_parameter_spaces_match_operator_signatures():
    for op_name, space_func in PARAM_SPACES.items():
        params = space_func(ProbeTrial())
        signature_params = set(inspect.signature(OP_REGISTRY[op_name]).parameters) - {"img"}
        assert set(params).issubset(signature_params), op_name


def test_laplacian_sharpen_space_no_sigma():
    assert param_space_keys("laplacian_sharpen") == ["ksize", "amount"]
    assert "sigma" not in param_space_keys("laplacian_sharpen")


def test_laplacian_sharpen_pipeline_accepts_parameter_space():
    img = np.zeros((16, 16), dtype=np.uint8)
    img[6:10, 6:10] = 200
    params = PARAM_SPACES["laplacian_sharpen"](ProbeTrial())
    pipeline = Pipeline.from_dict(
        {"steps": [{"name": "laplacian_sharpen", "enabled": True, "params": params}]}
    )
    out = pipeline.apply(img)
    assert out.shape == img.shape


def test_focus_curve_normalization_and_tie_breaking():
    normalized = normalize_focus_curve(np.asarray([2.0, 4.0, 4.0]))
    assert np.allclose(normalized, [0.0, 1.0, 1.0])
    assert predict_peak_index(normalized) == 2


def test_q1_metric_names_and_directionality():
    assert list(AUTOFOCUS_METRICS) == [
        "absolute_peak_localization_error",
        "fwhm",
        "curvature_at_peak",
        "steep_slope_width",
        "steep_to_gradual_slope_ratio",
        "false_maxima_count",
        "noise_level",
        "rrmse_under_additive_noise",
        "range_around_global_maximum",
        "execution_time_per_slice",
    ]
    assert METRIC_DIRECTION["curvature_at_peak"] is False
    metrics = compute_q1_metrics_for_stack(
        norm_curve=np.asarray([0.0, 0.5, 1.0, 0.5, 0.0]),
        reference_idx=2,
        execution_time=0.01,
        compute_rrmse=False,
    )
    assert metrics["absolute_peak_localization_error"] == 0.0
    assert metrics["execution_time_per_slice"] == 0.01


def test_cfm4_focus_measure_is_registered_and_finite():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 180
    assert "cfm4" in FOCUS_MEASURES
    assert "cfm4_bg_gse_ftsi_wde_db1" in FOCUS_MEASURES
    value = FOCUS_MEASURES["cfm4"](img)
    assert np.isfinite(value)


def test_cfm4_torch_cuda_backend_falls_back_or_returns_finite_curve():
    stack = np.zeros((3, 32, 32), dtype=np.float32)
    stack[:, 8:24, 8:24] = np.asarray([0.2, 0.8, 0.4], dtype=np.float32)[:, None, None]
    curve = compute_focus_curve(stack, name="cfm4", backend="torch_cuda_auto", cuda_fallback=True)
    assert curve.shape == (3,)
    assert np.isfinite(curve).all()


def test_label_provenance_source_and_surrogate():
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, smoke=True)
    labels, summary, loo = build_reference_labels(manifest, cfg)
    assert set(labels["label_source"]) == {"source"}
    assert {"dataset", "label_source", "n_stacks"}.issubset(summary.columns)

    surrogate_manifest = manifest.drop(columns=["best_focus_index", "confidence"])
    surrogate, surrogate_summary, surrogate_loo = build_reference_labels(surrogate_manifest, cfg)
    assert set(surrogate["label_source"]) == {"surrogate"}
    assert not surrogate_loo.empty
    assert set(surrogate_summary["label_source"]) == {"surrogate"}


def test_group_disjoint_splitting_and_smoke_manifest():
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, smoke=True)
    for group_id, subset in manifest.groupby("group_id"):
        assert subset["outer_split"].nunique() == 1, group_id
        assert subset["inner_split"].nunique() == 1, group_id


def test_lodo_split_construction_from_manifest():
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, smoke=True)
    datasets = sorted(manifest["dataset"].unique())
    for heldout in datasets:
        train = [dataset for dataset in datasets if dataset != heldout]
        assert heldout not in train
        assert train


def test_dataset_specific_split_construction_from_manifest():
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, smoke=True)
    for dataset, subset in manifest.groupby("dataset"):
        assert {"trainval", "test"}.issubset(set(subset["outer_split"])), dataset


def test_aggregation_calculations_lower_score_wins():
    rows = []
    for method, peak_error, runtime in [("good", 0.0, 0.01), ("bad", 2.0, 0.02)]:
        rows.append(
            {
                "dataset": "D1",
                "method": method,
                "failure": 0,
                **{metric: 1.0 for metric in AUTOFOCUS_METRICS},
                "absolute_peak_localization_error": peak_error,
                "execution_time_per_slice": runtime,
            }
        )
    summary = dataset_metric_summary_from_per_stack(rows)
    rank_rows, _ = compute_rank_based_summary(summary)
    assert rank_rows[0]["method"] == "good"


def test_q1_gp_search_computes_candidate_metrics_from_workflows(tmp_path):
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, smoke=True).head(2).copy()
    labels, _, _ = build_reference_labels(manifest, cfg)
    events = []
    result = run_q1_gp_search(
        manifest_df=manifest,
        labels_df=labels,
        config=cfg,
        opt_cfg={
            "gp_generations": 1,
            "population_size": 2,
            "elite_size": 1,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "max_pipeline_length": 2,
        },
        search_dir=tmp_path / "gp_search",
        search_name="pytest_gp",
        seed=123,
        datasets=None,
        outer_split=None,
        inner_split=None,
        resume=False,
        progress_cb=events.append,
    )
    assert result.best_score >= 1.0
    assert result.candidate_summary_csv.exists()
    candidate_summary = pd.read_csv(result.candidate_summary_csv)
    assert {"candidate_id", "pipeline_json", "absolute_peak_localization_error_mean"}.issubset(candidate_summary.columns)
    assert any(event.get("event") == "candidate" and event.get("cached") is False for event in events)


def test_q1_gp_search_forces_serial_workers_for_cuda_backend(tmp_path):
    cfg = yaml.safe_load(Path("configs/q1_smoke.yaml").read_text(encoding="utf-8"))
    cfg["focus_measure"]["backend"] = "torch_cuda_auto"
    manifest = build_manifest(cfg, smoke=True).head(2).copy()
    labels, _, _ = build_reference_labels(manifest, cfg)
    result = run_q1_gp_search(
        manifest_df=manifest,
        labels_df=labels,
        config=cfg,
        opt_cfg={
            "gp_generations": 1,
            "population_size": 2,
            "elite_size": 1,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "max_pipeline_length": 2,
            "candidate_num_workers": 4,
        },
        search_dir=tmp_path / "gp_search_cuda_policy",
        search_name="pytest_gp_cuda_policy",
        seed=123,
        datasets=None,
        outer_split=None,
        inner_split=None,
        resume=False,
    )
    metadata = yaml.safe_load((result.state_json.parent / "gp_search_metadata.json").read_text(encoding="utf-8"))
    assert metadata["candidate_num_workers_requested"] == 4
    assert metadata["candidate_num_workers"] == 1
    assert metadata["cuda_serial_worker_policy"] is True


def test_true_gp_resume_evidence_survives_stale_skipped_summary(tmp_path):
    cfg_path = _smoke_config(tmp_path, run_id="stale_summary_case")
    ctx = make_run_context(cfg_path, smoke=True, resume=True)
    stage = "05_generalized_workflow_search_lodo"
    summary_path = ctx.output_dir / "summaries" / f"{stage}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"stage": stage, "status": "skipped_existing"}),
        encoding="utf-8",
    )
    out_path = ctx.output_dir / "paper1_generalized" / "lodo_results_per_stack.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "method_type": "q1_gp_generalized_lodo",
                "workflow_origin": "fresh_gp_final_evaluation",
                "source_method": "generalized_gp_workflow",
            }
        ]
    ).to_csv(out_path, index=False)
    progress_path = out_path.parent / "lodo_results.progress.json"
    progress_path.write_text(
        yaml.safe_dump(
            {
                "status": "running",
                "optimizer_status": "genetic_programming",
                "source": "new_gp_candidate_curves_and_q1_metrics",
            }
        ),
        encoding="utf-8",
    )
    assert _has_true_gp_resume_evidence(ctx, stage, out_path, progress_path)


def test_stage07_fresh_transfer_resume_evidence_survives_stale_summary(tmp_path):
    cfg_path = _smoke_config(tmp_path, run_id="stale_transfer_case")
    ctx = make_run_context(cfg_path, smoke=True, resume=True)
    stage = "07_compare_generalized_vs_specific"
    summary_path = ctx.output_dir / "summaries" / f"{stage}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"stage": stage, "status": "complete", "source": "stage04_per_stack_metrics"}),
        encoding="utf-8",
    )
    progress_path = ctx.output_dir / "paper2_dataset_specific" / "transfer_matrix.progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        yaml.safe_dump({"status": "running", "source": "fresh_workflow_evaluation"}),
        encoding="utf-8",
    )
    per_stack_path = ctx.output_dir / "paper2_dataset_specific" / "transfer_results_per_stack.csv"
    pd.DataFrame(
        [
            {
                "method_type": "q1_gp_dataset_specific",
                "workflow_origin": "fresh_gp_final_evaluation",
                "source_method": "BMA_specific_gp_workflow",
            }
        ]
    ).to_csv(per_stack_path, index=False)
    assert _has_fresh_transfer_resume_evidence(ctx, stage, progress_path, per_stack_path)


def test_bootstrap_statistics_smoke_and_small_sample_warning():
    ci_low, ci_high = bootstrap_ci_mean([1.0, 2.0, 3.0], n_resamples=20, seed=1)
    assert ci_low <= ci_high
    friedman_rows, pairwise_rows, warnings = friedman_wilcoxon_holm(
        np.asarray([[1.0, 2.0], [2.0, 3.0]]),
        ["a", "b"],
        ["d1", "d2"],
        family="test",
    )
    assert friedman_rows[0]["valid"] is False
    assert pairwise_rows == []
    assert warnings


def test_failure_penalty_handling_in_summaries():
    rows = [
        {
            "dataset": "D1",
            "method": "failed",
            "failure": 1,
            **{metric: float("nan") for metric in AUTOFOCUS_METRICS},
            "execution_time_per_slice": 0.2,
        },
        {
            "dataset": "D1",
            "method": "ok",
            "failure": 0,
            **{metric: 0.0 for metric in AUTOFOCUS_METRICS},
            "execution_time_per_slice": 0.1,
        },
    ]
    summary = dataset_metric_summary_from_per_stack(rows)
    failure_lookup = {row["method"]: row["failure_count"] for row in summary}
    assert failure_lookup["failed"] == 1


def test_resumability_uses_existing_outputs(tmp_path):
    cfg_path = _smoke_config(tmp_path, run_id="resume_case")
    ctx = make_run_context(cfg_path, smoke=True, overwrite=True)
    first = stage01_build_stacks_and_manifest(ctx)
    assert Path(first["manifest_path"]).exists()
    resumed_ctx = make_run_context(cfg_path, smoke=True, resume=True)
    resumed = stage01_build_stacks_and_manifest(resumed_ctx)
    assert resumed["resumed_from_existing"] is True


def test_full_smoke_pipeline_exports_expected_outputs(smoke_run):
    root = smoke_run.output_dir
    expected = [
        root / "q1_metrics" / "per_stack_metrics.csv",
        root / "q1_metrics" / "per_stack_metrics.progress.json",
        root / "q1_metrics" / "dataset_metric_summary.csv",
        root / "q1_metrics" / "method_summary.csv",
        root / "labels" / "reference_labels.csv",
        root / "labels" / "reference_labels.progress.json",
        root / "reproducibility" / "manifest.progress.json",
        root / "paper1_generalized" / "lodo_results_per_stack.csv",
        root / "paper1_generalized" / "lodo_results.progress.json",
        root / "paper2_dataset_specific" / "dataset_specific_results.progress.json",
        root / "paper2_dataset_specific" / "transfer_matrix_rank_score.csv",
        root / "paper2_dataset_specific" / "transfer_matrix.progress.json",
        root / "statistics" / "bootstrap_ci.csv",
        root / "statistics" / "statistics.progress.json",
        root / "tables" / "Table_S1_ParameterSpace.csv",
        root / "summaries" / "09_export_paper_assets.progress.json",
        root / "paper1_generalized" / "assets" / "Figure_02_LODOGeneralization.svg",
        root / "paper2_dataset_specific" / "assets" / "Figure_02_TransferMatrix.svg",
        root / "reproducibility" / "run_summary.json",
        root / "reproducibility" / "source_tree_hash.json",
    ]
    for path in expected:
        assert path.exists(), path


def test_full_smoke_pipeline_exercises_true_gp_branches(tmp_path):
    cfg_path = _smoke_true_gp_config(tmp_path)
    ctx = make_run_context(cfg_path, smoke=True, overwrite=True)
    run_all_stages(ctx)
    stage05 = yaml.safe_load((ctx.output_dir / "summaries" / "05_generalized_workflow_search_lodo.json").read_text(encoding="utf-8"))
    stage06 = yaml.safe_load((ctx.output_dir / "summaries" / "06_dataset_specific_workflow_search.json").read_text(encoding="utf-8"))
    stage07 = yaml.safe_load((ctx.output_dir / "summaries" / "07_compare_generalized_vs_specific.json").read_text(encoding="utf-8"))
    assert stage05["optimizer_status"] == "genetic_programming"
    assert stage05["source"] == "new_gp_candidate_curves_and_q1_metrics"
    assert stage06["optimizer_status"] == "genetic_programming"
    assert stage06["source"] == "new_gp_candidate_curves_and_q1_metrics"
    assert stage07["source"] == "fresh_workflow_evaluation"
    assert (ctx.output_dir / "paper1_generalized" / "lodo_selected_focus_curves.csv").exists()
    assert (ctx.output_dir / "paper2_dataset_specific" / "dataset_specific_selected_focus_curves.csv").exists()
    assert (ctx.output_dir / "paper2_dataset_specific" / "transfer_focus_curves.csv").exists()


def test_smoke_pipeline_records_reference_metadata(smoke_run):
    metrics = pd.read_csv(smoke_run.output_dir / "q1_metrics" / "per_stack_metrics.csv")
    assert {"raw_curve", "normalized_curve"}.issubset(
        pd.read_csv(smoke_run.output_dir / "q1_curves" / "preprocessed_focus_curves.csv").columns
    )
    assert {"label_source", "predicted_focus_index", "reference_focus_index"}.issubset(metrics.columns)
    assert metrics["failure"].sum() == 0

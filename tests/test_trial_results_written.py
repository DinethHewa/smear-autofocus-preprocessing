import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

from autofocus.data.manifests import build_manifest_from_direct_inputs
from autofocus.data.splits import assign_nested_splits
from autofocus.preprocess.pipeline import Pipeline, PipelineStep
from autofocus.preprocess.optuna_tuner import OptunaTuner


def test_trial_results_written(tmp_path):
    stack = np.random.rand(2, 4, 16, 16).astype(np.float32)
    stack_path = tmp_path / "stack.npy"
    np.save(stack_path, stack)

    datasets_cfg = [{"name": "demo", "path": str(stack_path)}]
    manifest = build_manifest_from_direct_inputs(datasets_cfg)
    manifest = assign_nested_splits(manifest, group_col="group_id", outer_test_size=0.2, inner_val_size=0.2, seed=1)

    pipeline = Pipeline([
        PipelineStep(name="clahe_enhancement", enabled=True, params={"clip_limit": None, "tile_grid_size": None}),
    ])

    tuner = OptunaTuner(
        pipeline,
        focus_cfg={"name": "variance_of_laplacian", "params": {}},
        metrics_cfg={"failure_penalty": 1e6},
        n_trials=2,
        timeout_seconds=30,
        seed=1,
        evaluation_cfg={"mode": "raw", "reference_paths": {}},
        tuning_cfg={"allow_empty_search_space": False, "objective_mode": "reference_or_unsupervised", "unsupervised_objective": "unimodal_peak"},
    )
    trial_results_path = tmp_path / "trial_results.csv"
    tuner.run(manifest, tmp_path, trial_results_path=trial_results_path)

    df = pd.read_csv(trial_results_path)
    assert len(df) >= 2

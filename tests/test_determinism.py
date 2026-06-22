import numpy as np
import pandas as pd

from autofocus.data.manifests import build_manifest_from_direct_inputs
from autofocus.data.splits import assign_nested_splits
from autofocus.eval.evaluate import evaluate_pipeline
from autofocus.preprocess.pipeline import Pipeline, PipelineStep
from autofocus.utils.seed import set_global_seed


def test_determinism(tmp_path):
    stack = np.random.rand(4, 3, 16, 16).astype(np.float32)
    stack_path = tmp_path / 'stack.npy'
    np.save(stack_path, stack)

    datasets_cfg = [{'name': 'demo', 'path': str(stack_path)}]
    manifest = build_manifest_from_direct_inputs(datasets_cfg)
    manifest = assign_nested_splits(manifest, group_col='group_id', outer_test_size=0.25, inner_val_size=0.25, seed=123)
    manifest['ref_peak'] = 1
    manifest['confidence'] = 1.0

    pipeline = Pipeline([PipelineStep(name='gaussian_blur', params={'ksize': 3, 'sigma': 0.0})])
    fingerprint = pipeline.fingerprint()

    set_global_seed(123)
    summary1, results1 = evaluate_pipeline(
        manifest,
        pipeline,
        focus_cfg={'name': 'variance_of_laplacian', 'params': {}},
        metrics_cfg={'failure_penalty': 1e6},
        outer_split=['trainval'],
        inner_split=['train'],
        artifacts_dir=tmp_path / 'artifacts1',
        run_tag=None,
        failure_penalty=1e6,
    )

    set_global_seed(123)
    summary2, results2 = evaluate_pipeline(
        manifest,
        pipeline,
        focus_cfg={'name': 'variance_of_laplacian', 'params': {}},
        metrics_cfg={'failure_penalty': 1e6},
        outer_split=['trainval'],
        inner_split=['train'],
        artifacts_dir=tmp_path / 'artifacts2',
        run_tag=None,
        failure_penalty=1e6,
    )

    assert fingerprint == pipeline.fingerprint()
    assert summary1['generalization_score'] == summary2['generalization_score']
    assert results1['total_fitness'].tolist() == results2['total_fitness'].tolist()

    df1 = pd.read_csv(tmp_path / 'artifacts1' / 'per_stack_results.csv')
    df2 = pd.read_csv(tmp_path / 'artifacts2' / 'per_stack_results.csv')
    for df in (df1, df2):
        if "runtime_sec" in df.columns:
            df.drop(columns=["runtime_sec"], inplace=True)
    pd.testing.assert_frame_equal(df1, df2)

    ds1 = pd.read_csv(tmp_path / 'artifacts1' / 'dataset_summary.csv')
    ds2 = pd.read_csv(tmp_path / 'artifacts2' / 'dataset_summary.csv')
    pd.testing.assert_frame_equal(ds1, ds2)

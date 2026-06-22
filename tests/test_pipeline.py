import numpy as np

from autofocus.preprocess.pipeline import Pipeline, PipelineStep


def test_pipeline_apply_and_fingerprint():
    img = np.random.rand(16, 16).astype(np.float32)
    pipeline = Pipeline(steps=[PipelineStep(name='gaussian_blur', params={'ksize': 3, 'sigma': 0.0})])
    out = pipeline.apply(img)
    assert out.shape == img.shape
    assert pipeline.fingerprint() == pipeline.fingerprint()

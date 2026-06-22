import numpy as np

from autofocus.metrics.unsupervised import unimodal_peak_objective


def test_peaked_curve_beats_flat_curve():
    peaked = np.array([0.0, 1.0, 3.0, 1.0, 0.0], dtype=float)
    flat = np.ones(5, dtype=float)
    assert unimodal_peak_objective(peaked) < unimodal_peak_objective(flat)


def test_nonfinite_returns_large_penalty():
    curve = np.array([np.nan, np.nan, np.nan], dtype=float)
    score = unimodal_peak_objective(curve)
    assert np.isfinite(score)
    assert score >= 1e6

import numpy as np

from autofocus.metrics import objectives


def test_unimodal_peak_objective_finite():
    cfg = objectives.ObjectiveConfig()
    objectives.set_objective_config(cfg)
    curve = np.array([0.0, 1.0, 5.0, 1.0, 0.0], dtype=np.float32)
    out = objectives.unimodal_peak_fitness(curve)
    assert np.isfinite(out["total_fitness"])

    flat = np.ones(10, dtype=np.float32)
    out_flat = objectives.unimodal_peak_fitness(flat)
    assert out_flat["total_fitness"] > out["total_fitness"]

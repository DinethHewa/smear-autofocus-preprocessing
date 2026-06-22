import numpy as np

from autofocus.metrics import objectives


def test_stack_fitness_peak_error():
    cfg = objectives.ObjectiveConfig(alpha_softargmax=10.0, failure_penalty=1e6)
    objectives.set_objective_config(cfg)
    curve = np.array([0.0, 1.0, 5.0, 1.0, 0.0], dtype=np.float32)
    out = objectives.stack_fitness(curve, ref_peak=2, confidence=1.0)
    assert abs(out['pred_peak'] - 2.0) < 0.2
    assert out['penalty_nonfinite'] == 0.0
    assert out['is_failure'] == 0.0


def test_stack_fitness_nonfinite():
    cfg = objectives.ObjectiveConfig(failure_penalty=123.0)
    objectives.set_objective_config(cfg)
    curve = np.array([np.nan, np.nan], dtype=np.float32)
    out = objectives.stack_fitness(curve, ref_peak=0, confidence=1.0)
    assert out['is_failure'] == 1.0
    assert out['total_fitness'] == 123.0

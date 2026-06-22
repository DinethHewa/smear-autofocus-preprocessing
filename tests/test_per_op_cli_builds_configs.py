from autofocus.preprocess.optuna_single_op import build_per_op_pipeline
from autofocus.preprocess.param_spaces import param_space_keys


def test_per_op_pipeline_enables_only_target_and_baselines():
    cfg = {
        "pipeline": {
            "steps": [
                {"name": "identity", "enabled": True, "params": {}, "category": "structure"},
                {"name": "gaussian_blur", "enabled": True, "params": {"ksize": 3, "sigma": 0.0}, "category": "denoise", "baseline": True},
                {"name": "clahe_enhancement", "enabled": True, "params": {"clip_limit": 2.0, "tile_grid_size": 8}, "category": "contrast"},
                {"name": "unsharp_mask", "enabled": True, "params": {"sigma": 1.0, "amount": 1.0}, "category": "edge"},
            ]
        }
    }

    pipeline = build_per_op_pipeline(cfg, "clahe_enhancement")
    step_map = {step.name: step for step in pipeline.steps}

    assert step_map["identity"].enabled is True
    assert step_map["gaussian_blur"].enabled is True
    assert step_map["clahe_enhancement"].enabled is True
    assert step_map["unsharp_mask"].enabled is False

    tunables = param_space_keys("clahe_enhancement")
    for key in tunables:
        assert step_map["clahe_enhancement"].params.get(key) is None

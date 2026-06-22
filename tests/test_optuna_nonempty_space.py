import pytest

pytest.importorskip("optuna")

from autofocus.utils.config import load_config
from autofocus.preprocess.pipeline import Pipeline
from autofocus.preprocess.optuna_tuner import collect_tunable_params


def test_optuna_nonempty_space():
    cfg = load_config("configs/default.yaml")
    pipeline = Pipeline.from_dict(cfg["pipeline"])
    tunables = collect_tunable_params(cfg.get("tuning", {}), pipeline.steps)
    assert len(tunables) > 0

import pytest

import autofocus.preprocess.optuna_single_op as single_op
import autofocus.preprocess.param_spaces as param_spaces


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        single_op.tune_single_op({}, "not_real", "tmp", n_trials=1, smoke=True)


def test_empty_param_space_raises(monkeypatch, tmp_path):
    def _empty_space(trial):
        return {}

    monkeypatch.setattr(single_op, "PARAM_SPACES", {"clahe_enhancement": _empty_space})
    monkeypatch.setattr(param_spaces, "PARAM_SPACES", {"clahe_enhancement": _empty_space})

    with pytest.raises(ValueError):
        single_op.tune_single_op({}, "clahe_enhancement", tmp_path, n_trials=1, smoke=True)

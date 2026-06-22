import numpy as np
from autofocus.preprocess.ops import OP_REGISTRY, is_op_available
from autofocus.preprocess.param_spaces import DEFAULT_PARAMS


def test_ops_shape_and_finite():
    img = np.random.rand(32, 32).astype(np.float32)
    for name, func in OP_REGISTRY.items():
        if not is_op_available(name):
            continue
        params = DEFAULT_PARAMS.get(name, {})
        out = func(img, **params)
        assert out.shape == img.shape
        assert np.isfinite(out).all()
        assert out.min() >= -1e-3
        assert out.max() <= 1.0 + 1e-3
        out2 = func(img, **params)
        assert np.allclose(out, out2)

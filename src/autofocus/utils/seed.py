from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_tf: bool = True) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

    random.seed(seed)
    np.random.seed(seed)

    if deterministic_tf:
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except Exception:
            pass


def seed_metadata(seed: int) -> dict:
    return {
        'seed': seed,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
    }

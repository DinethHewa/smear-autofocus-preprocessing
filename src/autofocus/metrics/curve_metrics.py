from __future__ import annotations

from typing import Dict

import numpy as np

from . import objectives


def softargmax_peak(curve: np.ndarray, alpha: float) -> float:
    return objectives.softargmax_peak(curve, alpha)


def curve_quality_penalties(curve: np.ndarray) -> Dict[str, float]:
    return objectives.curve_quality_penalties(curve)


def evaluate_curve(curve: np.ndarray, ref_peak: int, confidence: float) -> Dict[str, float]:
    """
    Compatibility wrapper for the new Q1-grade objective.
    """
    return objectives.stack_fitness(curve, ref_peak, confidence)

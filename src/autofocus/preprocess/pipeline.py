from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import hashlib
import json

import numpy as np

from .ops import OP_REGISTRY, apply_op, OpInvariantError
from .param_spaces import DEFAULT_PARAMS


class PreprocessError(Exception):
    def __init__(self, op_name: str, message: str) -> None:
        super().__init__(message)
        self.op_name = op_name


class ShapeMismatchError(PreprocessError):
    pass


class InvalidValueError(PreprocessError):
    pass


class RangeError(PreprocessError):
    pass


class OOMError(PreprocessError):
    pass


class IOError(PreprocessError):
    pass


class TimeoutError(PreprocessError):
    pass


class UnknownError(PreprocessError):
    pass


@dataclass
class PipelineStep:
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    category: str | None = None
    baseline: bool = False


@dataclass
class Pipeline:
    steps: List[PipelineStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'steps': [
                {'name': s.name, 'enabled': s.enabled, 'params': s.params, 'category': s.category, 'baseline': s.baseline}
                for s in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        steps = [PipelineStep(**step) for step in data.get('steps', [])]
        return cls(steps=steps)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def resolved_defaults(self) -> 'Pipeline':
        steps: List[PipelineStep] = []
        for step in self.steps:
            params = dict(step.params or {})
            defaults = DEFAULT_PARAMS.get(step.name, {})
            for key, value in list(params.items()):
                if value is None:
                    if key not in defaults:
                        raise ValueError(f"Pipeline step '{step.name}' has null param '{key}' with no default.")
                    params[key] = defaults[key]
            steps.append(PipelineStep(name=step.name, enabled=step.enabled, params=params, category=step.category, baseline=step.baseline))
        return Pipeline(steps=steps)

    def _validate(self, img: np.ndarray, ref_shape: tuple, op_name: str) -> None:
        if img.shape != ref_shape:
            raise ShapeMismatchError(op_name, f'Expected shape {ref_shape}, got {img.shape}')
        if not np.isfinite(img).all():
            raise InvalidValueError(op_name, 'Found NaN or Inf values')
        if img.dtype == np.uint8:
            if img.min() < 0 or img.max() > 255:
                raise RangeError(op_name, 'uint8 image out of range')
        else:
            if img.min() < -1e-3 or img.max() > 1.0 + 1e-3:
                raise RangeError(op_name, 'float image out of range [0,1]')

    def apply(self, img: np.ndarray) -> np.ndarray:
        out = img
        ref_shape = img.shape
        for step in self.steps:
            if not step.enabled:
                continue
            try:
                out = apply_op(step.name, out, params=step.params, enforce_shape=True)
                self._validate(out, ref_shape, step.name)
            except OpInvariantError as exc:
                code = exc.code
                if code == 'ShapeMismatchError':
                    raise ShapeMismatchError(step.name, str(exc)) from exc
                if code == 'InvalidValueError':
                    raise InvalidValueError(step.name, str(exc)) from exc
                if code == 'RangeError':
                    raise RangeError(step.name, str(exc)) from exc
                raise UnknownError(step.name, str(exc)) from exc
            except PreprocessError:
                raise
            except Exception as exc:
                raise UnknownError(step.name, str(exc)) from exc
        return out

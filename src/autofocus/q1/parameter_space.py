from __future__ import annotations

import inspect
from typing import Any, Dict, List

from ..gp.genome import DEFAULT_OPS_BY_CATEGORY
from ..preprocess.ops import OP_REGISTRY, UNAVAILABLE_OPS, is_op_available
from ..preprocess.param_spaces import DEFAULT_PARAMS, PARAM_SPACES
from .config import display_category


class RecordingTrial:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def suggest_int(self, name: str, low: int, high: int | None = None, **kwargs):
        self.rows.append({"parameter": name, "type": "int", "range": f"{low}..{high}", "options": "", "default_probe": low, "notes": kwargs})
        return int(low)

    def suggest_float(self, name: str, low: float, high: float | None = None, **kwargs):
        self.rows.append({"parameter": name, "type": "float", "range": f"{low}..{high}", "options": "", "default_probe": low, "notes": kwargs})
        return float(low)

    def suggest_categorical(self, name: str, choices, **kwargs):
        choices_list = list(choices)
        self.rows.append({"parameter": name, "type": "categorical", "range": "", "options": "|".join(str(item) for item in choices_list), "default_probe": choices_list[0] if choices_list else "", "notes": kwargs})
        return choices_list[0] if choices_list else None


def parameter_space_rows() -> List[Dict[str, Any]]:
    op_to_category: Dict[str, str] = {}
    for category, ops in DEFAULT_OPS_BY_CATEGORY.items():
        for op_name in ops:
            op_to_category.setdefault(op_name, category)
    for op_name in OP_REGISTRY:
        op_to_category.setdefault(op_name, "uncategorized")

    rows: List[Dict[str, Any]] = []
    for op_name in sorted(OP_REGISTRY):
        category = display_category(op_to_category.get(op_name))
        available = is_op_available(op_name)
        unavailable_note = "unavailable in this environment" if op_name in UNAVAILABLE_OPS else ""
        defaults = DEFAULT_PARAMS.get(op_name, {})
        signature_params = []
        try:
            sig = inspect.signature(OP_REGISTRY[op_name])
            signature_params = [name for name in sig.parameters if name != "img"]
        except Exception:
            signature_params = []
        if op_name in PARAM_SPACES:
            trial = RecordingTrial()
            try:
                PARAM_SPACES[op_name](trial)
                for entry in trial.rows:
                    clean_name = str(entry["parameter"]).split(".")[-1]
                    rows.append(
                        {
                            "category": category,
                            "operator": op_name,
                            "parameter_name": clean_name,
                            "parameter_type": entry["type"],
                            "search_range_or_options": entry["range"] or entry["options"],
                            "default_or_fixed_value": defaults.get(clean_name, ""),
                            "availability_status": "available" if available else "unavailable",
                            "notes": unavailable_note,
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "category": category,
                        "operator": op_name,
                        "parameter_name": "",
                        "parameter_type": "invalid_space",
                        "search_range_or_options": "",
                        "default_or_fixed_value": "",
                        "availability_status": "available" if available else "unavailable",
                        "notes": f"parameter-space inspection failed: {exc}",
                    }
                )
        elif signature_params:
            for param in signature_params:
                rows.append(
                    {
                        "category": category,
                        "operator": op_name,
                        "parameter_name": param,
                        "parameter_type": "fixed",
                        "search_range_or_options": "",
                        "default_or_fixed_value": defaults.get(param, ""),
                        "availability_status": "available" if available else "unavailable",
                        "notes": unavailable_note or "fixed/default parameter",
                    }
                )
        else:
            rows.append(
                {
                    "category": category,
                    "operator": op_name,
                    "parameter_name": "",
                    "parameter_type": "none",
                    "search_range_or_options": "",
                    "default_or_fixed_value": "",
                    "availability_status": "available" if available else "unavailable",
                    "notes": unavailable_note or "no tunable parameters",
                }
            )
    return rows

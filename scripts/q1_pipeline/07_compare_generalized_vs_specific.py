from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage07_compare_generalized_vs_specific


if __name__ == "__main__":
    run_stage(stage07_compare_generalized_vs_specific, "Compare generalized and dataset-specific preprocessing workflows.")

from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage03_run_preprocessed_focus_curves


if __name__ == "__main__":
    run_stage(stage03_run_preprocessed_focus_curves, "Compute raw and normalized focus curves after preprocessing.")

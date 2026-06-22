from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage05_generalized_workflow_search_lodo


if __name__ == "__main__":
    run_stage(stage05_generalized_workflow_search_lodo, "Run LODO generalized preprocessing workflow search.")

from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage06_dataset_specific_workflow_search


if __name__ == "__main__":
    run_stage(stage06_dataset_specific_workflow_search, "Run dataset-specific preprocessing workflow search.")

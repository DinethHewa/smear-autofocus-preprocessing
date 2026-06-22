from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage04_evaluate_workflows_q1_metrics


if __name__ == "__main__":
    run_stage(stage04_evaluate_workflows_q1_metrics, "Evaluate preprocessing workflows with Q1 autofocus metrics.")

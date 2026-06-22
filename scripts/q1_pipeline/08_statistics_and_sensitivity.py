from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage08_statistics_and_sensitivity


if __name__ == "__main__":
    run_stage(stage08_statistics_and_sensitivity, "Run statistical testing and sensitivity analyses.")

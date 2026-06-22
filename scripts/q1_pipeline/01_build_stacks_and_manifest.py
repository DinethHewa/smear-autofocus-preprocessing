from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage01_build_stacks_and_manifest


if __name__ == "__main__":
    run_stage(stage01_build_stacks_and_manifest, "Build stack manifest for the Q1 preprocessing pipeline.")

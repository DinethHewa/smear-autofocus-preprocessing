from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage02_build_reference_labels


if __name__ == "__main__":
    run_stage(stage02_build_reference_labels, "Build source or surrogate reference focus labels with provenance.")

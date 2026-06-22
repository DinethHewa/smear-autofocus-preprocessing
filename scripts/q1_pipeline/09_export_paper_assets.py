from __future__ import annotations

from _common import run_stage
from autofocus.q1.stages import stage09_export_paper_assets


if __name__ == "__main__":
    run_stage(stage09_export_paper_assets, "Export paper-ready figures and tables for both Q1 papers.")

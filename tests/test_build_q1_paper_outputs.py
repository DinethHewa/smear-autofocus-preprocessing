from __future__ import annotations

import csv
import json
from pathlib import Path

from autofocus.cli.build_q1_paper_outputs import build_q1_paper_outputs


def test_build_q1_paper_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "artifacts" / "20260102_014123_c6f8c289"
    outdir = tmp_path / "paper_outputs"

    manifest = build_q1_paper_outputs(run_dir, outdir)

    assert manifest["fixed_single_op"] == "tophat_bothat"
    assert manifest["dataset_oracle_ops"]["TBF"] == "msr"
    assert (outdir / "README.md").exists()
    assert (outdir / "figures" / "Figure_03_GPConvergence.svg").exists()
    assert (outdir / "tables" / "Table_01_MainComparison.tex").exists()

    with (outdir / "tables" / "Table_01_MainComparison.csv").open(newline="", encoding="utf-8") as handle:
        main_rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in main_rows}
    assert set(by_method) == {"composite", "fixed_single", "oracle_single"}
    assert float(by_method["composite"]["generalization_score"]) < float(by_method["oracle_single"]["generalization_score"])
    assert float(by_method["oracle_single"]["generalization_score"]) < float(by_method["fixed_single"]["generalization_score"])

    with (outdir / "tables" / "Table_02_DatasetwiseComparison.csv").open(newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    assert len(dataset_rows) == 5

    with (outdir / "main" / "paper_summary_manifest.json").open(encoding="utf-8") as handle:
        saved_manifest = json.load(handle)
    assert saved_manifest["main_findings"]["best_generation"] == 24

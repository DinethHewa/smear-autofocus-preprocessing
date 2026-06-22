from pathlib import Path

import pandas as pd

from autofocus.utils.run_artifacts import build_run_summary_line


def test_run_summary_format(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"runtime_sec": [0.1, 0.2], "failed": [0, 1]})
    df.to_csv(run_dir / "eval_results_eval.csv", index=False)

    line = build_run_summary_line(run_dir, "run123")
    assert line.startswith("RUN_SUMMARY: run_id=run123")
    assert "generalization_score=" in line
    assert "stability_ratio=" in line
    assert "ms_per_image=" in line
    assert "failures=" in line
    assert "ms_per_image=150.000" in line

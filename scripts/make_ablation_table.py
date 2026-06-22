from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("artifacts")
    rows = []
    for summary_path in root.glob("*/optuna_per_op/per_op_summary.csv"):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        parent_run_id = summary_path.parent.parent.name
        df = df.copy()
        df["parent_run_id"] = parent_run_id
        rows.append(df)

    if not rows:
        print("No per_op_summary.csv files found under artifacts/")
        return

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_per_op_table.csv"

    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values(["op_name", "best_value"], ascending=[True, True])
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

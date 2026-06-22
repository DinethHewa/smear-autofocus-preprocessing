from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _compare_csv(original: Path, optimized: Path, rtol: float, atol: float, ignore_columns: set[str]) -> tuple[str, str]:
    a = pd.read_csv(original)
    b = pd.read_csv(optimized)
    drop_cols = [col for col in ignore_columns if col in a.columns or col in b.columns]
    if drop_cols:
        a = a.drop(columns=[col for col in drop_cols if col in a.columns])
        b = b.drop(columns=[col for col in drop_cols if col in b.columns])
    if list(a.columns) != list(b.columns):
        return "Failed", f"column mismatch: {list(a.columns)} != {list(b.columns)}"
    if len(a) != len(b):
        return "Failed", f"row count mismatch: {len(a)} != {len(b)}"
    diffs: List[str] = []
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            av = a[col].to_numpy()
            bv = b[col].to_numpy()
            if not np.allclose(av, bv, rtol=rtol, atol=atol, equal_nan=True):
                diffs.append(col)
        else:
            if not a[col].fillna("<NA>").astype(str).equals(b[col].fillna("<NA>").astype(str)):
                diffs.append(col)
    return ("Passed", "") if not diffs else ("Failed", "different columns: " + ", ".join(diffs[:20]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate original and optimized output directories.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--ignore-columns", default="", help="Comma-separated CSV columns to ignore during comparison.")
    args = parser.parse_args()
    original = Path(args.original)
    optimized = Path(args.optimized)
    ignore_columns = {item.strip() for item in args.ignore_columns.split(",") if item.strip()}
    rows = []
    original_files = {path.relative_to(original): path for path in original.rglob("*") if path.is_file()}
    optimized_files = {path.relative_to(optimized): path for path in optimized.rglob("*") if path.is_file()}
    for rel in sorted(set(original_files) | set(optimized_files)):
        if rel not in original_files:
            rows.append((str(rel), "Failed", "missing in original", "No", ""))
            continue
        if rel not in optimized_files:
            rows.append((str(rel), "Failed", "missing in optimized", "No", ""))
            continue
        op = original_files[rel]
        npth = optimized_files[rel]
        if op.suffix.lower() == ".csv":
            status, diff = _compare_csv(op, npth, args.rtol, args.atol, ignore_columns)
            rows.append((str(rel), status, diff, "Yes" if status == "Passed" else "No", "csv comparison"))
        else:
            same_size = op.stat().st_size == npth.stat().st_size
            rows.append((str(rel), "Passed" if same_size else "Check", "" if same_size else "file size differs", "Yes" if same_size else "Review", "non-csv size check"))
    lines = ["# Validation Report", "", "| Output | Match Status | Difference | Acceptable? | Notes |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    Path("performance_audit/validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [row for row in rows if row[1] == "Failed"]
    print(f"Compared {len(rows)} files; failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

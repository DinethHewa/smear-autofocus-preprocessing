from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


def _csv_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob(pattern) if path.is_file() and path.suffix.lower() == ".csv")


def convert_csv_to_parquet(csv_path: Path, output_dir: Path, *, overwrite: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}.parquet"
    if out_path.exists() and not overwrite:
        return {"csv": str(csv_path), "parquet": str(out_path), "status": "exists"}
    df = pd.read_csv(csv_path)
    df.to_parquet(out_path, index=False)
    check = pd.read_parquet(out_path)
    if len(df) != len(check) or list(df.columns) != list(check.columns):
        raise ValueError(f"Parquet validation failed for {csv_path}")
    return {
        "csv": str(csv_path),
        "parquet": str(out_path),
        "status": "written",
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional Parquet caches for repeated internal CSV reads.")
    parser.add_argument("--input", required=True, help="CSV file or directory containing CSV files.")
    parser.add_argument("--output-dir", default="optimized_safe/cache")
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    files = _csv_files(input_path, args.glob)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")
    results = [convert_csv_to_parquet(path, Path(args.output_dir), overwrite=args.overwrite) for path in files]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(Path(args.output_dir)),
        "files": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--out', default='artifacts/summary_tables.csv')
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    csvs = list(artifacts_dir.rglob('per_stack_results*.csv'))
    if not csvs:
        print('No evaluation CSVs found.')
        return

    frames = [pd.read_csv(path) for path in csvs]
    merged = pd.concat(frames, ignore_index=True)
    summary = merged.groupby(['dataset']).agg({'total_fitness': 'mean', 'is_failure': 'mean'}).reset_index()
    summary.to_csv(args.out, index=False)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

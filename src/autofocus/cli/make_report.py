from __future__ import annotations

import argparse
from pathlib import Path

from ..explain.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()

    report_path = build_report(Path(args.run_dir))
    print(f'Report written: {report_path}')


if __name__ == '__main__':
    main()

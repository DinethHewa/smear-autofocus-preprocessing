from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "scripts", "tests", "checl")
PATTERNS = [
    (r"\.iterrows\(", "Pandas row iteration", "Prefer vectorized operations or to_dict('records') for read-only iteration.", "Low"),
    (r"\.itertuples\(", "Pandas tuple iteration", "Usually faster than iterrows, but still check if vectorization is possible.", "Medium"),
    (r"\.apply\([^\\n]*axis\s*=\s*1", "Row-wise df.apply(axis=1)", "Prefer vectorized column expressions or Polars expressions.", "Medium"),
    (r"for\s+\w+\s+in\s+range\(len\(", "range(len(...)) loop", "Prefer vectorized NumPy/Pandas or direct iteration.", "Medium"),
    (r"\.loc\[[^\n]+\]\s*=", "Potential scalar df.loc assignment", "Prefer vectorized assignment when possible.", "Medium"),
    (r"pd\.concat\(", "pd.concat call", "If inside a loop, collect frames in a list and concatenate once.", "Medium"),
    (r"read_csv\(", "CSV read", "Cache repeated reads or use Parquet for repeated internal processing.", "Low"),
    (r"read_excel\(", "Excel read", "Cache repeated reads; Excel parsing is slow.", "Low"),
    (r"cv2\.imread\(", "OpenCV image read", "Avoid repeated image reads; batch/cache when deterministic and memory-safe.", "Medium"),
    (r"Image\.open\(", "PIL image read", "Avoid repeated image reads; batch/cache when deterministic and memory-safe.", "Medium"),
    (r"\.copy\(", "DataFrame/array copy", "Check whether copy is necessary on large data.", "Low"),
    (r"glob\(", "Glob scan", "Avoid uncontrolled repeated glob scans inside loops.", "Low"),
]


def _function_for_line(tree: ast.AST, line_no: int) -> str:
    best = ""
    best_line = -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", -1)
            end = getattr(node, "end_lineno", start)
            if start <= line_no <= end and start > best_line:
                best = node.name
                best_line = start
    return best


def _iter_files() -> Iterable[Path]:
    for scan_dir in SCAN_DIRS:
        base = ROOT / scan_dir
        if base.exists():
            yield from sorted(base.rglob("*.py"))


def scan() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in _iter_files():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, label, suggestion, risk in PATTERNS:
                if re.search(pattern, line):
                    rows.append(
                        {
                            "file": rel,
                            "line": str(line_no),
                            "function_or_class": _function_for_line(tree, line_no),
                            "pattern": label,
                            "why_slow": _why(label),
                            "suggested_safe_replacement": suggestion,
                            "risk_level": risk,
                            "can_auto_revise": "Yes" if risk == "Low" and label in {"Pandas row iteration", "CSV read"} else "No",
                        }
                    )
    return rows


def _why(label: str) -> str:
    return {
        "Pandas row iteration": "Python-level loops over DataFrame rows are slow on large manifests.",
        "Pandas tuple iteration": "Still Python-level iteration; acceptable for small control loops only.",
        "Row-wise df.apply(axis=1)": "Calls Python once per row and blocks query optimization.",
        "range(len(...)) loop": "Often indicates scalar Python work over data that may be vectorizable.",
        "Potential scalar df.loc assignment": "Repeated scalar assignment causes many index lookups/copies.",
        "pd.concat call": "Repeated concat inside loops is quadratic in data size.",
        "CSV read": "CSV parsing is CPU-heavy; repeated reads can dominate small stages.",
        "Excel read": "Excel parsing is slow and should be cached.",
        "OpenCV image read": "Image decoding in loops can dominate image pipelines.",
        "PIL image read": "Image decoding in loops can dominate image pipelines.",
        "DataFrame/array copy": "Large unnecessary copies increase memory and runtime.",
        "Glob scan": "Repeated filesystem scans are expensive on large trees.",
    }.get(label, "Potential performance hotspot.")


def write_outputs(rows: List[Dict[str, str]]) -> None:
    out_dir = ROOT / "performance_audit"
    csv_path = out_dir / "bottleneck_summary.csv"
    md_path = out_dir / "bottleneck_summary.md"
    fieldnames = [
        "file",
        "line",
        "function_or_class",
        "pattern",
        "why_slow",
        "suggested_safe_replacement",
        "risk_level",
        "can_auto_revise",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Bottleneck Summary", "", "| File | Line | Function/Class | Pattern | Why Slow | Safe Replacement | Risk | Auto-Revise |", "|---|---:|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| {file} | {line} | {function_or_class} | {pattern} | {why_slow} | {suggested_safe_replacement} | {risk_level} | {can_auto_revise} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Detected {len(rows)} potential bottlenecks")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    write_outputs(scan())


if __name__ == "__main__":
    main()

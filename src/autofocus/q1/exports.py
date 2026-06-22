from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(row) for row in rows]
    if fieldnames is None:
        keys: List[str] = []
        for row in rows_list:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows_list:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV input: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.4f}"
    return str(value)


def write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    rows_list = [dict(row) for row in rows]
    if columns is None:
        columns = list(rows_list[0].keys()) if rows_list else ["status"]
    align = "l" + "r" * max(0, len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\hline"]
    lines.append(" & ".join(_latex_escape(col) for col in columns) + r" \\")
    lines.append(r"\hline")
    if rows_list:
        for row in rows_list:
            lines.append(" & ".join(_latex_escape(_format_cell(row.get(col, ""))) for col in columns) + r" \\")
    else:
        lines.append("No data" + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_text(path, "\n".join(lines) + "\n")


def write_simple_bar_figure(path_base: Path, rows: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str, title: str, ylabel: str) -> List[str]:
    written: List[str] = []
    try:
        import matplotlib.pyplot as plt

        labels = [str(row.get(x_key, "")) for row in rows]
        values = [float(row.get(y_key, 0.0)) for row in rows]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(labels, values, color="#2563eb")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in (".svg", ".png", ".pdf"):
            out = path_base.with_suffix(ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300)
            written.append(str(out))
        plt.close(fig)
    except Exception:
        out = path_base.with_suffix(".svg")
        labels = ", ".join(str(row.get(x_key, "")) for row in rows)
        write_text(
            out,
            f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300">'
            f'<text x="20" y="40" font-size="22">{title}</text>'
            f'<text x="20" y="80" font-size="14">{labels}</text></svg>\n',
        )
        written.append(str(out))
    return written


def write_heatmap_figure(path_base: Path, matrix_rows: Sequence[Mapping[str, Any]], *, title: str) -> List[str]:
    written: List[str] = []
    if not matrix_rows:
        return write_simple_bar_figure(path_base, [], x_key="method", y_key="score", title=title, ylabel="score")
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        row_names = [str(row.get("dataset", row.get("evaluation_dataset", idx))) for idx, row in enumerate(matrix_rows)]
        col_names = [key for key in matrix_rows[0].keys() if key not in {"dataset", "evaluation_dataset"}]
        arr = np.asarray([[float(row.get(col, float("nan"))) for col in col_names] for row in matrix_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(max(8.0, len(col_names) * 0.8), max(4.8, len(row_names) * 0.6)))
        im = ax.imshow(arr, aspect="auto", cmap="viridis_r")
        ax.set_title(title)
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=35, ha="right")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        fig.colorbar(im, ax=ax, label="score (lower is better)")
        fig.tight_layout()
        for ext in (".svg", ".png", ".pdf"):
            out = path_base.with_suffix(ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300)
            written.append(str(out))
        plt.close(fig)
    except Exception:
        out = path_base.with_suffix(".svg")
        write_text(out, f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><text x="20" y="40">{title}</text></svg>\n')
        written.append(str(out))
    return written


def write_focus_curve_figure(path_base: Path, rows: Sequence[Mapping[str, Any]], *, title: str, max_curves: int = 8) -> List[str]:
    written: List[str] = []
    selected = [dict(row) for row in rows if row.get("normalized_curve")][:max_curves]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for row in selected:
            curve = json.loads(str(row["normalized_curve"]))
            values = [float(value) if value is not None else float("nan") for value in curve]
            label = f"{row.get('dataset', '')}:{row.get('method', '')}"
            ax.plot(range(len(values)), values, linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("Slice index")
        ax.set_ylabel("Normalized focus response")
        ax.set_ylim(-0.05, 1.05)
        if selected:
            ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for ext in (".svg", ".png", ".pdf"):
            out = path_base.with_suffix(ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300)
            written.append(str(out))
        plt.close(fig)
    except Exception:
        out = path_base.with_suffix(".svg")
        write_text(out, f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><text x="20" y="40">{title}</text></svg>\n')
        written.append(str(out))
    return written

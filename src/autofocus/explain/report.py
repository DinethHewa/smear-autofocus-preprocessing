from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gp_progress(run_dir: Path, report_dir: Path) -> Tuple[str, str] | None:
    path = run_dir / "gp_progress.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "generation" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df["generation"], df["best_score"], label="best")
    ax.plot(df["generation"], df["mean_score"], label="mean")
    ax.plot(df["generation"], df["std_score"], label="std")
    ax.set_xlabel("generation")
    ax.set_ylabel("score")
    ax.set_title("GP Progress")
    ax.legend()
    out = report_dir / "gp_progress.png"
    _save_plot(fig, out)
    return ("GP progress", out.name)


def _plot_dataset_summary(run_dir: Path, report_dir: Path) -> Tuple[str, str] | None:
    path = run_dir / "dataset_summary_eval.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "dataset" not in df.columns or "mean_fitness" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["dataset"], df["mean_fitness"], yerr=df.get("std_fitness", 0.0))
    ax.set_xlabel("dataset")
    ax.set_ylabel("mean fitness")
    ax.set_title("Dataset Summary")
    out = report_dir / "dataset_summary.png"
    _save_plot(fig, out)
    return ("Dataset summary", out.name)


def _plot_per_stack_hist(run_dir: Path, report_dir: Path) -> Tuple[str, str] | None:
    path = run_dir / "per_stack_results_eval.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "total_fitness" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["total_fitness"].dropna(), bins=30, alpha=0.8)
    ax.set_xlabel("total_fitness")
    ax.set_ylabel("count")
    ax.set_title("Per-stack fitness distribution")
    out = report_dir / "per_stack_fitness_hist.png"
    _save_plot(fig, out)
    return ("Per-stack fitness histogram", out.name)


def _plot_per_op_summary(run_dir: Path, report_dir: Path) -> Tuple[str, str] | None:
    path = run_dir / "optuna_per_op" / "per_op_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "op_name" not in df.columns or "best_value" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["op_name"], df["best_value"])
    ax.set_xlabel("best_value")
    ax.set_title("Per-operator tuning best values")
    out = report_dir / "per_op_summary.png"
    _save_plot(fig, out)
    return ("Per-operator tuning summary", out.name)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_report(run_dir: Path) -> Path:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    sections: List[Tuple[str, str]] = []
    for plotter in (_plot_gp_progress, _plot_dataset_summary, _plot_per_stack_hist, _plot_per_op_summary):
        item = plotter(run_dir, report_dir)
        if item:
            sections.append(item)

    best_pipeline = _load_json(run_dir / "best_pipeline.json")
    eval_summary = _load_json(run_dir / "eval_summary.json")
    tuned_params = _load_json(run_dir / "tuned_params_used.json")

    lines = [
        "# Explainability Report",
        "",
        "## Summary",
        f"- Run directory: `{run_dir}`",
        f"- Generalization score: `{eval_summary.get('generalization_score')}`",
        f"- Stability ratio: `{eval_summary.get('stability_ratio')}`",
        "",
        "## Plots",
    ]
    if sections:
        for title, filename in sections:
            lines.append(f"### {title}")
            lines.append(f"![{title}]({filename})")
            lines.append("")
    else:
        lines.append("No plots available.")
        lines.append("")

    lines.extend([
        "## Best Pipeline",
        "```json",
        json.dumps(best_pipeline, indent=2),
        "```",
    ])

    if tuned_params:
        lines.extend([
            "",
            "## Tuned Parameters Used",
            "```json",
            json.dumps(tuned_params, indent=2),
            "```",
        ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

from __future__ import annotations

from typing import Dict

import pandas as pd

from .evaluate import evaluate_pipeline
from ..preprocess.pipeline import Pipeline


def run_ablations(
    manifest_df: pd.DataFrame,
    base_pipeline: Pipeline,
    focus_cfg: Dict[str, float],
    metrics_cfg: Dict[str, float],
    artifacts_dir,
    evaluation_cfg: Dict[str, float] | None = None,
) -> pd.DataFrame:
    results = []

    summary, _ = evaluate_pipeline(
        manifest_df,
        base_pipeline,
        focus_cfg,
        metrics_cfg,
        outer_split=['trainval'],
        inner_split=['train', 'val'],
        artifacts_dir=artifacts_dir,
        run_tag='ablation_base',
        failure_penalty=1e6,
        evaluation_cfg=evaluation_cfg,
    )
    results.append({'ablation': 'base', 'generalization_score': summary['generalization_score'], 'status': 'ok'})

    no_stability = dict(metrics_cfg)
    no_stability['stability_threshold'] = 0.0
    summary, _ = evaluate_pipeline(
        manifest_df,
        base_pipeline,
        focus_cfg,
        no_stability,
        outer_split=['trainval'],
        inner_split=['train', 'val'],
        artifacts_dir=artifacts_dir,
        run_tag='ablation_no_stability',
        failure_penalty=1e6,
        evaluation_cfg=evaluation_cfg,
    )
    results.append({'ablation': 'no_stability', 'generalization_score': summary['generalization_score'], 'status': 'ok'})

    # Hooks for required ablations (populate by wiring the relevant runners).
    results.append({'ablation': 'tuned_vs_default', 'generalization_score': float('nan'), 'status': 'todo'})
    results.append({'ablation': 'with_vs_without_stain_norm', 'generalization_score': float('nan'), 'status': 'todo'})
    results.append({'ablation': 'gp_vs_random_search', 'generalization_score': float('nan'), 'status': 'todo'})
    results.append({'ablation': 'multi_dataset_objective', 'generalization_score': float('nan'), 'status': 'todo'})

    return pd.DataFrame(results)

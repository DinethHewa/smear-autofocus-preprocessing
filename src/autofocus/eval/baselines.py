from __future__ import annotations

from typing import List, Tuple
import random

import pandas as pd

from ..gp.genome import DEFAULT_CATEGORIES, DEFAULT_OPS_BY_CATEGORY, random_genome, genome_to_pipeline, filter_ops_by_availability
from .evaluate import evaluate_pipeline

from ..preprocess.pipeline import Pipeline, PipelineStep


def baseline_pipelines() -> List[Pipeline]:
    return [
        Pipeline([PipelineStep(name='identity', enabled=True, params={}, category='baseline')]),
        Pipeline([PipelineStep(name='gaussian_blur', enabled=True, params={'ksize': 3, 'sigma': 0.0}, category='denoise')]),
        Pipeline([PipelineStep(name='clahe_enhancement', enabled=True, params={'clip_limit': 2.0, 'tile_grid_size': 8}, category='contrast')]),
        Pipeline([PipelineStep(name='unsharp_mask', enabled=True, params={'ksize': 3, 'amount': 1.0}, category='edge')]),
        Pipeline([PipelineStep(name='reinhard_normalization', enabled=True, params={}, category='color')]),
        Pipeline([
            PipelineStep(name='gaussian_blur', enabled=True, params={'ksize': 3, 'sigma': 0.0}, category='denoise'),
            PipelineStep(name='clahe_enhancement', enabled=True, params={'clip_limit': 2.0, 'tile_grid_size': 8}, category='contrast'),
            PipelineStep(name='unsharp_mask', enabled=True, params={'ksize': 3, 'amount': 1.0}, category='edge'),
            PipelineStep(name='fft_highpass_filter', enabled=True, params={'cutoff': 0.1}, category='frequency'),
            PipelineStep(name='morph_open_close', enabled=True, params={'ksize': 3}, category='morph'),
            PipelineStep(name='reinhard_normalization', enabled=True, params={}, category='color'),
            PipelineStep(name='msr', enabled=True, params={'sigmas': (15.0, 80.0, 250.0)}, category='retinex'),
            PipelineStep(name='bicubic_upsample', enabled=True, params={'scale': 1.5}, category='upsample'),
            PipelineStep(name='guided_filter', enabled=True, params={'radius': 5, 'eps': 1e-3}, category='structure'),
        ]),
    ]


def random_search_baseline(
    manifest_df: pd.DataFrame,
    focus_cfg: dict,
    metrics_cfg: dict,
    artifacts_dir,
    budget: int = 20,
    seed: int = 0,
    evaluation_cfg: dict | None = None,
) -> Tuple[Pipeline, float]:
    rng = random.Random(seed)
    best_score = float('inf')
    best_pipeline = None

    ops_by_category = filter_ops_by_availability(DEFAULT_OPS_BY_CATEGORY)
    for i in range(budget):
        genome = random_genome(rng, DEFAULT_CATEGORIES, ops_by_category)
        pipeline = genome_to_pipeline(genome, DEFAULT_CATEGORIES)
        summary, _ = evaluate_pipeline(
            manifest_df,
            pipeline,
            focus_cfg,
            metrics_cfg,
            outer_split=['trainval'],
            inner_split=['train', 'val'],
            artifacts_dir=artifacts_dir,
            run_tag=f'random_search_{i}',
            failure_penalty=1e6,
            evaluation_cfg=evaluation_cfg,
        )
        if summary['generalization_score'] < best_score:
            best_score = summary['generalization_score']
            best_pipeline = pipeline

    if best_pipeline is None:
        best_pipeline = Pipeline([PipelineStep(name='identity', enabled=True, params={}, category='baseline')])
    return best_pipeline, float(best_score)

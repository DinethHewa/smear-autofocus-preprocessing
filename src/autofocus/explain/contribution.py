from __future__ import annotations

from typing import List

import pandas as pd

from ..preprocess.pipeline import Pipeline, PipelineStep


def category_drop_analysis(pipeline: Pipeline, categories: List[str], evaluate_fn, base_score: float) -> pd.DataFrame:
    rows = []
    for category in categories:
        new_steps = []
        for step in pipeline.steps:
            if step.category == category:
                new_steps.append(PipelineStep(name='identity', enabled=True, params={}, category=category))
            else:
                new_steps.append(step)
        drop_pipeline = Pipeline(new_steps)
        score = evaluate_fn(drop_pipeline)
        rows.append({'category': category, 'score': score, 'delta': score - base_score})
    return pd.DataFrame(rows)

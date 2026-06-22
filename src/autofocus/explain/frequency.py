from __future__ import annotations

from typing import List

import pandas as pd

from ..preprocess.pipeline import Pipeline


def operator_frequency(pipelines: List[Pipeline]) -> pd.DataFrame:
    rows = []
    for pipe in pipelines:
        for step in pipe.steps:
            rows.append({'operator': step.name, 'category': step.category})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    counts = df.groupby(['category', 'operator']).size().reset_index(name='count')
    return counts

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.io import load_stack
from ..metrics.focus_measures import compute_focus_curve
from ..preprocess.pipeline import Pipeline


def generate_exemplars(manifest_df: pd.DataFrame, pipeline: Pipeline, focus_name: str, out_dir: Path, max_items: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = manifest_df.head(max_items)
    for _, row in subset.iterrows():
        stack_index = row.get('stack_index', None)
        stack = load_stack(row['path'], stack_index=stack_index)
        processed = np.stack([pipeline.apply(stack[i]) for i in range(stack.shape[0])], axis=0)
        curve_before = compute_focus_curve(stack, name=focus_name)
        curve_after = compute_focus_curve(processed, name=focus_name)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes[0, 0].imshow(stack[0], cmap='gray')
        axes[0, 0].set_title('raw')
        axes[0, 1].imshow(processed[0], cmap='gray')
        axes[0, 1].set_title('processed')
        axes[1, 0].plot(curve_before)
        axes[1, 0].set_title('curve raw')
        axes[1, 1].plot(curve_after)
        axes[1, 1].set_title('curve processed')
        for ax in axes.ravel():
            if 'curve' not in ax.get_title():
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f"exemplar_{row['stack_id']}.png", dpi=150)
        plt.close(fig)

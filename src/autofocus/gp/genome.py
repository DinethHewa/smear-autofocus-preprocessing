from __future__ import annotations

from typing import Dict, List
import random

from ..preprocess.pipeline import Pipeline, PipelineStep
from ..preprocess.ops import is_op_available


DEFAULT_CATEGORIES = [
    'denoise',
    'contrast',
    'edge',
    'frequency',
    'morph',
    'color',
    'retinex',
    'upsample',
    'structure',
]

DEFAULT_OPS_BY_CATEGORY: Dict[str, List[str]] = {
    'denoise': ['identity', 'gaussian_blur', 'median_filter', 'bilateral_filter', 'nlm_denoising'],
    'contrast': ['identity', 'clahe_enhancement', 'histogram_equalization', 'gamma_correction'],
    'edge': ['identity', 'unsharp_mask', 'laplacian_sharpen', 'dog_filter'],
    'frequency': ['identity', 'fft_highpass_filter', 'dwt_filter', 'gabor_filter'],
    'morph': ['identity', 'gradient_morph', 'morph_open_close', 'tophat_bothat'],
    'color': ['identity', 'reinhard_normalization', 'macenko_normalization', 'vahadane_normalization'],
    'retinex': ['identity', 'ssr', 'msr', 'msrcr'],
    'upsample': ['identity', 'bicubic_upsample', 'sr_model'],
    'structure': ['identity', 'guided_filter', 'anisotropic_diffusion', 'steerable_filter'],
}


def filter_ops_by_availability(ops_by_category: Dict[str, List[str]]) -> Dict[str, List[str]]:
    filtered: Dict[str, List[str]] = {}
    for category, ops in ops_by_category.items():
        available = [op for op in ops if is_op_available(op)]
        if not available:
            raise ValueError(f"No available operators for category '{category}'")
        filtered[category] = available
    return filtered


def random_genome(rng: random.Random, categories: List[str], ops_by_category: Dict[str, List[str]]) -> List[str]:
    return [rng.choice(ops_by_category[cat]) for cat in categories]


def genome_to_pipeline(genome: List[str], categories: List[str]) -> Pipeline:
    steps = []
    for op_name, category in zip(genome, categories):
        steps.append(PipelineStep(name=op_name, enabled=True, params={}, category=category))
    return Pipeline(steps=steps)

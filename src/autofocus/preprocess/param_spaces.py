from __future__ import annotations

from typing import Any, Dict, Callable, List


DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    'identity': {},
    'gaussian_blur': {'ksize': 3, 'sigma': 0.0},
    'median_filter': {'ksize': 3},
    'bilateral_filter': {'d': 5, 'sigma_color': 50.0, 'sigma_space': 50.0},
    'bilateral_edge_preserve': {'d': 7, 'sigma_color': 75.0, 'sigma_space': 75.0},
    'nlm_denoising': {'h': 10.0, 'template_window_size': 7, 'search_window_size': 21},
    'clahe_enhancement': {'clip_limit': 2.0, 'tile_grid_size': 8},
    'histogram_equalization': {},
    'gamma_correction': {'gamma': 1.0},
    'unsharp_mask': {'sigma': 1.0, 'amount': 1.0, 'ksize': 3},
    'laplacian_sharpen': {'ksize': 3, 'amount': 1.0},
    'dog_filter': {'ksize1': 3, 'ksize2': 7, 'sigma1': 0.5, 'sigma2': 1.5},
    'gabor_filter': {'ksize': 9, 'sigma': 2.0, 'theta': 0.0, 'lam': 10.0, 'gamma': 0.5},
    'fft_highpass_filter': {'cutoff': 0.1},
    'dwt_filter': {'wavelet': 'haar'},
    'anisotropic_diffusion': {'n_iter': 5, 'kappa': 30.0, 'gamma': 0.1},
    'guided_filter': {'radius': 5, 'eps': 1e-3},
    'gradient_morph': {'ksize': 3},
    'morph_open_close': {'ksize': 3},
    'tophat_bothat': {'ksize': 5},
    'bicubic_upsample': {'scale': 1.5},
    'sr_model': {'scale': 1.5},
    'steerable_filter': {},
    'ssr': {'sigma': 15.0},
    'msr': {'sigmas': (15.0, 80.0, 250.0)},
    'msrcr': {'sigmas': (15.0, 80.0, 250.0), 'gain': 1.0, 'offset': 0.0},
    'reinhard_normalization': {},
    'apply_reinhard': {},
    'macenko_normalization': {},
    'vahadane_normalization': {},
}


def gaussian_blur_space(trial) -> Dict[str, Any]:
    return {
        'ksize': trial.suggest_int('ksize', 1, 9, step=2),
        'sigma': trial.suggest_float('sigma', 0.0, 2.0),
    }


def median_filter_space(trial) -> Dict[str, Any]:
    return {'ksize': trial.suggest_int('ksize', 1, 9, step=2)}


def bilateral_filter_space(trial) -> Dict[str, Any]:
    return {
        'd': trial.suggest_int('d', 3, 9, step=2),
        'sigma_color': trial.suggest_float('sigma_color', 10.0, 100.0),
        'sigma_space': trial.suggest_float('sigma_space', 10.0, 100.0),
    }


def nlm_denoising_space(trial) -> Dict[str, Any]:
    return {
        'h': trial.suggest_float('h', 5.0, 20.0),
        'template_window_size': trial.suggest_int('template_window_size', 3, 7, step=2),
        'search_window_size': trial.suggest_int('search_window_size', 7, 21, step=2),
    }


def clahe_space(trial) -> Dict[str, Any]:
    return {
        'clip_limit': trial.suggest_float('clahe.clip_limit', 1.0, 5.0),
        'tile_grid_size': trial.suggest_int('clahe.tile_grid_size', 4, 16),
    }


def gamma_space(trial) -> Dict[str, Any]:
    return {'gamma': trial.suggest_float('gamma', 0.5, 2.0)}


def unsharp_space(trial) -> Dict[str, Any]:
    return {
        'sigma': trial.suggest_float('unsharp.sigma', 0.5, 3.0),
        'amount': trial.suggest_float('unsharp.amount', 0.0, 2.0),
    }


def laplacian_sharpen_space(trial) -> Dict[str, Any]:
    return {
        'ksize': trial.suggest_int('laplacian.ksize', 1, 9, step=2),
        'amount': trial.suggest_float('laplacian.amount', 0.0, 2.0),
    }


def dog_space(trial) -> Dict[str, Any]:
    return {
        'ksize1': trial.suggest_int('ksize1', 1, 7, step=2),
        'ksize2': trial.suggest_int('ksize2', 3, 11, step=2),
        'sigma1': trial.suggest_float('sigma1', 0.3, 2.0),
        'sigma2': trial.suggest_float('sigma2', 0.5, 3.0),
    }


def gabor_space(trial) -> Dict[str, Any]:
    return {
        'ksize': trial.suggest_int('ksize', 5, 15, step=2),
        'sigma': trial.suggest_float('sigma', 1.0, 5.0),
        'theta': trial.suggest_float('theta', 0.0, 3.1415),
        'lam': trial.suggest_float('lam', 5.0, 20.0),
        'gamma': trial.suggest_float('gamma', 0.2, 0.9),
    }


def fft_highpass_space(trial) -> Dict[str, Any]:
    return {'cutoff': trial.suggest_float('cutoff', 0.05, 0.3)}


def dwt_space(trial) -> Dict[str, Any]:
    return {'wavelet': trial.suggest_categorical('wavelet', ['haar', 'db1', 'db2'])}


def anisotropic_space(trial) -> Dict[str, Any]:
    return {
        'n_iter': trial.suggest_int('n_iter', 3, 10),
        'kappa': trial.suggest_float('kappa', 10.0, 50.0),
        'gamma': trial.suggest_float('gamma', 0.05, 0.2),
    }


def guided_space(trial) -> Dict[str, Any]:
    return {
        'radius': trial.suggest_int('radius', 3, 9, step=2),
        'eps': trial.suggest_float('eps', 1e-4, 1e-2),
    }


def morph_space(trial) -> Dict[str, Any]:
    return {'ksize': trial.suggest_int('ksize', 3, 9, step=2)}


def bicubic_space(trial) -> Dict[str, Any]:
    return {'scale': trial.suggest_float('scale', 1.0, 2.0)}


def ssr_space(trial) -> Dict[str, Any]:
    return {'sigma': trial.suggest_float('sigma', 5.0, 50.0)}


def msr_space(trial) -> Dict[str, Any]:
    return {'sigmas': (15.0, 80.0, 250.0)}


def msrcr_space(trial) -> Dict[str, Any]:
    return {'sigmas': (15.0, 80.0, 250.0), 'gain': trial.suggest_float('gain', 0.5, 2.0), 'offset': trial.suggest_float('offset', -1.0, 1.0)}


PARAM_SPACES: Dict[str, Callable] = {
    'gaussian_blur': gaussian_blur_space,
    'median_filter': median_filter_space,
    'bilateral_filter': bilateral_filter_space,
    'bilateral_edge_preserve': bilateral_filter_space,
    'nlm_denoising': nlm_denoising_space,
    'clahe_enhancement': clahe_space,
    'gamma_correction': gamma_space,
    'unsharp_mask': unsharp_space,
    'laplacian_sharpen': laplacian_sharpen_space,
    'dog_filter': dog_space,
    'gabor_filter': gabor_space,
    'fft_highpass_filter': fft_highpass_space,
    'dwt_filter': dwt_space,
    'anisotropic_diffusion': anisotropic_space,
    'guided_filter': guided_space,
    'gradient_morph': morph_space,
    'morph_open_close': morph_space,
    'tophat_bothat': morph_space,
    'bicubic_upsample': bicubic_space,
    'sr_model': bicubic_space,
    'ssr': ssr_space,
    'msr': msr_space,
    'msrcr': msrcr_space,
}


class _KeyTrial:
    def suggest_int(self, name, low, high=None, **kwargs):
        return int(low)

    def suggest_float(self, name, low, high=None, **kwargs):
        return float(low)

    def suggest_categorical(self, name, choices, **kwargs):
        return choices[0] if choices else None


def param_space_keys(step_name: str) -> List[str]:
    space = PARAM_SPACES.get(step_name)
    if space is None:
        return []
    try:
        params = space(_KeyTrial())
    except Exception:
        params = DEFAULT_PARAMS.get(step_name, {})
    if not isinstance(params, dict):
        return []
    return list(params.keys())

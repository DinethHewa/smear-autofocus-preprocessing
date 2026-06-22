from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import ndimage as ndi

try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None

EPS = 1e-8


def _to_float_gray(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    return np.asarray(gray, dtype=np.float32, order='C')


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = _to_float_gray(img)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < EPS:
        return np.zeros_like(arr, dtype=np.uint8)
    if vmax <= 1.5 and vmin >= 0.0:
        arr = arr * 255.0
    else:
        arr = (arr - vmin) / (vmax - vmin + EPS) * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(img.dtype)
    return img[..., 0]


def variance_of_laplacian(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    lap = ndi.laplace(gray)
    return float(lap.var())


def gradient_squared_energy(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    return float(np.sum(gx**2 + gy**2))


def brenner_gradient(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    if gray.shape[0] < 3:
        return 0.0
    diff = gray[:-2, :] - gray[2:, :]
    return float(np.sum(diff**2))


def edge_width_sharpness_index(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    blur = ndi.gaussian_filter(gray, sigma=1.0)
    gx = ndi.sobel(blur, axis=1)
    gy = ndi.sobel(blur, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    return float(np.mean(grad))


def image_entropy(img: np.ndarray) -> float:
    u8 = _to_uint8(img)
    hist, _ = np.histogram(u8.flatten(), bins=256, range=(0, 256))
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    prob = hist.astype(np.float64) / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def intensity_skewness(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    flat = gray.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat)) + EPS
    return float(np.mean(((flat - mean) / std) ** 3))


def fourier_transform_energy(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    f = np.fft.fftshift(np.fft.fft2(gray))
    return float(np.sum(np.abs(f) ** 2))


def fourier_transform_sharpness_index(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    f = np.fft.fftshift(np.fft.fft2(gray))
    return float(np.mean(np.abs(f)))


def wavelet_detail_energy_db1(img: np.ndarray) -> float:
    if pywt is None:  # pragma: no cover
        raise ImportError("pywt is required for Wavelet Detail Energy (db1)")

    coeffs = pywt.wavedec2(_to_float_gray(img), "db1", level=1)
    return float(sum(np.sum(np.abs(c)) for c in coeffs[1:]))


def threshold_abs_grad(img: np.ndarray, thr: float = 50.0) -> float:
    gray = _to_float_gray(img)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    max_val = float(np.max(gray))
    thr_eff = float(thr)
    if max_val <= 1.5:
        thr_eff = thr / 255.0
    return float(np.sum(grad > thr_eff))


def roberts_focus_measure(img: np.ndarray) -> float:
    gray = _to_float_gray(img)
    gx = gray[1:, 1:] - gray[:-1, :-1]
    gy = gray[1:, :-1] - gray[:-1, 1:]
    return float(np.sum(gx**2 + gy**2))


def if_gt(a: float, b: float, x: float, y: float) -> float:
    if np.isnan(a) or np.isnan(b) or np.isnan(x) or np.isnan(y):
        return float('nan')
    return float(x if a > b else y)


def psqrt(value: float) -> float:
    return float(np.sqrt(np.abs(float(value)) + EPS))


def composite_if_gt_focus_measure(img: np.ndarray) -> float:
    """
    Composite measure:
    if_gt(
        Threshold Absolute Gradient,
        Image Entropy / (Gradient Squared Energy + Fourier Transform Energy),
        log1p(Roberts Focus Measure) * Intensity Skewness Index,
        Edge Width Sharpness Index
    )
    """
    a = threshold_abs_grad(img)
    b = image_entropy(img) / (gradient_squared_energy(img) + fourier_transform_energy(img) + EPS)
    x = np.log1p(roberts_focus_measure(img)) * intensity_skewness(img)
    y = edge_width_sharpness_index(img)
    return if_gt(float(a), float(b), float(x), float(y))


def cfm4_focus_measure(img: np.ndarray) -> float:
    """
    CFM4 = FTSI - WDE-db1 * psqrt(psqrt(BG - GSE)).

    Definitions:
    BG = Brenner Gradient
    GSE = Gradient Squared Energy
    FTSI = Fourier Transform Sharpness Index
    WDE-db1 = Wavelet Detail Energy using db1
    psqrt(x) = sqrt(abs(x) + eps)
    """
    bg = brenner_gradient(img)
    gse = gradient_squared_energy(img)
    ftsi = fourier_transform_sharpness_index(img)
    wde_db1 = wavelet_detail_energy_db1(img)
    return float(ftsi - wde_db1 * psqrt(psqrt(bg - gse)))

def focus_measure(img: np.ndarray, name: str = 'variance_of_laplacian', **kwargs) -> float:
    func = FOCUS_MEASURES.get(name)
    if func is None:
        raise ValueError(f'Unknown focus measure: {name}')
    return float(func(img, **kwargs))


def _compute_focus_curve_cpu(stack: np.ndarray, name: str = 'variance_of_laplacian', **kwargs) -> np.ndarray:
    scores = [focus_measure(stack[i], name=name, **kwargs) for i in range(stack.shape[0])]
    return np.asarray(scores, dtype=np.float32)


def _curve_peak_index(curve: np.ndarray) -> int:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return -1
    max_val = np.nanmax(y)
    idx = np.where(np.isclose(y, max_val))[0]
    return int(idx[-1]) if idx.size else int(np.nanargmax(y))


def _curve_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return 0.0
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_focus_curve(
    stack: np.ndarray,
    name: str = 'variance_of_laplacian',
    *,
    backend: str = "cpu",
    cuda_validate: bool = False,
    cuda_validation_min_corr: float = 0.95,
    cuda_validation_max_peak_delta: int = 1,
    cuda_fallback: bool = True,
    **kwargs,
) -> np.ndarray:
    backend_name = str(backend or "cpu").lower()
    cfm4_names = {"cfm4", "cfm4_bg_gse_ftsi_wde_db1"}
    if name in cfm4_names and backend_name in {"auto", "cuda", "torch_cuda", "torch_cuda_auto"}:
        try:
            from .torch_cfm4 import cfm4_focus_curve_torch, torch_cuda_available

            if torch_cuda_available():
                cuda_curve = cfm4_focus_curve_torch(stack, device="cuda")
                if cuda_validate:
                    cpu_curve = _compute_focus_curve_cpu(stack, name=name, **kwargs)
                    corr = _curve_corr(cpu_curve, cuda_curve)
                    peak_delta = abs(_curve_peak_index(cpu_curve) - _curve_peak_index(cuda_curve))
                    if corr < float(cuda_validation_min_corr) or peak_delta > int(cuda_validation_max_peak_delta):
                        if cuda_fallback:
                            return cpu_curve
                        raise RuntimeError(
                            "CUDA CFM4 validation failed: "
                            f"corr={corr:.6g}, peak_delta={peak_delta}, "
                            f"min_corr={cuda_validation_min_corr}, max_peak_delta={cuda_validation_max_peak_delta}"
                        )
                return cuda_curve
            if backend_name in {"cuda", "torch_cuda"} and not cuda_fallback:
                raise RuntimeError("Torch CUDA requested but unavailable")
        except Exception:
            if not cuda_fallback:
                raise
    return _compute_focus_curve_cpu(stack, name=name, **kwargs)


FOCUS_MEASURES: Dict[str, Callable[..., float]] = {
    'variance_of_laplacian': variance_of_laplacian,
    'gradient_squared_energy': gradient_squared_energy,
    'brenner_gradient': brenner_gradient,
    'edge_width_sharpness_index': edge_width_sharpness_index,
    'image_entropy': image_entropy,
    'intensity_skewness': intensity_skewness,
    'fourier_transform_energy': fourier_transform_energy,
    'fourier_transform_sharpness_index': fourier_transform_sharpness_index,
    'wavelet_detail_energy_db1': wavelet_detail_energy_db1,
    'threshold_abs_grad': threshold_abs_grad,
    'roberts_focus_measure': roberts_focus_measure,
    'if_gt_composite': composite_if_gt_focus_measure,
    'cfm4': cfm4_focus_measure,
    'cfm4_bg_gse_ftsi_wde_db1': cfm4_focus_measure,
}

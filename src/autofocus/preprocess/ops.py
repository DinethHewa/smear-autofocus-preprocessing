from __future__ import annotations

from typing import Dict, Callable, Tuple
import inspect

import numpy as np
import pywt
from scipy import ndimage as ndi
from skimage import color, exposure, filters, morphology, restoration, transform

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

UNAVAILABLE_OPS: list[str] = [
    'macenko_normalization',
    'vahadane_normalization',
]
if not (HAS_CV2 and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter')):
    UNAVAILABLE_OPS.append('guided_filter')


def backend_info() -> dict:
    return {
        'opencv_available': HAS_CV2,
        'unavailable_ops': UNAVAILABLE_OPS,
    }


def is_op_available(op_name: str) -> bool:
    return op_name in OP_REGISTRY and op_name not in UNAVAILABLE_OPS


def _require_available(op_name: str) -> None:
    if op_name in UNAVAILABLE_OPS:
        raise OpInvariantError('UnavailableOp', f'{op_name} is unavailable in this environment')


def _to_float01(img: np.ndarray) -> np.ndarray:
    # Range policy: preprocessing ops return float32 in [0,1].
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
    img = img.astype(np.float32)
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.clip(0.0, 1.0)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = _to_float01(img)
    return (img * 255.0).round().astype(np.uint8)


class OpInvariantError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def assert_same_shape(output: np.ndarray, ref_shape: tuple, op_name: str) -> None:
    if output.shape != ref_shape:
        raise OpInvariantError('ShapeMismatchError', f'{op_name}: expected {ref_shape}, got {output.shape}')


def assert_finite(output: np.ndarray, op_name: str) -> None:
    if not np.isfinite(output).all():
        raise OpInvariantError('InvalidValueError', f'{op_name}: non-finite output detected')


def assert_range(output: np.ndarray, op_name: str) -> None:
    if output.dtype == np.uint8:
        if output.min() < 0 or output.max() > 255:
            raise OpInvariantError('RangeError', f'{op_name}: uint8 out of range')
    else:
        if output.min() < -1e-3 or output.max() > 1.0 + 1e-3:
            raise OpInvariantError('RangeError', f'{op_name}: float out of range [0,1]')


def apply_op(op_name: str, img: np.ndarray, params: Dict[str, object] | None = None, enforce_shape: bool = True) -> np.ndarray:
    if op_name not in OP_REGISTRY:
        raise OpInvariantError('UnknownError', f'Unknown operator: {op_name}')
    _require_available(op_name)
    func = OP_REGISTRY[op_name]
    out = func(img, **(params or {}))
    if enforce_shape:
        assert_same_shape(out, img.shape, op_name)
    assert_finite(out, op_name)
    assert_range(out, op_name)
    return out


def _ensure_kernel_size(ksize: int) -> int:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return ksize


def _apply_per_channel(img: np.ndarray, func) -> np.ndarray:
    if img.ndim == 2:
        return func(img)
    channels = [func(img[..., c]) for c in range(img.shape[-1])]
    return np.stack(channels, axis=-1)


def _supports_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _call_with_channel(func, img: np.ndarray, **kwargs) -> np.ndarray:
    if img.ndim == 3:
        if _supports_param(func, 'channel_axis'):
            return func(img, channel_axis=-1, **kwargs)
        if _supports_param(func, 'multichannel'):
            return func(img, multichannel=True, **kwargs)
        return _apply_per_channel(img, lambda x: func(x, **kwargs))
    if _supports_param(func, 'channel_axis'):
        return func(img, channel_axis=None, **kwargs)
    if _supports_param(func, 'multichannel'):
        return func(img, multichannel=False, **kwargs)
    return func(img, **kwargs)


def _sigma_from_ksize(ksize: int) -> float:
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def identity(img: np.ndarray) -> np.ndarray:
    return _to_float01(img)


def gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    ksize = _ensure_kernel_size(ksize)
    img_f = _to_float01(img)
    if HAS_CV2:
        out = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
        return _to_float01(out)
    sigma_val = float(sigma) if sigma > 0 else _sigma_from_ksize(ksize)
    out = _call_with_channel(filters.gaussian, img_f, sigma=sigma_val, preserve_range=True)
    return _to_float01(out)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = _ensure_kernel_size(ksize)
    if HAS_CV2:
        out = cv2.medianBlur(_to_uint8(img), ksize)
        return _to_float01(out)
    radius = max(1, ksize // 2)
    selem = morphology.disk(radius)
    out = _apply_per_channel(_to_float01(img), lambda x: filters.median(x, selem))
    return _to_float01(out)


def apply_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return median_filter(img, ksize=ksize)


def bilateral_filter(img: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    if HAS_CV2:
        out = cv2.bilateralFilter(_to_float01(img), int(d), sigma_color, sigma_space)
        return _to_float01(out)
    img_f = _to_float01(img)
    sigma_c = float(sigma_color)
    if sigma_c > 1.0:
        sigma_c = sigma_c / 255.0
    out = _call_with_channel(
        restoration.denoise_bilateral,
        img_f,
        sigma_color=sigma_c,
        sigma_spatial=float(sigma_space),
    )
    return _to_float01(out)


def bilateral_edge_preserve(img: np.ndarray, d: int = 7, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    return bilateral_filter(img, d=d, sigma_color=sigma_color, sigma_space=sigma_space)


def nlm_denoising(img: np.ndarray, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
    if HAS_CV2:
        img_u8 = _to_uint8(img)
        if img_u8.ndim == 2:
            out = cv2.fastNlMeansDenoising(img_u8, None, h, template_window_size, search_window_size)
        else:
            out = cv2.fastNlMeansDenoisingColored(img_u8, None, h, h, template_window_size, search_window_size)
        return _to_float01(out)

    img_f = _to_float01(img)
    h_val = float(h)
    if h_val > 1.0:
        h_val = h_val / 255.0
    out = _call_with_channel(
        restoration.denoise_nl_means,
        img_f,
        h=h_val,
        patch_size=int(template_window_size),
        patch_distance=int(search_window_size),
        fast_mode=True,
    )
    return _to_float01(out)


def clahe_enhancement(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    if HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        img_u8 = _to_uint8(img)
        if img_u8.ndim == 2:
            out = clahe.apply(img_u8)
        else:
            channels = [clahe.apply(img_u8[..., c]) for c in range(img_u8.shape[-1])]
            out = np.stack(channels, axis=-1)
        return _to_float01(out)

    img_f = _to_float01(img)
    clip = max(1e-4, float(clip_limit) * 0.01)
    out = _call_with_channel(
        exposure.equalize_adapthist,
        img_f,
        kernel_size=(int(tile_grid_size), int(tile_grid_size)),
        clip_limit=clip,
    )
    return _to_float01(out)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        img_u8 = _to_uint8(img)
        if img_u8.ndim == 2:
            out = cv2.equalizeHist(img_u8)
        else:
            channels = [cv2.equalizeHist(img_u8[..., c]) for c in range(img_u8.shape[-1])]
            out = np.stack(channels, axis=-1)
        return _to_float01(out)

    img_f = _to_float01(img)
    out = _call_with_channel(exposure.equalize_hist, img_f)
    return _to_float01(out)


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    img_f = _to_float01(img)
    gamma = max(1e-6, float(gamma))
    out = exposure.adjust_gamma(img_f, gamma)
    return _to_float01(out)


def unsharp_mask(img: np.ndarray, ksize: int = 3, amount: float = 1.0, sigma: float | None = None) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        if sigma is not None:
            ksize_eff = max(1, int(round(sigma * 2)) | 1)
            blurred = gaussian_blur(img_f, ksize=ksize_eff, sigma=float(sigma))
        else:
            blurred = gaussian_blur(img_f, ksize=ksize)
        out = img_f + amount * (img_f - blurred)
        return _to_float01(out)
    radius = max(1.0, float(sigma)) if sigma is not None else max(1.0, ksize / 2.0)
    out = _call_with_channel(filters.unsharp_mask, img_f, radius=radius, amount=amount)
    return _to_float01(out)


def laplacian_sharpen(img: np.ndarray, ksize: int = 3, amount: float = 1.0) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        lap = cv2.Laplacian(img_f, cv2.CV_32F, ksize=_ensure_kernel_size(ksize))
    else:
        lap = _apply_per_channel(img_f, lambda x: ndi.laplace(x))
    out = img_f + amount * lap
    return _to_float01(out)


def dog_filter(img: np.ndarray, ksize1: int = 3, ksize2: int = 7, sigma1: float = 0.5, sigma2: float = 1.5) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        g1 = gaussian_blur(img_f, ksize=ksize1, sigma=sigma1)
        g2 = gaussian_blur(img_f, ksize=ksize2, sigma=sigma2)
        out = g1 - g2
        return _to_float01(out)
    out = _apply_per_channel(img_f, lambda x: filters.difference_of_gaussians(x, sigma1, sigma2))
    return _to_float01(out)


def gabor_filter(img: np.ndarray, ksize: int = 9, sigma: float = 2.0, theta: float = 0.0, lam: float = 10.0, gamma: float = 0.5) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        ksize = _ensure_kernel_size(ksize)
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
        if img_f.ndim == 2:
            out = cv2.filter2D(img_f, cv2.CV_32F, kernel)
        else:
            channels = [cv2.filter2D(img_f[..., c], cv2.CV_32F, kernel) for c in range(img_f.shape[-1])]
            out = np.stack(channels, axis=-1)
        return _to_float01(out)

    def _gabor(x: np.ndarray) -> np.ndarray:
        real, imag = filters.gabor(x, frequency=1.0 / max(lam, 1e-3), theta=theta, bandwidth=1.0, sigma_x=sigma, sigma_y=sigma)
        return np.sqrt(real ** 2 + imag ** 2)

    out = _apply_per_channel(img_f, _gabor)
    return _to_float01(out)


def fft_highpass_filter(img: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    img_f = _to_float01(img)
    if img_f.ndim == 3:
        channels = [fft_highpass_filter(img_f[..., c], cutoff=cutoff) for c in range(img_f.shape[-1])]
        return _to_float01(np.stack(channels, axis=-1))
    h, w = img_f.shape
    f = np.fft.fft2(img_f)
    fshift = np.fft.fftshift(f)
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * cutoff)
    mask = np.ones((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    mask[(y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] = 0.0
    fshift *= mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    out = np.real(img_back)
    return _to_float01(out)


def dwt_filter(img: np.ndarray, wavelet: str = 'haar') -> np.ndarray:
    img_f = _to_float01(img)
    if img_f.ndim == 3:
        channels = [dwt_filter(img_f[..., c], wavelet=wavelet) for c in range(img_f.shape[-1])]
        return _to_float01(np.stack(channels, axis=-1))
    coeffs2 = pywt.dwt2(img_f, wavelet)
    cA, (cH, cV, cD) = coeffs2
    zero = np.zeros_like(cA)
    recon = pywt.idwt2((zero, (cH, cV, cD)), wavelet)
    recon = recon[: img_f.shape[0], : img_f.shape[1]]
    return _to_float01(recon)


def anisotropic_diffusion(img: np.ndarray, n_iter: int = 5, kappa: float = 30.0, gamma: float = 0.1) -> np.ndarray:
    img_f = _to_float01(img)
    if img_f.ndim == 3:
        channels = [anisotropic_diffusion(img_f[..., c], n_iter=n_iter, kappa=kappa, gamma=gamma) for c in range(img_f.shape[-1])]
        return _to_float01(np.stack(channels, axis=-1))

    out = img_f.copy()
    for _ in range(int(n_iter)):
        north = np.roll(out, -1, axis=0) - out
        south = np.roll(out, 1, axis=0) - out
        east = np.roll(out, -1, axis=1) - out
        west = np.roll(out, 1, axis=1) - out
        cN = np.exp(-(north / kappa) ** 2)
        cS = np.exp(-(south / kappa) ** 2)
        cE = np.exp(-(east / kappa) ** 2)
        cW = np.exp(-(west / kappa) ** 2)
        out = out + gamma * (cN * north + cS * south + cE * east + cW * west)
    return _to_float01(out)


def guided_filter(img: np.ndarray, radius: int = 5, eps: float = 1e-3) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2 and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        out = cv2.ximgproc.guidedFilter(img_f, img_f, radius, eps)
        return _to_float01(out)
    raise NotImplementedError('guided_filter requires opencv-contrib (cv2.ximgproc)')


def gradient_morph(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    img_f = _to_float01(img)
    radius = max(1, _ensure_kernel_size(ksize) // 2)
    selem = morphology.disk(radius)
    def _grad(x: np.ndarray) -> np.ndarray:
        return morphology.dilation(x, selem) - morphology.erosion(x, selem)
    out = _apply_per_channel(img_f, _grad)
    return _to_float01(out)


def morph_open_close(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    img_f = _to_float01(img)
    radius = max(1, _ensure_kernel_size(ksize) // 2)
    selem = morphology.disk(radius)
    def _op(x: np.ndarray) -> np.ndarray:
        return morphology.closing(morphology.opening(x, selem), selem)
    out = _apply_per_channel(img_f, _op)
    return _to_float01(out)


def tophat_bothat(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    img_f = _to_float01(img)
    radius = max(1, _ensure_kernel_size(ksize) // 2)
    selem = morphology.disk(radius)
    def _op(x: np.ndarray) -> np.ndarray:
        return morphology.white_tophat(x, selem) + morphology.black_tophat(x, selem)
    out = _apply_per_channel(img_f, _op)
    return _to_float01(out)


def bicubic_upsample(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    img_f = _to_float01(img)
    h, w = img_f.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    if HAS_CV2:
        up = cv2.resize(img_f, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        back = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)
        return _to_float01(back)

    if img_f.ndim == 2:
        up = transform.resize(img_f, (new_h, new_w), order=3, preserve_range=True, anti_aliasing=False)
        back = transform.resize(up, (h, w), order=3, preserve_range=True, anti_aliasing=False)
    else:
        up = transform.resize(img_f, (new_h, new_w, img_f.shape[-1]), order=3, preserve_range=True, anti_aliasing=False)
        back = transform.resize(up, (h, w, img_f.shape[-1]), order=3, preserve_range=True, anti_aliasing=False)
    return _to_float01(back)


def sr_model(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    return bicubic_upsample(img, scale=scale)


def steerable_filter(img: np.ndarray) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        if img_f.ndim == 3:
            gray = img_f.mean(axis=-1)
        else:
            gray = img_f
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag0 = np.abs(gx)
        mag90 = np.abs(gy)
        mag45 = np.abs(gx + gy)
        mag135 = np.abs(gx - gy)
        out = np.maximum.reduce([mag0, mag90, mag45, mag135])
    else:
        if img_f.ndim == 3:
            gray = img_f.mean(axis=-1)
        else:
            gray = img_f
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        mag0 = np.abs(gx)
        mag90 = np.abs(gy)
        mag45 = np.abs(gx + gy)
        mag135 = np.abs(gx - gy)
        out = np.maximum.reduce([mag0, mag90, mag45, mag135])

    if img_f.ndim == 3:
        out = np.stack([out] * img_f.shape[-1], axis=-1)
    return _to_float01(out)


def _retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    img_f = _to_float01(img)
    if HAS_CV2:
        blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    else:
        blur = _call_with_channel(filters.gaussian, img_f, sigma=sigma, preserve_range=True)
    return np.log(img_f + 1e-6) - np.log(blur + 1e-6)


def ssr(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    out = _retinex(img, sigma)
    return _to_float01(out)


def msr(img: np.ndarray, sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0)) -> np.ndarray:
    ret = sum(_retinex(img, sigma) for sigma in sigmas) / float(len(sigmas))
    return _to_float01(ret)


def msrcr(img: np.ndarray, sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0), gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    img_f = _to_float01(img)
    ret = msr(img_f, sigmas=sigmas)
    if img_f.ndim == 3:
        intensity = img_f.sum(axis=-1, keepdims=True)
        color_term = np.log(1.0 + gain * img_f) - np.log(1.0 + intensity)
        out = ret + offset + color_term
    else:
        out = ret
    return _to_float01(out)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 3:
        return img
    return np.stack([img[..., 0]] * 3, axis=-1)


def rgb_to_od(img: np.ndarray) -> np.ndarray:
    img_f = _to_float01(ensure_rgb(img))
    return -np.log(img_f + 1e-6)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.exp(-od)


def standardize_luminosity(img: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    img_f = _to_float01(img)
    p = np.percentile(img_f, percentile)
    return _to_float01(img_f / (p + 1e-6))


def compute_lab_mean_std(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_rgb = ensure_rgb(img)
    img_f = _to_float01(img_rgb)
    if HAS_CV2:
        img_u8 = _to_uint8(img_rgb)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    else:
        lab = color.rgb2lab(img_f).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(axis=0)
    std = lab.reshape(-1, 3).std(axis=0)
    return mean, std


def reinhard_normalization(img: np.ndarray, target_mean: np.ndarray | None = None, target_std: np.ndarray | None = None) -> np.ndarray:
    is_gray = img.ndim == 2
    img_rgb = ensure_rgb(img)
    img_f = _to_float01(img_rgb)
    mean, std = compute_lab_mean_std(img)
    target_mean = mean if target_mean is None else np.asarray(target_mean)
    target_std = std if target_std is None else np.asarray(target_std)

    if HAS_CV2:
        img_u8 = _to_uint8(img_rgb)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        norm = (lab - mean) / (std + 1e-6)
        norm = norm * target_std + target_mean
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(norm, cv2.COLOR_LAB2RGB)
        if is_gray:
            rgb = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(rgb.dtype)
        return _to_float01(rgb)

    lab = color.rgb2lab(img_f)
    norm = (lab - mean) / (std + 1e-6)
    norm = norm * target_std + target_mean
    rgb = color.lab2rgb(norm)
    if is_gray:
        rgb = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(rgb.dtype)
    return _to_float01(rgb)


def apply_reinhard(img: np.ndarray, target_mean: np.ndarray | None = None, target_std: np.ndarray | None = None) -> np.ndarray:
    return reinhard_normalization(img, target_mean=target_mean, target_std=target_std)


def get_stain_matrix_macenko(img: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Macenko stain matrix estimation is not implemented in this package.')


def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Stain concentration estimation is not implemented in this package.')


def macenko_normalization(img: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Macenko normalization is not implemented in this package.')


def vahadane_normalization(img: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Vahadane normalization is not implemented in this package.')


OP_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    'identity': identity,
    'gaussian_blur': gaussian_blur,
    'median_filter': median_filter,
    'apply_median_filter': apply_median_filter,
    'bilateral_filter': bilateral_filter,
    'bilateral_edge_preserve': bilateral_edge_preserve,
    'nlm_denoising': nlm_denoising,
    'clahe_enhancement': clahe_enhancement,
    'histogram_equalization': histogram_equalization,
    'gamma_correction': gamma_correction,
    'unsharp_mask': unsharp_mask,
    'laplacian_sharpen': laplacian_sharpen,
    'dog_filter': dog_filter,
    'gabor_filter': gabor_filter,
    'fft_highpass_filter': fft_highpass_filter,
    'dwt_filter': dwt_filter,
    'anisotropic_diffusion': anisotropic_diffusion,
    'guided_filter': guided_filter,
    'gradient_morph': gradient_morph,
    'morph_open_close': morph_open_close,
    'tophat_bothat': tophat_bothat,
    'bicubic_upsample': bicubic_upsample,
    'sr_model': sr_model,
    'steerable_filter': steerable_filter,
    'ssr': ssr,
    'msr': msr,
    'msrcr': msrcr,
    'reinhard_normalization': reinhard_normalization,
    'apply_reinhard': apply_reinhard,
    'macenko_normalization': macenko_normalization,
    'vahadane_normalization': vahadane_normalization,
}

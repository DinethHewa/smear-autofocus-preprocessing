from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-8


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _to_gray_tensor(stack: np.ndarray, *, device: str) -> Any:
    import torch

    arr = np.asarray(stack)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
        tensor = 0.299 * tensor[..., 0] + 0.587 * tensor[..., 1] + 0.114 * tensor[..., 2]
    elif arr.ndim == 4:
        tensor = torch.as_tensor(arr[..., 0], dtype=torch.float32, device=device)
    elif arr.ndim == 3:
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
    elif arr.ndim == 2:
        tensor = torch.as_tensor(arr[None, ...], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Expected stack shape (Z,H,W), (Z,H,W,C), or (H,W); got {arr.shape}")
    return tensor.contiguous()


def _sobel_energy(gray: Any) -> Any:
    import torch
    import torch.nn.functional as F

    # SciPy ndimage.sobel is not a CUDA kernel. These Sobel kernels preserve the
    # same focus-measure intent and are validated/fallback-controlled by caller.
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    x = gray[:, None, :, :]
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    return (gx.square() + gy.square()).sum(dim=(1, 2, 3))


def _haar_db1_detail_energy(gray: Any) -> Any:
    # Orthonormal 2D Haar/db1 one-level detail coefficients. PyWavelets handles
    # odd edges with extension modes; for CUDA screening we crop to even shape.
    h_even = (gray.shape[-2] // 2) * 2
    w_even = (gray.shape[-1] // 2) * 2
    if h_even < 2 or w_even < 2:
        return gray.new_zeros((gray.shape[0],))
    x = gray[:, :h_even, :w_even]
    a = x[:, 0::2, 0::2]
    b = x[:, 0::2, 1::2]
    c = x[:, 1::2, 0::2]
    d = x[:, 1::2, 1::2]
    c_h = (a + b - c - d) * 0.5
    c_v = (a - b + c - d) * 0.5
    c_d = (a - b - c + d) * 0.5
    return c_h.abs().sum(dim=(1, 2)) + c_v.abs().sum(dim=(1, 2)) + c_d.abs().sum(dim=(1, 2))


def cfm4_focus_curve_torch(stack: np.ndarray, *, device: str = "cuda") -> np.ndarray:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")
    with torch.no_grad():
        gray = _to_gray_tensor(stack, device=device)
        if gray.shape[-2] < 3:
            bg = gray.new_zeros((gray.shape[0],))
        else:
            bg = (gray[:, :-2, :] - gray[:, 2:, :]).square().sum(dim=(1, 2))
        gse = _sobel_energy(gray)
        ftsi = torch.fft.fft2(gray).abs().mean(dim=(1, 2))
        wde_db1 = _haar_db1_detail_energy(gray)
        curve = ftsi - wde_db1 * torch.sqrt(torch.sqrt((bg - gse).abs() + EPS) + EPS)
        return curve.detach().cpu().numpy().astype(np.float32, copy=False)

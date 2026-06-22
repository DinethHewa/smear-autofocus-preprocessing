from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import imageio.v2 as imageio
    _imread = imageio.imread
except Exception:
    from skimage import io as skio
    _imread = skio.imread
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(img.dtype)
    return img[..., 0]


@lru_cache(maxsize=128)
def _load_images_from_dir_cached(path: str) -> np.ndarray:
    path = Path(path)
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise IOError(f'No images found in directory: {path}')
    images = [_imread(p) for p in files]
    stack = np.stack([_to_gray(img) for img in images], axis=0)
    return stack


def _load_images_from_dir(path: Path) -> np.ndarray:
    return _load_images_from_dir_cached(str(path.expanduser().resolve()))


@lru_cache(maxsize=16)
def _open_npy_cached(path: str) -> np.ndarray:
    return np.load(path, mmap_mode='r')


def open_npy(path: str | Path) -> np.ndarray:
    return _open_npy_cached(str(Path(path).expanduser().resolve()))


def load_stack(path: str | Path, stack_index: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.npy':
        arr = open_npy(path)
    elif path.suffix.lower() == '.npz':
        data = np.load(path, mmap_mode='r')
        key = data.files[0]
        arr = data[key]
    elif path.is_dir():
        arr = _load_images_from_dir(path)
    else:
        raise IOError(f'Unsupported stack path: {path}')

    # IO policy: return raw stack arrays; normalization is handled in preprocessing ops.
    if arr.ndim == 4:
        if stack_index is None:
            raise ValueError(f'stack_index required for multi-stack array: {path}')
        arr = arr[int(stack_index)]
    elif arr.ndim == 3:
        if stack_index is None or int(stack_index) == 0:
            arr = arr
        else:
            arr = arr[int(stack_index)]
            arr = arr[None, ...]
    elif arr.ndim == 2:
        arr = arr[None, ...]
    else:
        raise ValueError(f'Unsupported stack shape: {arr.shape}')

    if arr.ndim != 3:
        raise ValueError(f'Expected stack to be 3D (slices, H, W). Got {arr.shape}')
    return np.asarray(arr)


def _load_measure_names(scores_npz) -> list[str]:
    for key in ("measure_names", "measure_names_all", "names"):
        if key in scores_npz:
            names = scores_npz[key].tolist()
            out = []
            for n in names:
                if isinstance(n, bytes):
                    out.append(n.decode("utf-8"))
                else:
                    out.append(str(n))
            return out
    return []


def load_scores_npz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(Path(path), allow_pickle=True)
    score_key = None
    for key in ("scores_norm", "scores", "scores_all", "scores_norm_all"):
        if key in data:
            score_key = key
            break
    if score_key is None:
        raise KeyError(f"No scores key found in {path}. Keys: {data.files}")
    scores = np.asarray(data[score_key])
    measure_names = _load_measure_names(data)

    if scores.ndim == 2:
        scores = scores[:, :, None]
    if scores.ndim != 3:
        raise ValueError(f"Scores array must be 3D (N,Z,M) or (N,M,Z). Got {scores.shape}")

    n_measures = len(measure_names) if measure_names else scores.shape[-1]
    if scores.shape[2] == n_measures:
        scores = scores
    elif scores.shape[1] == n_measures:
        scores = scores.transpose(0, 2, 1)
    elif scores.shape[0] == n_measures:
        scores = scores.transpose(1, 2, 0)
    else:
        raise ValueError(f"Cannot infer scores axes with measure_names={n_measures}, shape={scores.shape}")

    if not measure_names:
        measure_names = [f"measure_{i}" for i in range(scores.shape[2])]
    return scores.astype(np.float32), measure_names


def select_measure_curve(
    scores: np.ndarray,
    measure_names: list[str],
    target_name: str | None = None,
    target_index: int | None = None,
) -> tuple[np.ndarray, str]:
    if target_index is not None:
        idx = int(target_index)
        if idx < 0 or idx >= scores.shape[2]:
            raise IndexError(f"target_index out of range: {idx} for scores shape {scores.shape}")
        name = measure_names[idx] if idx < len(measure_names) else f"measure_{idx}"
        return scores[:, :, idx], name

    if target_name and target_name in measure_names:
        idx = measure_names.index(target_name)
        return scores[:, :, idx], target_name

    name = measure_names[0] if measure_names else "measure_0"
    return scores[:, :, 0], name

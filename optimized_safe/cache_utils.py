from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def load_cached_dataframe(path: str | Path, *, fallback_csv: str | Path | None = None, **read_kwargs: Any) -> pd.DataFrame:
    """Load a dataframe from Parquet when available, otherwise from CSV."""
    cache_path = Path(path)
    if cache_path.exists():
        if cache_path.suffix.lower() == ".parquet":
            return pd.read_parquet(cache_path, **read_kwargs)
        return pd.read_csv(cache_path, **read_kwargs)
    if fallback_csv is None:
        raise FileNotFoundError(cache_path)
    csv_path = Path(fallback_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path, **read_kwargs)


def save_cached_dataframe(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Save a dataframe cache without changing the final publication CSV exports."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".parquet":
        if not _parquet_available():
            raise RuntimeError("pyarrow is required for Parquet cache writing")
        df.to_parquet(cache_path, index=index)
    else:
        df.to_csv(cache_path, index=index)
    return cache_path


def load_or_build_cache(
    cache_path: str | Path,
    builder: Callable[[], pd.DataFrame],
    *,
    source_paths: Iterable[str | Path] = (),
    validate: Callable[[pd.DataFrame], None] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load a dataframe cache, rebuilding when absent, forced, or older than sources."""
    path = Path(cache_path)
    sources = [Path(item) for item in source_paths]
    stale = force or not path.exists()
    if path.exists() and sources:
        cache_mtime = path.stat().st_mtime
        stale = any(source.exists() and source.stat().st_mtime > cache_mtime for source in sources)
    if stale:
        df = builder()
        if validate is not None:
            validate(df)
        save_cached_dataframe(df, path)
        return df
    df = load_cached_dataframe(path)
    if validate is not None:
        validate(df)
    return df


def validate_same_shape_columns(original: pd.DataFrame, cached: pd.DataFrame) -> None:
    if len(original) != len(cached):
        raise ValueError(f"Cache row count mismatch: {len(original)} != {len(cached)}")
    if list(original.columns) != list(cached.columns):
        raise ValueError("Cache columns differ from source dataframe")

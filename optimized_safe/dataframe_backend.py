from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

BackendName = Literal["pandas", "polars", "auto"]


def polars_available() -> bool:
    try:
        import polars  # noqa: F401

        return True
    except Exception:
        return False


def choose_backend(backend: BackendName = "auto", *, rows_hint: int | None = None) -> str:
    if backend == "pandas":
        return "pandas"
    if backend == "polars":
        if not polars_available():
            raise RuntimeError("backend='polars' requested but polars is not installed")
        return "polars"
    if polars_available() and rows_hint is not None and rows_hint >= 100_000:
        return "polars"
    return "pandas"


def read_csv(path: str | Path, *, backend: BackendName = "auto", rows_hint: int | None = None):
    selected = choose_backend(backend, rows_hint=rows_hint)
    if selected == "polars":
        import polars as pl

        return pl.scan_csv(path).collect()
    return pd.read_csv(path)


def to_pandas(df):
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise TypeError(f"Unsupported dataframe object: {type(df)!r}")


def filter_equal(df, column: str, value, *, backend: BackendName = "auto"):
    selected = "pandas" if isinstance(df, pd.DataFrame) else "polars"
    if selected == "polars":
        import polars as pl

        return df.filter(pl.col(column) == value)
    return df[df[column] == value]

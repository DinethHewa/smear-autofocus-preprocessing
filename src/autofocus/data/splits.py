from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

import pandas as pd


def validate_no_leakage(df: pd.DataFrame) -> None:
    split_counts = df.groupby('group_id')['split'].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        leaked_groups = leaked.index.tolist()
        raise ValueError(f'Group leakage detected in groups: {leaked_groups}')


def allowed_splits(df: pd.DataFrame, allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    invalid = sorted(set(df['split']) - allowed_set)
    if invalid:
        raise ValueError(f'Invalid split labels in manifest: {invalid}')


def _size_to_count(size: float, total: int) -> int:
    if size < 0:
        raise ValueError('split size must be non-negative')
    if size < 1:
        return int(round(size * total))
    return int(size)


def assign_nested_splits(
    df: pd.DataFrame,
    group_col: str,
    outer_test_size: float,
    inner_val_size: float,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()
    group_ids = sorted(df[group_col].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_test = _size_to_count(outer_test_size, n_groups)
    if n_test <= 0 and n_groups >= 3:
        n_test = 1
    n_test = min(max(n_test, 0), n_groups - 1) if n_groups > 1 else 0

    test_groups = set(group_ids[:n_test])
    outer_split = df[group_col].apply(lambda g: 'test' if g in test_groups else 'trainval')
    df['outer_split'] = outer_split

    trainval_groups = [g for g in group_ids if g not in test_groups]
    n_trainval = len(trainval_groups)
    n_val = _size_to_count(inner_val_size, n_trainval)
    if n_val <= 0 and n_trainval >= 3:
        n_val = 1
    n_val = min(max(n_val, 0), n_trainval - 1) if n_trainval > 1 else 0

    val_groups = set(trainval_groups[:n_val])
    inner_split = []
    for g in df[group_col]:
        if g in test_groups:
            inner_split.append('test')
        elif g in val_groups:
            inner_split.append('val')
        else:
            inner_split.append('train')
    df['inner_split'] = inner_split
    return df


def assert_no_group_overlap(df: pd.DataFrame, group_col: str, split_col: str) -> None:
    split_counts = df.groupby(group_col)[split_col].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        leaked_groups = leaked.index.tolist()
        raise ValueError(f'Group leakage detected for {split_col}: {leaked_groups}')


def assert_test_never_in_inner(df: pd.DataFrame) -> None:
    test_in_inner = df[(df['outer_split'] == 'test') & (df['inner_split'].isin(['train', 'val']))]
    if not test_in_inner.empty:
        raise ValueError('Outer test groups present in inner splits')

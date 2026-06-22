from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .io import open_npy

REQUIRED_COLUMNS = ['dataset', 'stack_id', 'group_id', 'path', 'stack_index']


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Manifest missing required columns: {missing}')
    if df['path'].isnull().any():
        raise ValueError('Manifest contains null paths')
    if df['group_id'].isnull().any():
        raise ValueError('Manifest contains null group_id values')


def load_manifest(path: str | Path, smoke_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')
    df = pd.read_csv(path)
    base_dir = path.parent
    def _resolve_path(p: str) -> str:
        path_obj = Path(p)
        if path_obj.is_absolute():
            return str(path_obj)
        candidate = base_dir / path_obj
        if candidate.exists():
            return str(candidate)
        return str(path_obj)
    df['path'] = df['path'].apply(_resolve_path)
    validate_manifest(df)
    if smoke_rows is not None:
        df = df.head(int(smoke_rows)).copy()
    return df


def normalize_direct_inputs(datasets_cfg) -> List[Dict[str, str]]:
    if isinstance(datasets_cfg, dict):
        return [{'name': str(k), 'path': str(v)} for k, v in datasets_cfg.items()]
    if isinstance(datasets_cfg, list):
        out = []
        for item in datasets_cfg:
            name = item.get('name') or item.get('dataset')
            if not name:
                raise ValueError('Each dataset entry must include a name')
            out.append({'name': str(name), 'path': str(item['path'])})
        return out
    raise ValueError('direct_inputs.datasets must be a list or dict of name/path entries')


def _load_group_map(path: str | Path) -> Tuple[Dict[Tuple[str, int], str], Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'group_map_path not found: {path}')
    df = pd.read_csv(path)
    if 'group_id' not in df.columns:
        raise ValueError('group_map_path must include group_id column')

    by_dataset_index: Dict[Tuple[str, int], str] = {}
    by_stack_id: Dict[str, str] = {}

    if 'stack_id' in df.columns:
        for row in df.to_dict('records'):
            stack_id = str(row['stack_id'])
            group_id = str(row['group_id'])
            by_stack_id[stack_id] = group_id
    if 'dataset' in df.columns and 'stack_index' in df.columns:
        for row in df.to_dict('records'):
            dataset = str(row['dataset'])
            try:
                idx = int(row['stack_index'])
            except Exception:
                continue
            group_id = str(row['group_id'])
            by_dataset_index[(dataset, idx)] = group_id

    if not by_stack_id and not by_dataset_index:
        raise ValueError('group_map_path must include stack_id or dataset+stack_index columns')
    return by_dataset_index, by_stack_id


def _apply_group_map(df: pd.DataFrame, group_map_path: str | Path) -> pd.DataFrame:
    by_dataset_index, by_stack_id = _load_group_map(group_map_path)
    group_ids = []
    for row in df.to_dict('records'):
        stack_id = str(row['stack_id'])
        dataset = str(row['dataset'])
        idx = int(row['stack_index'])
        if stack_id in by_stack_id:
            group_ids.append(by_stack_id[stack_id])
        else:
            group_ids.append(by_dataset_index.get((dataset, idx)))
    df = df.copy()
    df['group_id'] = group_ids
    if df['group_id'].isnull().any():
        missing = df[df['group_id'].isnull()][['dataset', 'stack_id', 'stack_index']].head(5)
        raise ValueError(f'group_map_path missing entries for some stacks. Examples: {missing.to_dict(orient="records")}')
    return df


def build_manifest_from_direct_inputs(datasets_cfg, group_map_path: str | Path | None = None) -> pd.DataFrame:
    datasets = normalize_direct_inputs(datasets_cfg)
    records = []

    for entry in datasets:
        dataset = entry['name']
        path_str = entry['path']
        path_obj = Path(path_str)
        if not path_obj.exists():
            raise FileNotFoundError(f'Dataset path not found: {path_obj}')

        path_norm = str(Path(path_str))
        if path_obj.suffix.lower() == '.npy':
            arr = open_npy(path_obj)
            if arr.ndim == 4:
                num_stacks = arr.shape[0]
                for idx in range(num_stacks):
                    stack_id = f'{dataset}_{idx:06d}'
                    records.append({
                        'dataset': dataset,
                        'stack_id': stack_id,
                        'group_id': stack_id,
                        'path': path_norm,
                        'stack_index': idx,
                    })
            elif arr.ndim == 3:
                stack_id = f'{dataset}_000000'
                records.append({
                    'dataset': dataset,
                    'stack_id': stack_id,
                    'group_id': stack_id,
                    'path': path_norm,
                    'stack_index': 0,
                })
            elif arr.ndim == 2:
                stack_id = f'{dataset}_000000'
                records.append({
                    'dataset': dataset,
                    'stack_id': stack_id,
                    'group_id': stack_id,
                    'path': path_norm,
                    'stack_index': 0,
                })
            else:
                raise ValueError(f'Unsupported array shape in {path_obj}: {arr.shape}')
        elif path_obj.is_dir():
            entries = sorted([p for p in path_obj.iterdir() if p.is_dir()])
            if not entries:
                entries = sorted([p for p in path_obj.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}])
            if not entries:
                raise IOError(f'No stack entries found in directory: {path_obj}')
            for idx, entry in enumerate(entries):
                stack_id = f'{dataset}_{idx:06d}'
                records.append({
                    'dataset': dataset,
                    'stack_id': stack_id,
                    'group_id': stack_id,
                    'path': str(entry),
                    'stack_index': idx,
                })
        else:
            raise IOError(f'Unsupported dataset path: {path_obj}')

    df = pd.DataFrame(records)
    if group_map_path:
        df = _apply_group_map(df, group_map_path)
    validate_manifest(df)
    return df


def save_manifest_snapshot(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

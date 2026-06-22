from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.io import load_stack, open_npy
from ..data.splits import assign_nested_splits, assert_no_group_overlap, assert_test_never_in_inner
from ..metrics.focus_measures import FOCUS_MEASURES, compute_focus_curve
from .curves import predict_peak_index


SOURCE_LABEL_COLUMNS = ("reference_focus_index", "best_focus_index", "reference_peak", "ref_peak", "peak")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_path(raw_path: str, base_dir: Path) -> str:
    path = Path(str(raw_path))
    if path.is_absolute():
        return str(path)
    candidate = base_dir / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _natural_key(text: str) -> List[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def _image_files(folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and ":Zone.Identifier" not in path.name
        ],
        key=lambda path: _natural_key(path.name),
    )


def _discover_stack_dirs(dataset_root: Path, *, min_images_per_stack: int = 2) -> List[Path]:
    children = sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: _natural_key(path.name))
    immediate = [folder for folder in children if len(_image_files(folder)) >= min_images_per_stack]
    if immediate:
        return immediate
    if len(_image_files(dataset_root)) >= min_images_per_stack:
        return [dataset_root]
    recursive = sorted([path for path in dataset_root.rglob("*") if path.is_dir()], key=lambda path: _natural_key(str(path.relative_to(dataset_root))))
    return [folder for folder in recursive if len(_image_files(folder)) >= min_images_per_stack]


def _label_values_for_item(item: Mapping[str, Any], expected_count: int, dataset: str) -> np.ndarray | None:
    label_path_raw = item.get("label_path") or item.get("reference_label_path")
    if not label_path_raw:
        return None
    label_path = Path(str(label_path_raw)).expanduser()
    if not label_path.is_absolute():
        label_path = Path.cwd() / label_path
    if not label_path.exists():
        raise FileNotFoundError(f"[{dataset}] label path not found: {label_path}")
    labels = np.asarray(np.load(label_path, allow_pickle=True)).reshape(-1)
    if labels.shape[0] != expected_count:
        raise ValueError(
            f"[{dataset}] label count mismatch for {label_path}: "
            f"{labels.shape[0]} labels for {expected_count} stacks"
        )
    return labels


def build_manifest(config: Mapping[str, Any], *, smoke: bool = False) -> pd.DataFrame:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    records: List[Dict[str, Any]] = []
    if data_cfg.get("manifest_path"):
        manifest_path = Path(data_cfg["manifest_path"]).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest path not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        base_dir = manifest_path.parent
        if "stack_index" not in df.columns:
            df["stack_index"] = 0
        if "group_id" not in df.columns:
            df["group_id"] = df["stack_id"] if "stack_id" in df.columns else [f"stack_{idx}" for idx, _ in enumerate(df.index)]
        if "stack_id" not in df.columns:
            df["stack_id"] = [f"{dataset}_{idx:06d}" for idx, dataset in enumerate(df["dataset"].astype(str))]
        df["path"] = df["path"].apply(lambda value: _resolve_path(str(value), base_dir))
    else:
        datasets_cfg = data_cfg.get("datasets", [])
        if not datasets_cfg:
            raise ValueError("Q1 config must provide data.manifest_path or data.datasets")
        for item in datasets_cfg:
            dataset = str(item.get("name") or item.get("dataset"))
            path = Path(str(item["path"])).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                raise FileNotFoundError(f"Dataset path not found: {path}")
            arr = open_npy(path) if path.suffix.lower() == ".npy" else None
            if arr is not None and arr.ndim == 4:
                label_values = _label_values_for_item(item, int(arr.shape[0]), dataset)
                for idx in range(arr.shape[0]):
                    record = {"dataset": dataset, "stack_id": f"{dataset}_{idx:06d}", "group_id": f"{dataset}_{idx:06d}", "path": str(path), "stack_index": idx}
                    if label_values is not None:
                        record["best_focus_index"] = int(label_values[idx])
                        record["label_source"] = str(item.get("label_source", "imported_reference"))
                        record["confidence"] = item.get("label_confidence", np.nan)
                    records.append(record)
            elif path.is_dir():
                stack_dirs = _discover_stack_dirs(path)
                if not stack_dirs:
                    raise ValueError(f"[{dataset}] no stack directories found under {path}")
                label_values = _label_values_for_item(item, len(stack_dirs), dataset)
                source_stack_file = str(item.get("source_stack_file", ""))
                for idx, folder in enumerate(stack_dirs):
                    record = {
                        "dataset": dataset,
                        "stack_id": f"{dataset}_{idx:06d}",
                        "group_id": f"{dataset}_{idx:06d}",
                        "path": str(folder),
                        "stack_index": 0,
                        "source_stack_file": source_stack_file,
                        "source_stack_index": idx,
                        "n_slices": len(_image_files(folder)),
                    }
                    if label_values is not None:
                        record["best_focus_index"] = int(label_values[idx])
                        record["label_source"] = str(item.get("label_source", "imported_reference"))
                        record["confidence"] = item.get("label_confidence", np.nan)
                    records.append(record)
            else:
                records.append({"dataset": dataset, "stack_id": f"{dataset}_000000", "group_id": f"{dataset}_000000", "path": str(path), "stack_index": 0})
        df = pd.DataFrame(records)
    required = {"dataset", "stack_id", "group_id", "path", "stack_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if smoke:
        max_rows = int(data_cfg.get("smoke_rows", 0) or config.get("smoke_rows", 0) or 0)
        if max_rows > 0:
            if "dataset" in df.columns and df["dataset"].nunique() > 1:
                per_dataset = max(1, int(np.ceil(max_rows / df["dataset"].nunique())))
                df = (
                    df.groupby("dataset", group_keys=False, sort=True)
                    .head(per_dataset)
                    .head(max_rows)
                    .copy()
                )
            else:
                df = df.head(max_rows).copy()
    split_cfg = config.get("splits", {}) if isinstance(config.get("splits"), dict) else {}
    if "outer_split" not in df.columns or "inner_split" not in df.columns:
        if bool(split_cfg.get("stratify_by_dataset", True)) and "dataset" in df.columns:
            split_parts = []
            for offset, (_, subset) in enumerate(df.groupby("dataset", sort=True)):
                split_parts.append(
                    assign_nested_splits(
                        subset,
                        group_col="group_id",
                        outer_test_size=float(split_cfg.get("outer_test_size", 0.2)),
                        inner_val_size=float(split_cfg.get("inner_val_size", 0.2)),
                        seed=int(config.get("seed", config.get("random_seed", 0))) + offset,
                    )
                )
            df = pd.concat(split_parts, ignore_index=True)
        else:
            df = assign_nested_splits(
                df,
                group_col="group_id",
                outer_test_size=float(split_cfg.get("outer_test_size", 0.2)),
                inner_val_size=float(split_cfg.get("inner_val_size", 0.2)),
                seed=int(config.get("seed", config.get("random_seed", 0))),
            )
    assert_no_group_overlap(df, "group_id", "outer_split")
    assert_no_group_overlap(df[df["outer_split"] == "trainval"], "group_id", "inner_split")
    assert_test_never_in_inner(df)
    if "n_slices" not in df.columns:
        n_slices = []
        for row in df.to_dict("records"):
            path = Path(str(row["path"]))
            if path.is_dir():
                n_slices.append(len(_image_files(path)))
            else:
                stack = load_stack(row["path"], stack_index=int(row["stack_index"]))
                n_slices.append(int(stack.shape[0]))
        df["n_slices"] = n_slices
    return df


def _source_label_for_row(row: Mapping[str, Any]) -> Tuple[int | None, float | None, str]:
    for col in SOURCE_LABEL_COLUMNS:
        if col in row and not pd.isna(row[col]):
            return int(row[col]), float(row.get("confidence", 1.0) if not pd.isna(row.get("confidence", 1.0)) else 1.0), col
    return None, None, ""


def _surrogate_label(stack: np.ndarray, voter_names: Sequence[str]) -> Tuple[int, float, str]:
    votes: List[int] = []
    used: List[str] = []
    for name in voter_names:
        if name not in FOCUS_MEASURES:
            continue
        curve = compute_focus_curve(stack, name=name)
        votes.append(predict_peak_index(curve))
        used.append(name)
    if not votes:
        raise ValueError("No valid surrogate voters available")
    counts: Dict[int, int] = {}
    for vote in votes:
        counts[vote] = counts.get(vote, 0) + 1
    max_count = max(counts.values())
    candidates = [idx for idx, count in counts.items() if count == max_count]
    label = sorted(candidates)[len(candidates) // 2]
    return int(label), float(max_count / len(votes)), "|".join(used)


def build_reference_labels(manifest_df: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_cfg = config.get("labels", {}) if isinstance(config.get("labels"), dict) else {}
    voters = list(label_cfg.get("surrogate_voters") or ["variance_of_laplacian", "gradient_squared_energy", "roberts_focus_measure"])
    label_rows: List[Dict[str, Any]] = []
    loo_rows: List[Dict[str, Any]] = []
    for row in manifest_df.to_dict("records"):
        source_idx, source_conf, source_col = _source_label_for_row(row)
        if source_idx is not None:
            n_slices = int(row.get("n_slices", 0) or 0)
            if n_slices <= 0:
                stack = load_stack(row["path"], stack_index=int(row["stack_index"]))
                n_slices = int(stack.shape[0])
            ref_idx = int(np.clip(source_idx, 0, n_slices - 1))
            label_source = str(row.get("label_source", "source"))
            if not label_source or label_source == "nan":
                label_source = "source"
            confidence = float(source_conf if source_conf is not None else 1.0)
            notes = f"source_column={source_col}"
            if label_source != "source":
                notes += f"; imported_label_source={label_source}"
        else:
            stack = load_stack(row["path"], stack_index=int(row["stack_index"]))
            n_slices = int(stack.shape[0])
            ref_idx, confidence, used_voters = _surrogate_label(stack, voters)
            label_source = "surrogate"
            notes = f"surrogate_voters={used_voters}"
            if bool(label_cfg.get("leave_one_out", True)):
                for target in voters:
                    loo_voters = [name for name in voters if name != target]
                    if not loo_voters:
                        continue
                    loo_idx, loo_conf, loo_used = _surrogate_label(stack, loo_voters)
                    loo_rows.append(
                        {
                            "stack_id": row["stack_id"],
                            "dataset": row["dataset"],
                            "target_measure": target,
                            "reference_focus_index": loo_idx,
                            "label_confidence": loo_conf,
                            "label_source": "leave_one_out_surrogate",
                            "voters": loo_used,
                        }
                    )
        label_rows.append(
            {
                "stack_id": row["stack_id"],
                "dataset": row["dataset"],
                "group_id": row["group_id"],
                "stack_index": int(row["stack_index"]),
                "n_slices": int(n_slices),
                "reference_focus_index": int(ref_idx),
                "label_source": label_source,
                "label_confidence": float(confidence),
                "notes": notes,
            }
        )
    labels = pd.DataFrame(label_rows)
    summary = (
        labels.groupby(["dataset", "label_source"])
        .size()
        .reset_index(name="n_stacks")
        .sort_values(["dataset", "label_source"])
    )
    loo = pd.DataFrame(loo_rows)
    return labels, summary, loo

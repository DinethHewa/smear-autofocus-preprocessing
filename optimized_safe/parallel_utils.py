from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_ordered(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    num_workers: int = 1,
    chunksize: int = 1,
    disable_parallel: bool = False,
) -> List[R]:
    """Run independent tasks while preserving deterministic result order."""
    item_list = list(items)
    if disable_parallel or num_workers <= 1 or len(item_list) <= 1:
        return [func(item) for item in item_list]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(func, item_list, chunksize=max(1, int(chunksize))))

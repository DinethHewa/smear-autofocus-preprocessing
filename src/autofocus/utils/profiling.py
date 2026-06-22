from __future__ import annotations

import time
from typing import Optional


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is None:
            self.elapsed = None
        else:
            self.elapsed = time.perf_counter() - self.start_time

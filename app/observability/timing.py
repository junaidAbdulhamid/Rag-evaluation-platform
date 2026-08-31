"""Stage timing. A deliberately tiny helper; the trace recorder and the experiment
runner both use it."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def record_ms(store: dict, key: str):
    """Time the ``with`` block and write its duration (milliseconds) to ``store[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - start) * 1000.0, 3)

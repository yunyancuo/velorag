from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


@contextmanager
def timer(name: str = "operation"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] took {elapsed:.4f}s")


class BenchResult:
    def __init__(self, name: str):
        self.name = name
        self.duration: float = 0.0
        self.metadata: dict[str, Any] = {}

    def __repr__(self):
        return f"BenchResult({self.name}: {self.duration:.4f}s)"


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        bench = BenchResult(func.__name__)
        bench.duration = elapsed
        bench.metadata["args_count"] = len(args)
        bench.metadata["kwargs"] = str(kwargs)[:100]
        return result, bench
    return wrapper

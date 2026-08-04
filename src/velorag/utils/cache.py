from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional


class MemoryCache:
    def __init__(self, max_entries: int = 10000):
        self._store: dict[str, Any] = {}
        self._max_entries = max_entries

    @lru_cache(maxsize=1024)
    def _get_cached(self, key: str):
        return self._store.get(key)

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any):
        if len(self._store) >= self._max_entries:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

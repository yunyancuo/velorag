from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class MemoryCache:
    """线程安全的有界 LRU + TTL 内存缓存。"""

    def __init__(self, max_entries: int = 10000, ttl_seconds: Optional[float] = None):
        self._store: OrderedDict[str, tuple[Any, Optional[float]]] = OrderedDict()
        self._max_entries = max(0, int(max_entries))
        self._ttl_seconds = float(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None
        self._lock = threading.RLock()

    def _is_expired(self, expires_at: Optional[float], now: float) -> bool:
        return expires_at is not None and now >= expires_at

    def get(self, key: str) -> Optional[Any]:
        if self._max_entries <= 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if self._is_expired(expires_at, time.monotonic()):
                del self._store[key]
                return None
            # 命中后移至队尾维护真正 LRU 语义
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if self._max_entries <= 0:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        expires_at = (time.monotonic() + float(ttl)) if ttl and ttl > 0 else None

        with self._lock:
            if key in self._store:
                self._store[key] = (value, expires_at)
                self._store.move_to_end(key)
                return

            now = time.monotonic()
            # 优先清理队头已过期条目
            while self._store and len(self._store) >= self._max_entries:
                oldest_key, (_, oldest_exp) = next(iter(self._store.items()))
                if self._is_expired(oldest_exp, now):
                    del self._store[oldest_key]
                else:
                    self._store.popitem(last=False)

            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            now = time.monotonic()
            expired = [k for k, (_, exp) in self._store.items() if self._is_expired(exp, now)]
            for k in expired:
                del self._store[k]
            return len(self._store)

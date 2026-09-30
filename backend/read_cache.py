"""Bounded process-local metadata cache; Pinecone remains the source of truth."""

import copy
import json
import time
from collections import OrderedDict
from functools import wraps
from threading import RLock


class ReadCache:
    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = RLock()

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def read(self, key, loader, ttl):
        # Serialize loads and writes so a late fetch cannot restore stale data.
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                saved_at, size, value = entry
                if ttl is None or time.monotonic() - saved_at < ttl:
                    self.entries.move_to_end(key)
                    return copy.deepcopy(value)
                self.bytes -= size
                del self.entries[key]
            value = loader()  # Exceptions are never cached.
            size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            if value and size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    _, (_, old_size, _) = self.entries.popitem(last=False)
                    self.bytes -= old_size
                self.entries[key] = (time.monotonic(), size, copy.deepcopy(value))
                self.bytes += size
            return value


def cached_read(ttl=60):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            key = (method.__name__, json.dumps([args, kwargs], sort_keys=True))
            return self._read_cache.read(key, lambda: method(self, *args, **kwargs), ttl)
        return wrapped
    return decorate


def invalidates_reads(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._read_cache.lock:
            self._read_cache.clear()
            try:
                return method(self, *args, **kwargs)
            finally:
                # Also invalidate partial writes and reads made inside a write.
                self._read_cache.clear()
    return wrapped

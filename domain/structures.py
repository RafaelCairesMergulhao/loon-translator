from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, Generic, Iterable, TypeVar


T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """Buffer limitado, cronológico e seguro para produtores concorrentes."""

    def __init__(self, capacity: int = 100) -> None:
        if capacity < 1:
            raise ValueError("capacity deve ser maior que zero")
        self._items: Deque[T] = deque(maxlen=capacity)
        self._lock = Lock()

    def insert(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        with self._lock:
            self._items.extend(items)

    def get_all(self) -> list[T]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


CircularAudioBuffer = CircularBuffer
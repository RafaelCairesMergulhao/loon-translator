from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

from domain.sorting import insertion_sort, percentile
from domain.structures import CircularBuffer

T = TypeVar("T")


class LatencyTracker:
    """Últimas latências de uma sessão; mediana e p95 saem do Quickselect em O(n)."""

    def __init__(self, capacity: int = 60) -> None:
        self._samples: CircularBuffer[float] = CircularBuffer(capacity)

    def add(self, milliseconds: float) -> None:
        self._samples.insert(float(milliseconds))

    def __len__(self) -> int:
        return len(self._samples)

    def percentile(self, rank: float) -> float | None:
        samples = self._samples.get_all()
        return percentile(samples, rank) if samples else None

    def summary(self) -> dict[str, float]:
        samples = self._samples.get_all()
        if not samples:
            return {}
        return {
            "count": float(len(samples)),
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
        }


class ProviderHealth:
    """Disjuntor por provedor: quem falhou há pouco vai para o fim da fila.

    A fila de provedores tem poucos itens e quase nunca muda de ordem entre
    uma frase e outra, o melhor caso da inserção direta (O(n) e estável: os
    provedores saudáveis mantêm a ordem de preferência do modo).
    """

    def __init__(
        self,
        cooldown_s: float = 30.0,
        smoothing: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_s = cooldown_s
        self.smoothing = smoothing
        self._clock = clock
        self._failed_at: dict[str, float] = {}
        self._latency: dict[str, float] = {}
        self._lock = threading.Lock()

    def success(self, name: str, milliseconds: float) -> None:
        with self._lock:
            self._failed_at.pop(name, None)
            previous = self._latency.get(name)
            self._latency[name] = (
                milliseconds
                if previous is None
                else previous + self.smoothing * (milliseconds - previous)
            )

    def failure(self, name: str) -> None:
        with self._lock:
            self._failed_at[name] = self._clock()

    def is_open(self, name: str) -> bool:
        with self._lock:
            failed = self._failed_at.get(name)
        return failed is not None and self._clock() - failed < self.cooldown_s

    def latency_ms(self, name: str) -> float | None:
        with self._lock:
            return self._latency.get(name)

    def order(self, entries: Iterable[T], name: Callable[[T], str] = str) -> list[T]:
        return insertion_sort(entries, key=lambda entry: self.is_open(name(entry)))

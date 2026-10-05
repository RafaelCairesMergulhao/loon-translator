"""Métodos de ordenação por comparação usados pelo Loon.

Cada função devolve uma nova lista ordenada, aceita ``key`` como ``sorted`` e,
se receber um ``SortStats``, conta comparações de chaves C(n) e movimentações
de registros M(n). Uma troca vale três movimentações.

=================  ========  =========  ========  ======  =======
Método             Melhor    Médio      Pior      Extra   Estável
=================  ========  =========  ========  ======  =======
Seleção direta     n²        n²         n²        O(1)    não
Bolha              n         n²         n²        O(1)    sim
Shakersort         n         n²         n²        O(1)    sim
Inserção direta    n         n²         n²        O(1)    sim
Shellsort (Knuth)  n log n   ~n^1,25    n^1,5     O(1)    não
Quicksort          n log n   n log n    n²        log n   não
Heapsort           n log n   n log n    n log n   O(1)    não
Mergesort          n         n log n    n log n   O(n)    sim
=================  ========  =========  ========  ======  =======

Onde cada um trabalha no aplicativo:

* Inserção direta: fila de provedores de tradução (n < 10, quase ordenada).
* Shellsort: lista de dispositivos de áudio (dezenas de itens, sem recursão).
* Mergesort: trechos de gírias, em que a estabilidade decide empates.
* Heapsort/heap binário: seleção dos arquivos mais recentes do cache de voz.
* Quicksort (partição de Hoare): Quickselect para mediana e p95 de latência.
* Seleção, bolha e shaker ficam para comparação em ``tools.benchmark_sorting``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
KeyFunc = Callable[[Any], Any]

_QUICK_CUTOFF = 16


@dataclass(slots=True)
class SortStats:
    comparisons: int = 0
    moves: int = 0


class _Records:
    """Vetor de pares (chave, item): a chave é calculada uma única vez."""

    __slots__ = ("data", "stats")

    def __init__(self, items: Iterable[Any], key: KeyFunc | None, stats: SortStats | None) -> None:
        self.data: list[tuple[Any, Any]] = [
            ((key(item) if key else item), item) for item in items
        ]
        self.stats = stats if stats is not None else SortStats()

    def less(self, i: int, j: int) -> bool:
        self.stats.comparisons += 1
        return self.data[i][0] < self.data[j][0]

    def less_key(self, left: Any, right: Any) -> bool:
        self.stats.comparisons += 1
        return left < right

    def swap(self, i: int, j: int) -> None:
        self.stats.moves += 3
        data = self.data
        data[i], data[j] = data[j], data[i]

    def items(self) -> list[Any]:
        return [item for _key, item in self.data]


def selection_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Seleciona o menor restante e o troca para a posição i: n-1 trocas no máximo."""
    records = _Records(items, key, stats)
    size = len(records.data)
    for i in range(size - 1):
        smallest = i
        for j in range(i + 1, size):
            if records.less(j, smallest):
                smallest = j
        if smallest != i:
            records.swap(i, smallest)
    return records.items()


def bubble_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Troca vizinhos fora de ordem; para quando uma passada não troca nada."""
    records = _Records(items, key, stats)
    end = len(records.data) - 1
    while end > 0:
        last_swap = 0
        for j in range(end):
            if records.less(j + 1, j):
                records.swap(j, j + 1)
                last_swap = j
        end = last_swap
    return records.items()


def shaker_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Bolha nos dois sentidos; as bordas encolhem até a última troca de cada passada."""
    records = _Records(items, key, stats)
    left, right = 0, len(records.data) - 1
    while left < right:
        last = left
        for j in range(left, right):
            if records.less(j + 1, j):
                records.swap(j, j + 1)
                last = j
        right = last
        last = right
        for j in range(right, left, -1):
            if records.less(j, j - 1):
                records.swap(j - 1, j)
                last = j
        left = last
    return records.items()


def _insertion(records: _Records, low: int, high: int) -> None:
    data = records.data
    stats = records.stats
    for i in range(low + 1, high + 1):
        current = data[i]
        stats.moves += 1
        j = i - 1
        while j >= low and records.less_key(current[0], data[j][0]):
            data[j + 1] = data[j]
            stats.moves += 1
            j -= 1
        data[j + 1] = current
        stats.moves += 1


def insertion_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Insere cada item na parte já ordenada; O(n) quando a entrada quase não muda."""
    records = _Records(items, key, stats)
    _insertion(records, 0, len(records.data) - 1)
    return records.items()


def shell_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Inserção com saltos h = 1, 4, 13, 40, ... (sequência de Knuth)."""
    records = _Records(items, key, stats)
    data = records.data
    size = len(data)
    gap = 1
    while gap < size // 3:
        gap = 3 * gap + 1
    while gap >= 1:
        for i in range(gap, size):
            current = data[i]
            records.stats.moves += 1
            j = i
            while j >= gap and records.less_key(current[0], data[j - gap][0]):
                data[j] = data[j - gap]
                records.stats.moves += 1
                j -= gap
            data[j] = current
            records.stats.moves += 1
        gap //= 3
    return records.items()


def _partition(records: _Records, low: int, high: int) -> tuple[int, int]:
    """Partição de Hoare com pivô pela mediana de três.

    Devolve (i, j) com ``low..j`` <= pivô <= ``i..high``.
    """
    middle = (low + high) // 2
    if records.less(middle, low):
        records.swap(middle, low)
    if records.less(high, low):
        records.swap(high, low)
    if records.less(high, middle):
        records.swap(high, middle)
    pivot = records.data[middle][0]
    data = records.data
    i, j = low, high
    while i <= j:
        while records.less_key(data[i][0], pivot):
            i += 1
        while records.less_key(pivot, data[j][0]):
            j -= 1
        if i <= j:
            if i != j:
                records.swap(i, j)
            i += 1
            j -= 1
    return i, j


def quick_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Quicksort iterativo: a menor partição é resolvida primeiro, então a pilha fica em O(log n).

    Partições com até 16 itens terminam na inserção direta.
    """
    records = _Records(items, key, stats)
    pending = [(0, len(records.data) - 1)]
    while pending:
        low, high = pending.pop()
        while high - low + 1 > _QUICK_CUTOFF:
            i, j = _partition(records, low, high)
            if j - low < high - i:
                pending.append((i, high))
                high = j
            else:
                pending.append((low, j))
                low = i
        _insertion(records, low, high)
    return records.items()


def _sift_down(records: _Records, root: int, end: int) -> None:
    data = records.data
    stats = records.stats
    item = data[root]
    stats.moves += 1
    child = 2 * root + 1
    while child <= end:
        if child < end and records.less(child, child + 1):
            child += 1
        if not records.less_key(item[0], data[child][0]):
            break
        data[root] = data[child]
        stats.moves += 1
        root = child
        child = 2 * root + 1
    data[root] = item
    stats.moves += 1


def heap_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Constrói um heap máximo em O(n) e retira o maior n-1 vezes."""
    records = _Records(items, key, stats)
    size = len(records.data)
    for start in range(size // 2 - 1, -1, -1):
        _sift_down(records, start, size - 1)
    for end in range(size - 1, 0, -1):
        records.swap(0, end)
        _sift_down(records, 0, end - 1)
    return records.items()


def _merge_sort(records: _Records, buffer: list[tuple[Any, Any]], low: int, high: int) -> None:
    if high - low < 2:
        return
    middle = (low + high) // 2
    _merge_sort(records, buffer, low, middle)
    _merge_sort(records, buffer, middle, high)
    data = records.data
    if not records.less_key(data[middle][0], data[middle - 1][0]):
        return
    buffer[low:high] = data[low:high]
    records.stats.moves += high - low
    i, j, k = low, middle, low
    while i < middle and j < high:
        if records.less_key(buffer[j][0], buffer[i][0]):
            data[k] = buffer[j]
            j += 1
        else:
            data[k] = buffer[i]
            i += 1
        k += 1
        records.stats.moves += 1
    while i < middle:
        data[k] = buffer[i]
        i += 1
        k += 1
        records.stats.moves += 1


def merge_sort(items: Iterable[T], key: KeyFunc | None = None, stats: SortStats | None = None) -> list[T]:
    """Intercalação estável: em empate, o item da metade esquerda sai primeiro.

    Metades já em ordem não são intercaladas, o que dá O(n) em entrada ordenada.
    """
    records = _Records(items, key, stats)
    buffer = list(records.data)
    _merge_sort(records, buffer, 0, len(records.data))
    return records.items()


def quickselect(values: Sequence[Any], k: int, key: KeyFunc | None = None) -> Any:
    """k-ésimo menor item (base 0) em O(n) médio, com a partição do Quicksort."""
    if not 0 <= k < len(values):
        raise IndexError("k fora do intervalo")
    records = _Records(values, key, None)
    low, high = 0, len(records.data) - 1
    while low < high:
        i, j = _partition(records, low, high)
        if k <= j:
            high = j
        elif k >= i:
            low = i
        else:
            break
    return records.data[k][1]


def percentile(values: Sequence[float], rank: float) -> float:
    """Percentil pelo posto mais próximo, sem ordenar a amostra inteira."""
    if not values:
        raise ValueError("amostra vazia")
    position = round(max(0.0, min(100.0, rank)) / 100.0 * (len(values) - 1))
    return float(quickselect(values, position))


class BinaryHeap(Generic[T]):
    """Heap mínimo sobre um vetor: pai em (i-1)//2, filhos em 2i+1 e 2i+2."""

    def __init__(self, key: KeyFunc | None = None) -> None:
        self._key = key or (lambda item: item)
        self._data: list[tuple[Any, int, T]] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._data)

    def push(self, item: T) -> None:
        self._data.append((self._key(item), self._counter, item))
        self._counter += 1
        self._sift_up(len(self._data) - 1)

    def peek(self) -> T:
        return self._data[0][2]

    def pop(self) -> T:
        data = self._data
        last = data.pop()
        if not data:
            return last[2]
        top = data[0]
        data[0] = last
        self._sift_down(0)
        return top[2]

    def replace(self, item: T) -> T:
        """Retira o menor e insere ``item`` com um único ``sift_down``."""
        top = self._data[0]
        self._data[0] = (self._key(item), self._counter, item)
        self._counter += 1
        self._sift_down(0)
        return top[2]

    def _sift_up(self, index: int) -> None:
        data = self._data
        entry = data[index]
        while index > 0:
            parent = (index - 1) // 2
            if data[parent][:2] <= entry[:2]:
                break
            data[index] = data[parent]
            index = parent
        data[index] = entry

    def _sift_down(self, index: int) -> None:
        data = self._data
        size = len(data)
        entry = data[index]
        child = 2 * index + 1
        while child < size:
            if child + 1 < size and data[child + 1][:2] < data[child][:2]:
                child += 1
            if entry[:2] <= data[child][:2]:
                break
            data[index] = data[child]
            index = child
            child = 2 * index + 1
        data[index] = entry


def nlargest(items: Iterable[T], count: int, key: KeyFunc | None = None) -> list[T]:
    """Os ``count`` maiores em O(n log k): o heap mínimo guarda só o top-k."""
    if count <= 0:
        return []
    ranking = key or (lambda item: item)
    heap: BinaryHeap[T] = BinaryHeap(ranking)
    for item in items:
        if len(heap) < count:
            heap.push(item)
        elif ranking(heap.peek()) < ranking(item):
            heap.replace(item)
    result: list[T] = []
    while heap:
        result.append(heap.pop())
    result.reverse()
    return result


ALGORITHMS: dict[str, Callable[..., list]] = {
    "Seleção direta": selection_sort,
    "Bolha": bubble_sort,
    "Shakersort": shaker_sort,
    "Inserção direta": insertion_sort,
    "Shellsort": shell_sort,
    "Quicksort": quick_sort,
    "Heapsort": heap_sort,
    "Mergesort": merge_sort,
}

STABLE = frozenset({"Bolha", "Shakersort", "Inserção direta", "Mergesort"})

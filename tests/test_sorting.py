import random

import pytest

from domain.sorting import (
    ALGORITHMS,
    STABLE,
    BinaryHeap,
    SortStats,
    insertion_sort,
    merge_sort,
    nlargest,
    percentile,
    quick_sort,
    quickselect,
    selection_sort,
)

ALL = list(ALGORITHMS.items())


def _inputs() -> dict[str, list[int]]:
    generator = random.Random(42)
    randomized = [generator.randint(-500, 500) for _ in range(300)]
    nearly = sorted(randomized)
    for _ in range(10):
        i, j = generator.randrange(300), generator.randrange(300)
        nearly[i], nearly[j] = nearly[j], nearly[i]
    return {
        "vazio": [],
        "um": [7],
        "dois": [2, 1],
        "aleatório": randomized,
        "ordenado": sorted(randomized),
        "invertido": sorted(randomized, reverse=True),
        "quase ordenado": nearly,
        "repetidos": [3, 1, 3, 1, 3, 1, 2, 2, 2] * 20,
        "iguais": [5] * 50,
    }


@pytest.mark.parametrize(("name", "algorithm"), ALL)
@pytest.mark.parametrize("case", list(_inputs()))
def test_every_method_matches_sorted(name, algorithm, case) -> None:
    data = _inputs()[case]
    original = list(data)
    assert algorithm(data) == sorted(data), (name, case)
    assert data == original, "a entrada não pode ser alterada"


@pytest.mark.parametrize(("name", "algorithm"), ALL)
def test_key_function_is_respected(name, algorithm) -> None:
    words = ["banana", "Kiwi", "abacaxi", "uva", "Manga", "figo"]
    assert algorithm(words, key=str.casefold) == sorted(words, key=str.casefold), name


@pytest.mark.parametrize("name", sorted(STABLE))
def test_stable_methods_keep_the_order_of_ties(name) -> None:
    records = [(value % 5, index) for index, value in enumerate(random.Random(3).sample(range(200), 200))]
    result = ALGORITHMS[name](records, key=lambda item: item[0])
    assert result == sorted(records, key=lambda item: item[0])


def test_insertion_is_linear_on_sorted_input() -> None:
    stats = SortStats()
    insertion_sort(range(1_000), stats=stats)
    assert stats.comparisons == 999


def test_merge_skips_merging_sorted_halves() -> None:
    stats = SortStats()
    merge_sort(range(1_024), stats=stats)
    assert stats.comparisons == 1_023
    assert stats.moves == 0


def test_selection_makes_at_most_n_minus_one_swaps() -> None:
    stats = SortStats()
    selection_sort(list(range(100, 0, -1)), stats=stats)
    assert stats.moves <= 3 * 99
    assert stats.comparisons == 100 * 99 // 2


def test_quicksort_handles_the_worst_case_for_a_naive_pivot() -> None:
    stats = SortStats()
    data = list(range(5_000))
    assert quick_sort(data, stats=stats) == data
    assert stats.comparisons < 5_000 * 20


def test_quickselect_and_percentile() -> None:
    values = random.Random(9).sample(range(1_000), 500)
    ordered = sorted(values)
    for k in (0, 1, 250, 498, 499):
        assert quickselect(values, k) == ordered[k]
    assert percentile(values, 50) == ordered[round(0.5 * 499)]
    assert percentile(values, 95) == ordered[round(0.95 * 499)]
    with pytest.raises(IndexError):
        quickselect(values, 500)


def test_heap_pops_in_order_and_nlargest_is_top_k() -> None:
    values = random.Random(5).sample(range(10_000), 800)
    heap: BinaryHeap[int] = BinaryHeap()
    for value in values:
        heap.push(value)
    assert [heap.pop() for _ in range(len(values))] == sorted(values)
    assert nlargest(values, 10) == sorted(values, reverse=True)[:10]
    assert nlargest(values, 0) == []
    assert nlargest([3, 1], 5) == [3, 1]

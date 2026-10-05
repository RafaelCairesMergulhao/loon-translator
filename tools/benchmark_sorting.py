"""Compara os oito métodos de ordenação de ``domain.sorting``.

Uso:
    python -m tools.benchmark_sorting
    python -m tools.benchmark_sorting --sizes 100 1000 5000 --seed 7

Para cada tamanho e distribuição mostra tempo, comparações C(n) e
movimentações M(n). Métodos quadráticos são pulados acima de ``--quadratic-limit``.
"""

from __future__ import annotations

import argparse
import random
import time

from domain.sorting import ALGORITHMS, STABLE, SortStats

QUADRATIC = {"Seleção direta", "Bolha", "Shakersort", "Inserção direta"}


def distributions(size: int, rng: random.Random) -> dict[str, list[int]]:
    ordered = list(range(size))
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    nearly = ordered[:]
    for _ in range(max(1, size // 20)):
        i, j = rng.randrange(size), rng.randrange(size)
        nearly[i], nearly[j] = nearly[j], nearly[i]
    return {
        "aleatório": shuffled,
        "ordenado": ordered,
        "invertido": ordered[::-1],
        "quase ordenado": nearly,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 2000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quadratic-limit", type=int, default=2000)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    header = f"{'método':<18}{'estável':<9}{'tempo (ms)':>12}{'C(n)':>14}{'M(n)':>14}"
    for size in args.sizes:
        for label, values in distributions(size, rng).items():
            expected = sorted(values)
            print(f"\nn = {size}, {label}")
            print(header)
            print("-" * len(header))
            for name, method in ALGORITHMS.items():
                if name in QUADRATIC and size > args.quadratic_limit:
                    continue
                stats = SortStats()
                started = time.perf_counter()
                result = method(values, stats=stats)
                elapsed = (time.perf_counter() - started) * 1000
                if result != expected:
                    raise AssertionError(f"{name} não ordenou n={size} {label}")
                stable = "sim" if name in STABLE else "não"
                print(f"{name:<18}{stable:<9}{elapsed:>12.2f}{stats.comparisons:>14,}{stats.moves:>14,}")


if __name__ == "__main__":
    main()

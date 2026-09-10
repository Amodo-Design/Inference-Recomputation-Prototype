from __future__ import annotations

from app.allocation import allocate
from app.service import ModelAllocation


def alloc(*specs: tuple[str, int]) -> list[ModelAllocation]:
    return [ModelAllocation(model=m, percent=p, concurrency=1) for m, p in specs]


def test_exact_split():
    slices = allocate(list(range(10)), alloc(("a", 50), ("b", 25), ("c", 25)))
    assert [len(s) for s in slices] == [5, 3, 2] or [len(s) for s in slices] == [5, 2, 3]
    assert sum(len(s) for s in slices) == 10
    # Contiguous, order-preserving: concatenation reproduces the input.
    assert [x for s in slices for x in s] == list(range(10))


def test_largest_remainder_totals_exactly():
    # 33/33/34 over 7 items: floors are 2/2/2 (sum 6), largest remainder
    # (34% -> 2.38) gets the leftover item.
    slices = allocate(list(range(7)), alloc(("a", 33), ("b", 33), ("c", 34)))
    assert sum(len(s) for s in slices) == 7
    assert [len(s) for s in slices] == [2, 2, 3]


def test_single_allocation_gets_everything():
    slices = allocate(list(range(3)), alloc(("a", 100)))
    assert slices == [[0, 1, 2]]


def test_more_models_than_items():
    # 3 models, 2 items: someone gets an empty slice; total still exact.
    slices = allocate([1, 2], alloc(("a", 34), ("b", 33), ("c", 33)))
    assert sum(len(s) for s in slices) == 2
    assert len(slices) == 3


def test_empty_items():
    assert allocate([], alloc(("a", 100))) == [[]]

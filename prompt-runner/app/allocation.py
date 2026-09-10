"""Largest-remainder split of the selected prompt runs across model
allocations. Contiguous slices (not round-robin) keep each model's share a
clean block, and floor+remainder guarantees the sizes total exactly
len(items) — no prompt is dropped or double-sent."""

from __future__ import annotations

import math
from typing import Any, Sequence


def allocate(items: Sequence[Any], allocations: Sequence[Any]) -> list[list[Any]]:
    n = len(items)
    quotas = [a.percent * n / 100 for a in allocations]
    sizes = [math.floor(q) for q in quotas]
    leftover = n - sum(sizes)
    # Hand leftover items to the largest fractional remainders (ties: first
    # in list order, via stable sort on -remainder).
    order = sorted(range(len(allocations)), key=lambda i: quotas[i] - sizes[i], reverse=True)
    for i in order[:leftover]:
        sizes[i] += 1
    slices: list[list[Any]] = []
    cursor = 0
    for size in sizes:
        slices.append(list(items[cursor : cursor + size]))
        cursor += size
    return slices

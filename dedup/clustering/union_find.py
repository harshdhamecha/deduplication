"""Union-Find (disjoint-set) over arbitrary hashable ids.

WHY this lives in its own module used from Stage 2 onward (not only Stage 4):
duplicates are *transitive* — if A~B and B~C are flagged, {A,B,C} is one group,
not two pairs. Any stage that turns pairwise flags into "which set does each item
belong to" needs exactly this structure, so we implement it once, correctly, and
reuse it. Stage 4 builds its cross-stage clustering on top of this same class.

Implementation uses path compression + union by rank, giving near-O(1) amortised
operations (inverse-Ackermann) — so clustering millions of flagged pairs is cheap.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}
        self._rank: dict[Hashable, int] = {}

    def add(self, x: Hashable) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: Hashable) -> Hashable:
        # Iterative find with path compression — recursion would risk a stack
        # overflow on a long chain (e.g. a sorted run of near-identical frames).
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Attach the shorter tree under the taller — keeps trees shallow.
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> list[list[Hashable]]:
        """Return the connected components as lists of members."""
        groups: dict[Hashable, list[Hashable]] = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return list(groups.values())


def connected_components(
    items: Iterable[Hashable], pairs: Iterable[tuple[Hashable, Hashable]]
) -> list[list[Hashable]]:
    """Convenience: build components from a node set + a list of equivalence pairs.

    Singletons (items in no pair) are included as their own component, so callers
    get a complete partition of ``items``.
    """
    uf = UnionFind()
    for it in items:
        uf.add(it)
    for a, b in pairs:
        uf.union(a, b)
    return uf.components()

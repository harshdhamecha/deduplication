"""Near-duplicate candidate retrieval over 64-bit perceptual hashes.

Three strategies behind one interface, so the pipeline can pick by scale and a
test can assert they all agree. All three are *exact* for the "find every pair
within Hamming distance t" problem — they differ only in how much work they do to
get there. That exactness is what the recall test pins down: multi-index hashing
and the BK-tree must return precisely the same pair set as brute force.

Hashes are represented as Python ints (64-bit). Hamming distance is popcount of
the XOR — ``int.bit_count()`` on 3.10+.

    Strategy            Candidate generation        Use when
    ------------------- --------------------------  -------------------------
    BruteForceHamming   all N*(N-1)/2 pairs          small N (exact, simplest)
    MultiIndexHashing   pigeonhole-pruned buckets    medium/large N
    BKTree              metric-tree range queries    comparison / learning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable

from dedup import get_logger

logger = get_logger()

Pair = tuple[Hashable, Hashable]


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer hash codes (popcount of XOR)."""
    return (a ^ b).bit_count()


class NearDuplicateSearch(ABC):
    """Finds all id-pairs whose hashes are within ``threshold`` Hamming distance."""

    @abstractmethod
    def find_pairs(self, hashes: dict[Hashable, int], threshold: int) -> set[Pair]:
        ...


def _ordered(a: Hashable, b: Hashable) -> Pair:
    """Canonicalise a pair so {a,b} and {b,a} dedupe to one entry."""
    return (a, b) if str(a) <= str(b) else (b, a)


class BruteForceHamming(NearDuplicateSearch):
    """All-pairs comparison. O(N^2) — the ground truth other strategies match."""

    def find_pairs(self, hashes: dict[Hashable, int], threshold: int) -> set[Pair]:
        items = list(hashes.items())
        out: set[Pair] = set()
        for i in range(len(items)):
            id_i, h_i = items[i]
            for j in range(i + 1, len(items)):
                id_j, h_j = items[j]
                if hamming(h_i, h_j) <= threshold:
                    out.add(_ordered(id_i, id_j))
        return out


class MultiIndexHashing(NearDuplicateSearch):
    """Pigeonhole-pruned candidate generation, then exact verification.

    Pigeonhole principle: split each 64-bit code into (threshold+1) disjoint
    chunks. If two codes differ in at most ``threshold`` bits, those differing
    bits cannot touch all (threshold+1) chunks — so at least one chunk must be
    *identical* between them. Therefore: bucket ids by (chunk_position,
    chunk_value); any true near-pair shares at least one bucket. We only compare
    ids that co-occur in a bucket, then verify exact Hamming to drop the
    false positives the bucketing lets through.

    This is exact (no recall loss) and turns an O(N^2) scan into roughly
    O(N * bucket_occupancy) — a big win when duplicates are sparse.
    """

    def __init__(self, bits: int = 64):
        self.bits = bits

    def _chunks(self, h: int, num_chunks: int) -> list[tuple[int, int]]:
        # Split `bits` into num_chunks near-equal slices; return (position, value).
        out = []
        base, extra = divmod(self.bits, num_chunks)
        shift = 0
        for pos in range(num_chunks):
            width = base + (1 if pos < extra else 0)
            mask = (1 << width) - 1
            out.append((pos, (h >> shift) & mask))
            shift += width
        return out

    def find_pairs(self, hashes: dict[Hashable, int], threshold: int) -> set[Pair]:
        num_chunks = threshold + 1  # the pigeonhole guarantee needs t+1 chunks
        buckets: dict[tuple[int, int], list[Hashable]] = defaultdict(list)
        for id_, h in hashes.items():
            for key in self._chunks(h, num_chunks):
                buckets[key].append(id_)

        # Gather candidate pairs that share any chunk bucket, then verify exactly.
        out: set[Pair] = set()
        seen: set[Pair] = set()
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = _ordered(ids[i], ids[j])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if hamming(hashes[pair[0]], hashes[pair[1]]) <= threshold:
                        out.add(pair)
        return out


class _BKNode:
    __slots__ = ("id", "hash", "children")

    def __init__(self, id_: Hashable, h: int):
        self.id = id_
        self.hash = h
        self.children: dict[int, _BKNode] = {}  # edge label = distance to child


class BKTree(NearDuplicateSearch):
    """Burkhard-Keller tree: a metric tree keyed by Hamming distance.

    Insert each code as a child edge labelled by its distance to the current
    node. A range query for radius t prunes using the triangle inequality: from a
    node at distance d to the query, only children with edge label in
    [d-t, d+t] can hold matches. Exact, and typically sublinear when the data is
    clustered — included mainly to contrast a metric-tree approach with the
    bit-bucketing of multi-index hashing.
    """

    def find_pairs(self, hashes: dict[Hashable, int], threshold: int) -> set[Pair]:
        root: _BKNode | None = None
        out: set[Pair] = set()
        for id_, h in hashes.items():
            if root is None:
                root = _BKNode(id_, h)
                continue
            # Query existing tree for neighbours within threshold, then insert.
            for nb_id in self._query(root, h, threshold):
                out.add(_ordered(id_, nb_id))
            self._insert(root, id_, h)
        return out

    def _insert(self, root: _BKNode, id_: Hashable, h: int) -> None:
        node = root
        while True:
            d = hamming(h, node.hash)
            if d == 0:  # identical code: still a distinct id, hang it off distance 0
                if 0 in node.children:
                    node = node.children[0]
                    continue
                node.children[0] = _BKNode(id_, h)
                return
            if d in node.children:
                node = node.children[d]
            else:
                node.children[d] = _BKNode(id_, h)
                return

    def _query(self, root: _BKNode, h: int, t: int) -> list[Hashable]:
        found: list[Hashable] = []
        stack = [root]
        while stack:
            node = stack.pop()
            d = hamming(h, node.hash)
            if d <= t:
                found.append(node.id)
            lo, hi = d - t, d + t
            for label, child in node.children.items():
                if lo <= label <= hi:
                    stack.append(child)
        return found


_STRATEGIES = {
    "bruteforce": BruteForceHamming,
    "multiindex": MultiIndexHashing,
    "bktree": BKTree,
}

# Above this many items, the O(N^2) brute-force scan stops being instant
# (~12.5M comparisons at 5k); multi-index hashing's pruning takes over. The
# cutover is a speed choice only — recall is identical either side of it.
_AUTO_BRUTEFORCE_MAX = 5000


def get_search_strategy(name: str, n_items: int) -> NearDuplicateSearch:
    """Pick a strategy, logging the decision. ``name`` may be a concrete strategy
    or "auto" (brute force for small N, multi-index hashing otherwise)."""
    if name == "auto":
        chosen = "bruteforce" if n_items <= _AUTO_BRUTEFORCE_MAX else "multiindex"
        logger.info(
            "stage2 search: %s (auto) — %d items %s %d brute-force cutover.",
            chosen, n_items,
            "<=" if chosen == "bruteforce" else ">", _AUTO_BRUTEFORCE_MAX,
        )
        return _STRATEGIES[chosen]()
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown search strategy '{name}' (have: {sorted(_STRATEGIES)})")
    logger.info("stage2 search: %s (explicit) — %d items", name, n_items)
    return _STRATEGIES[name]()

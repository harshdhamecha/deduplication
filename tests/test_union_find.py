"""Union-Find correctness, including the transitive-duplicate property that
makes clustering (not pairing) the right model: A~B, B~C => {A,B,C}."""

from dedup.clustering.union_find import UnionFind, connected_components


def test_basic_union_and_find():
    uf = UnionFind()
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")
    assert uf.find("a") != uf.find("c")


def test_transitive_grouping():
    # A~B and B~C must collapse to a single 3-member component.
    comps = connected_components(["A", "B", "C", "D"], [("A", "B"), ("B", "C")])
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 3]                       # {D} and {A,B,C}
    big = next(c for c in comps if len(c) == 3)
    assert set(big) == {"A", "B", "C"}


def test_singletons_are_their_own_component():
    comps = connected_components(["x", "y", "z"], [])
    assert sorted(len(c) for c in comps) == [1, 1, 1]


def test_components_cover_every_item_exactly_once():
    items = list(range(10))
    pairs = [(0, 1), (1, 2), (5, 6)]
    comps = connected_components(items, pairs)
    flat = [m for c in comps for m in c]
    assert sorted(flat) == items                 # partition: no loss, no duplication


def test_large_chain_no_recursion_error():
    # A long chain would blow a recursive find()'s stack; ours is iterative.
    n = 50_000
    pairs = [(i, i + 1) for i in range(n - 1)]
    comps = connected_components(range(n), pairs)
    assert len(comps) == 1 and len(comps[0]) == n

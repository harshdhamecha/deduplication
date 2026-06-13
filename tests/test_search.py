"""The recall-agreement test: multi-index hashing and the BK-tree must return
EXACTLY the same near-duplicate pairs as brute force.

This is the load-bearing correctness test for Stage 2. Multi-index hashing and
the BK-tree are optimisations of an exact problem ("all pairs within Hamming t"),
so any disagreement with the O(N^2) ground truth is a bug, not a tradeoff. We
check across a range of thresholds and on randomised data.
"""

import random

import pytest

from dedup.hashing.search import (
    BKTree,
    BruteForceHamming,
    MultiIndexHashing,
    get_search_strategy,
    hamming,
)


def _random_hashes(n, bits=64, seed=0):
    rng = random.Random(seed)
    # Mix purely-random codes with deliberate near-duplicates so there ARE pairs
    # to find (random 64-bit codes are almost never within 8 of each other).
    hashes = {}
    for i in range(n):
        hashes[f"id{i}"] = rng.getrandbits(bits)
    # Plant near-duplicates: flip a few bits of some existing codes.
    base_ids = list(hashes)
    for k in range(n // 4):
        src = hashes[rng.choice(base_ids)]
        flips = rng.randint(0, 12)
        h = src
        for _ in range(flips):
            h ^= 1 << rng.randrange(bits)
        hashes[f"dup{k}"] = h
    return hashes


def test_hamming_basic():
    assert hamming(0b1010, 0b1000) == 1
    assert hamming(0, (1 << 64) - 1) == 64


@pytest.mark.parametrize("threshold", [0, 2, 4, 8, 12])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_multiindex_and_bktree_agree_with_bruteforce(threshold, seed):
    hashes = _random_hashes(200, seed=seed)

    truth = BruteForceHamming().find_pairs(hashes, threshold)
    mih = MultiIndexHashing().find_pairs(hashes, threshold)
    bk = BKTree().find_pairs(hashes, threshold)

    assert mih == truth, f"multi-index disagreed at t={threshold}"
    assert bk == truth, f"bk-tree disagreed at t={threshold}"


def test_multiindex_actually_finds_planted_near_dup():
    a = 0
    b = (1 << 3) - 1          # differs from a in exactly 3 bits  -> near
    far = (1 << 40) - 1       # 40 bits set -> Hamming 40 from a   -> far
    hashes = {"a": a, "b": b, "far": far}
    pairs = MultiIndexHashing().find_pairs(hashes, threshold=4)
    assert ("a", "b") in pairs
    assert all("far" not in p for p in pairs)


def test_auto_strategy_selection_by_scale():
    assert isinstance(get_search_strategy("auto", 100), BruteForceHamming)
    assert isinstance(get_search_strategy("auto", 50_000), MultiIndexHashing)

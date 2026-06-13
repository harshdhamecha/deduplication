"""The scale-adaptive index selector — the architectural centerpiece.

Given an estimated vector count, the embedding dimension, and a RAM budget, pick
the cheapest index that still meets recall needs, and LOG the decision verbatim
with the numbers behind it. The whole thesis of the project is here: never trade
accuracy for speed you don't need.

Decision policy (overridable: set stage3.index_type to a concrete tier to force):

    Tier            Chosen when                         What you accept
    --------------- ----------------------------------- ----------------------
    IndexFlatIP     fits in RAM AND N < flat_max        nothing (exact)
    IndexIVFFlat    N < ivf_max (full vectors fit RAM)  cell-pruning recall hit
    IndexIVFPQ      otherwise (too big for RAM)         PQ error + re-rank to fix

The RAM check is the real gate for Flat: even below the count threshold, if the
fp32 matrix wouldn't fit the budget we step down a tier. We compute the actual
footprint and print it, so the choice is auditable rather than magical.
"""

from __future__ import annotations

import math

from dedup import get_logger
from dedup.config import Stage3Config
from dedup.indexing.base import VectorIndex
from dedup.indexing.flat import FlatIndex
from dedup.indexing.ivfflat import IvfFlatIndex
from dedup.indexing.ivfpq import IvfPqIndex

logger = get_logger()

# Count thresholds per the spec's scale tiers. These are deliberately round
# numbers, not tuned constants — the RAM check below is what actually protects us.
FLAT_MAX = 1_000_000      # < 1M  -> exact Flat
IVFFLAT_MAX = 10_000_000  # 1M-10M -> IVFFlat; 10M+ -> IVFPQ


def _fp32_gb(n: int, dim: int) -> float:
    return n * dim * 4 / 1024**3


def _auto_nlist(n: int) -> int:
    """nlist ~ sqrt(N), clamped. More cells = finer partitions = faster queries
    but each needs enough training points, so we don't let it run away."""
    return max(1, min(65536, int(4 * math.sqrt(max(n, 1)))))


def select_index(n_vectors: int, dim: int, cfg: Stage3Config) -> VectorIndex:
    """Build and return the index for this scale, logging why."""
    # ---- explicit override path ----
    if cfg.index_type != "auto":
        logger.info("index: %s (EXPLICIT override; auto-selection skipped) for "
                    "%d x %dd vectors", cfg.index_type, n_vectors, dim)
        return _build(cfg.index_type, n_vectors, dim, cfg, auto=False)

    # ---- auto path: decide tier, then log the reasoning with real numbers ----
    flat_gb = _fp32_gb(n_vectors, dim)
    fits_ram = flat_gb <= cfg.ram_budget_gb

    if n_vectors < FLAT_MAX and fits_ram:
        logger.info(
            "selected IndexFlatIP: %s x %dd ~= %.2fGB fits in %.0fGB budget and "
            "N < %s — exact search preferred (zero approximation, perfect recall).",
            f"{n_vectors:,}", dim, flat_gb, cfg.ram_budget_gb, f"{FLAT_MAX:,}",
        )
        return _build("flat", n_vectors, dim, cfg, auto=True)

    if n_vectors < IVFFLAT_MAX and fits_ram:
        nlist = _auto_nlist(n_vectors)
        logger.info(
            "selected IndexIVFFlat: %s x %dd ~= %.2fGB still fits %.0fGB but N >= %s "
            "makes a full scan slow; partitioning into nlist=%d cells (full vectors "
            "retained -> only cell-pruning approximation).",
            f"{n_vectors:,}", dim, flat_gb, cfg.ram_budget_gb, f"{FLAT_MAX:,}", nlist,
        )
        return _build("ivfflat", n_vectors, dim, cfg, auto=True, nlist=nlist)

    nlist = _auto_nlist(n_vectors)
    pq_bytes = cfg.pq_m * cfg.pq_nbits / 8
    logger.info(
        "selected IndexIVFPQ: %s x %dd ~= %.2fGB exceeds %.0fGB budget (or N >= %s); "
        "product-quantizing to ~%.0f bytes/vector (m=%d, nbits=%d), nlist=%d, "
        "exact re-rank of top-%d %s.",
        f"{n_vectors:,}", dim, flat_gb, cfg.ram_budget_gb, f"{IVFFLAT_MAX:,}",
        pq_bytes, cfg.pq_m, cfg.pq_nbits, nlist, cfg.rerank_k,
        "ENABLED (recovers PQ precision)" if cfg.rerank else "DISABLED",
    )
    return _build("ivfpq", n_vectors, dim, cfg, auto=True, nlist=nlist)


def _build(kind: str, n: int, dim: int, cfg: Stage3Config, auto: bool,
           nlist: int | None = None) -> VectorIndex:
    """Instantiate a tier. In auto mode nlist is derived from N; in explicit mode
    we honour the configured nlist/nprobe/m/nbits as given."""
    nlist = nlist if (auto and nlist is not None) else cfg.nlist
    if kind == "flat":
        return FlatIndex(dim)
    if kind == "ivfflat":
        return IvfFlatIndex(dim, nlist=nlist, nprobe=cfg.nprobe)
    if kind == "ivfpq":
        return IvfPqIndex(dim, nlist=nlist, nprobe=cfg.nprobe, m=cfg.pq_m,
                          nbits=cfg.pq_nbits, rerank=cfg.rerank, rerank_k=cfg.rerank_k)
    raise ValueError(f"Unknown index_type '{kind}' (use auto|flat|ivfflat|ivfpq)")

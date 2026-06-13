"""ANN indexing — the architectural centerpiece (filled in Step 4).

Planned contents:
  base.py      VectorIndex interface (add / search / save / load).
  flat.py      IndexFlatIP   — exact cosine via normalized vectors; small N.
  ivfflat.py   IndexIVFFlat  — partitioned, full vectors retained; medium N.
  ivfpq.py     IndexIVFPQ    — product-quantized + disk-backed; large N; with
               exact re-rank of top candidates on the memmapped vectors.
  selector.py  THE auto-selector. Chooses a tier from estimated vector count +
               available RAM, is overridable by config, and LOGS the decision
               verbatim, e.g.:
                 "selected IndexFlatIP: 320K x 768d ~= 0.9GB fits in 24GB budget,
                  exact search preferred (zero approximation)."

Scale tiers (do not collapse into one path):
    < 1M vectors   -> IndexFlatIP   (exact, zero approximation)
    1M - 10M       -> IndexIVFFlat  (partitioned, full vectors)
    10M+           -> IndexIVFPQ    (compressed, disk-backed, re-rank)
"""

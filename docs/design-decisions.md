# Design Decisions & Tradeoffs

A walk through the choices that make this a deduplication *system* rather than a
script, and what each one trades away. Written to be adaptable into a blog post.

## 1. Why a four-stage cascade (cheap → expensive)

Each stage is strictly more expensive and more semantically powerful than the
last, and each only sees the survivors of the previous one:

```
SHA-256  →  perceptual hash  →  embedding  →  cluster + leakage
(bytes)     (downsampled px)    (full sem.)   (graph + labels)
```

**Tradeoff:** running the cheap stages first means the embedding model — the
expensive part — processes the fewest possible images. The cost is that an exact
or near-dup pair is resolved by a weaker signal first; that's fine because those
signals are *more* certain about what they catch (byte-identity is certain;
pixel-hash identity is nearly so).

## 2. The scale-adaptive index — the centerpiece

The headline idea: **architecture should change shape with dataset size.**

- **< 1M vectors → `IndexFlatIP` (exact).** If it fits in RAM, there is no reason
  to accept approximation error. Perfect recall, no parameters, no training.
- **1M–10M → `IndexIVFFlat`.** Partition into Voronoi cells; search only the
  `nprobe` nearest. Full vectors are retained, so the *only* error is cell
  pruning, tunable back toward exact via `nprobe`.
- **10M+ or won't fit RAM → `IndexIVFPQ`.** Product-quantize to a few bytes per
  vector so the index fits at all. Now there are two error sources (cell pruning
  + lossy codes), so we **re-rank**: take a wide approximate shortlist, then
  recompute exact cosine on the memory-mapped originals and keep the true top-k.

**The subtle correctness point:** the gate between Flat and IVFFlat is *RAM, not
count*. IVFFlat also stores full vectors, so if the fp32 matrix won't fit the
budget, IVFFlat won't help — the selector must jump to the compressed IVFPQ. The
selector computes the actual GB footprint and logs it, so the decision is
auditable, e.g.:

```
selected IndexFlatIP: 320,000 x 768d ~= 0.92GB fits in 24GB budget and
  N < 1,000,000 — exact search preferred (zero approximation, perfect recall).
```

**When you'd override:** force IVFPQ to benchmark recall loss vs Flat on your
data; force Flat on a "medium" set if you have the RAM and want zero error.

## 3. Why DINOv2 by default (not CLIP)

DINOv2 is self-supervised with dense objectives, so its features preserve
*spatial structure* — it cares about "same scene / same objects", which is what
detection dedup needs. CLIP's contrastive image-text features encode *global
semantics*: great at "two beach photos", often too loose here (it would merge
genuinely different scenes that are thematically alike). CLIP and SSCD are
included as swappable backbones so the difference is observable, not asserted.

**Tradeoff:** DINOv2 is heavier than a tiny hash and needs a GPU to be fast.
That's why it's the *last* dedup stage — it only runs on what survived the cheap
filters.

## 4. The conservative thresholds (and the small-object caveat)

Detection is uniquely sensitive to over-merging. A perceptual hash downsamples to
8×8 before hashing, discarding exactly the high-frequency detail that small
objects (a distant pedestrian, a sign) live in — so two frames differing *only*
in such an object can hash identically. Merge them and you delete a labelled
object. Hence:

- pHash/dHash Hamming threshold default **≤ 8** (conservative).
- Cosine threshold default **0.92** (only near-identical scenes merge).

**When you'd loosen them:** classification-style dedup, or when you've confirmed
small objects aren't decision-relevant for your task.

## 5. Cluster, don't pair (Union-Find)

Duplicates are transitive: if A≈B and B≈C, then {A,B,C} is one group. Pairwise
deletion double-counts and can delete the wrong survivor. We build a graph from
all flagged pairs (Stage 2 ∪ Stage 3) and take connected components via
Union-Find (path compression + union by rank, iterative to survive long chains
of sequential frames). Stage 4 then re-resolves authoritatively, discarding the
provisional greedy keepers stages 2/3 used just to run standalone.

## 6. Annotation-aware resolution

For detection, the "best" duplicate is the one with the richest supervision, not
an arbitrary survivor. Default `keep_most_annotated` (most boxes, then class
diversity). Alternatives: `keep_highest_res`, `keep_central` (closest to the
embedding centroid), and `weighted_sample` (delete nothing; emit `1/cluster_size`
weights so redundancy is corrected at train time). When near-duplicates carry
*conflicting* class labels, we refuse to auto-resolve and flag for review — auto-
picking one would bake in a labelling error.

## 7. Leakage — the headline metric

A near-duplicate of a test image in the training set means the model memorised
the test sample; reported mAP is inflated. We reuse the *same* pair signal the
dedup stages produced: a test image is "leaked" if it shares a flagged pair with
any training image. No new threshold, no new model. The optional train-twice
harness then demonstrates the consequence directly: train on raw vs deduped,
compare mAP on the same test set, and watch the raw number deflate.

**Hard partition key:** if images carry source metadata (video id, capture
session), frames from one source must never straddle the split — same scene on
both sides is leakage by construction even when feature similarity didn't trip.

## 8. Throughput: the bottleneck is JPEG decode, not the GPU

For a ViT forward pass, CPU-side JPEG decode routinely starves the GPU. The
loader is built decode-first (many workers), logs achieved img/sec + GPU memory
so the bottleneck is *visible*, and offers an optional pre-resize-to-disk pass
and a DALI GPU-decode hook for when CPUs still can't keep up.

## 9. Resumability & storage

Every stage checkpoints to disk so a crash loses ≤ one batch: Stage 1's LMDB map
is durable; Stage 3 writes a memory-mapped embedding array plus a per-batch
`done` mask. Embeddings are normalized fp16 by default (half the footprint;
negligible cosine error, and still far more precise than the PQ codes they
re-rank against).

## 10. Environment note (Blackwell GPU)

The RTX 5070 Ti is `sm_120` and needs CUDA 12.8 PyTorch (`cu128`), installed via
pip since conda-forge doesn't reliably carry it. `torchvision` must come from the
same cu128 index or `transformers`' image processor fails to import. FAISS is
CPU-only here: GPU FAISS wheels don't reliably target sm_120, and at our scale
index search isn't the bottleneck — embedding extraction (on the GPU) is. The
scale-tier *logic* is identical on CPU FAISS; only wall-clock differs.

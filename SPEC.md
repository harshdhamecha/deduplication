# Claude Code Prompt — Adaptive Image Deduplication Pipeline for Object Detection

> Paste everything below the line into Claude Code. It is written as a build spec.
> The goal is a learning-maximizing, demo-able, measurable project — not the
> shortest path to "it runs."

---

## 0. Project Intent (read this first)

Build a **scale-adaptive image deduplication pipeline specifically for object
detection datasets**. The headline idea — and the thing I want the code to make
visually obvious — is that **the right architecture changes shape with dataset
size**, and that we should never trade accuracy for speed we don't need.

Concretely:
- On a small dataset (<1M), the pipeline must do **exact** nearest-neighbor search
  (zero approximation error) because it fits in memory and there's no reason to
  give up recall.
- On a medium dataset (~1–10M), it should switch to a partitioned-but-uncompressed
  index (recall stays high, speed improves).
- On a large dataset (10M+), it should switch to a compressed, disk-backed index
  (the only thing that fits, accepting bounded approximation).

I am building this as a **resume/portfolio project**, so optimize for: (1) maximum
learning across the design space, (2) easy to demo on a laptop/single GPU, and
(3) easy to *measure impact* with a defensible number. Treat me as someone who
wants to understand every tradeoff, not just get a working script. Comment the
*why* behind design choices, not just the *what*.

Primary hardware target: a single consumer GPU machine (e.g. RTX-class, 16GB VRAM)
plus 32–64GB system RAM. Must also be runnable CPU-only for small datasets.

---

## 1. Core Architecture — A Four-Stage Cascade

Implement the pipeline as a cascade of independent, individually-runnable stages.
Each stage consumes the survivors of the previous stage. Every stage must be
toggleable and independently benchmarkable.

### Stage 1 — Exact Duplicate Detection
- Compute SHA-256 over raw file bytes.
- Group byte-identical files.
- **Storage abstraction:** provide two backends behind one interface — an
  in-memory dict (default, for small data) and a disk-backed key-value store
  (LMDB or RocksDB) for large data. Auto-select by an estimated-item-count
  threshold, but allow manual override.

### Stage 2 — Perceptual / Near-Duplicate Detection
- Implement **multiple hash algorithms** as selectable options, each behind a
  common interface: `phash`, `dhash`, `ahash`, `whash` (wavelet). Default: pHash
  **and** dHash combined (catch compression variants + spatial shifts).
- For candidate retrieval, implement **three** search strategies and let the user
  pick or auto-select by scale:
  1. **Brute-force Hamming** (exact, for small N) — all-pairs or query-vs-set.
  2. **Multi-index hashing** (split the 64-bit hash into chunks, build inverted
     indices per chunk, use the pigeonhole principle to prune) — for medium/large N.
  3. **BK-tree** — as an alternative metric-tree approach, for comparison/learning.
- Expose the Hamming-distance threshold as config. Default conservative for
  detection (≤ 8), with a documented note on *why* detection wants conservatism
  (small objects / box-position sensitivity).

### Stage 3 — Embedding-Based Semantic Deduplication
- **Embedding model as a swappable option.** Implement a common extractor
  interface with at least:
  - **DINOv2 (ViT-B/14)** — default, document *why* (self-supervised, spatial-
    structure sensitivity suits detection).
  - **CLIP (ViT-B or L)** — for comparison (global semantic similarity).
  - **SSCD** (Meta's copy-detection model) — for true copy detection vs semantic.
  - Make adding a new backbone a ~10-line change.
- **ANN index as a scale-adaptive option** — this is the centerpiece. Implement
  all three FAISS index types behind one interface, with an auto-selector:
  1. **`IndexFlatIP`** (exact, cosine via normalized vectors) — small N, zero
     approximation, perfect recall.
  2. **`IndexIVFFlat`** — medium N, partitioned but full vectors retained.
     Expose `nlist` and `nprobe`.
  3. **`IndexIVFPQ`** — large N, product-quantized + disk-backed. Expose `nlist`,
     `nprobe`, `m` (subquantizers), `nbits`. Train on a random sample, populate
     in batches, optionally **re-rank top candidates with exact distance** on the
     original memory-mapped vectors to recover precision lost to PQ.
- The auto-selector picks the index from estimated vector count and available RAM,
  but the choice must be **overridable and logged** ("selected IndexFlatIP: 320K
  vectors × 768 dims fit in 0.9GB, exact search preferred").
- Store embeddings as **memory-mapped numpy arrays** (fp16 option to halve size).
- Expose cosine-similarity threshold as config (default ~0.92 for detection) and
  provide a utility to sweep thresholds and dump cluster visualizations.

### Stage 4 — Cluster Resolution + Cross-Split Leakage Check
- **Cluster, don't pair.** Build a similarity graph from flagged pairs across
  Stages 2 and 3, then find **connected components via Union-Find**. Handle
  transitive duplicates (A≈B, B≈C ⇒ {A,B,C} one cluster).
- **Annotation-aware resolution** (object-detection specific). For each cluster,
  implement selectable keep-strategies:
  - `keep_most_annotated` (default) — keep the image with richest supervision
    (most boxes / most diverse classes).
  - `keep_highest_res`.
  - `keep_central` — embedding closest to cluster centroid.
  - `weighted_sample` — don't delete; emit per-image sampling weights (1/cluster_size).
  - When near-duplicates have **conflicting annotations** for the same scene,
    flag for review rather than auto-resolving.
- **Cross-split leakage check.** Critical for the resume story. After dedup,
  before splitting: detect any val/test image with a training-set neighbor above
  threshold and report it. If source metadata exists (video ID, URL, capture
  session), support a **hard partition key** so same-scene frames never split
  across train/val/test.

---

## 2. Object-Detection-Specific Requirements

- Parse annotations in **COCO JSON** format (primary) with a pluggable parser
  interface so YOLO/Pascal-VOC can be added later.
- Annotation-aware resolution must actually read box counts / class diversity.
- Add an optional **video-frame subsampling** pre-filter (keep every k-th frame
  when sequence metadata is present) so we don't burn embedding compute on
  near-identical consecutive frames.
- Document the small-object caveat (aggressive pHash at 32×32 can discard images
  that differ only in detection-relevant small objects) directly in the Stage 2
  code comments.

---

## 3. Measurability — The Resume Payload

The project's value is the *number* it produces. Build a measurement harness:

- **Duplicate-distribution profiler** (run BEFORE removing anything): report what
  fraction of the dataset is exact / near-dup / semantic-dup, with a histogram.
  This profiling-first discipline should be a first-class feature.
- **Leakage report:** quantify train/test contamination — e.g. "X% of test images
  have a near-duplicate in train." This is the headline metric.
- **Optional train-twice harness (stretch goal, keep it lightweight):** a thin
  wrapper to train a small detector (e.g. a compact torchvision detection model)
  on raw vs deduped data and report the mAP delta — to demonstrate that fixing
  leakage corrects an *overstated* metric. Keep this optional and fast; the point
  is the methodology, not SOTA training.
- Every stage logs: items in, items removed, time elapsed, index type chosen,
  peak memory. Emit a final JSON + a human-readable summary.

---

## 4. Performance & Throughput (don't get this wrong)

- The embedding stage's real bottleneck is usually **CPU-side JPEG decode**, not
  the GPU. Build the data loader with this in mind: multi-worker `DataLoader`,
  optional **NVIDIA DALI** path for GPU decode, and an optional **pre-resize-to-disk**
  step. Include a small benchmark that prints achieved img/sec and GPU utilization
  so the bottleneck is visible.
- Batched, streaming processing throughout — never assume the dataset fits in RAM
  except where the scale-tier logic has explicitly decided it does.
- Make the pipeline **resumable/checkpointed** (so a spot-instance preemption or a
  crash doesn't lose hours). Each stage writes intermediate state to disk.

---

## 5. Engineering / UX Requirements

- **Config-driven** (YAML or similar) with sane defaults; every threshold, model,
  index type, and strategy overridable from config and CLI.
- **CLI** with per-stage subcommands plus a `run-all`. Easy to demo:
  `dedup profile`, `dedup run --stage 1..4`, `dedup report`.
- Clean module layout: `hashing/`, `embeddings/`, `indexing/`, `clustering/`,
  `resolution/`, `leakage/`, `profiling/`, `io/`, `cli.py`.
- Type hints, docstrings, and **inline comments explaining the tradeoff** at every
  decision point (this is a learning project — the comments are part of the deliverable).
- Unit tests for the tricky logic: Union-Find correctness, multi-index hashing
  recall vs brute force (assert they agree on a small set), PQ re-ranking, COCO parsing.
- A **README** that includes: the scale-tier decision table (which index at which
  size and why), a quickstart on a small public sample, and a short "design
  decisions & tradeoffs" section I can adapt into a blog post.
- Include a tiny bundled toy dataset (or a script to download a small public
  detection sample like a COCO val subset) so the whole thing demos end-to-end on
  a laptop in minutes.

---

## 6. Build Order (do it incrementally, confirm as you go)

1. Project scaffold, config system, CLI skeleton, toy-dataset loader, COCO parser.
2. Stage 1 (SHA-256) + storage-backend abstraction + profiler.
3. Stage 2 (hashes + the three search strategies) + recall-agreement tests.
4. Stage 3 (DINOv2 default + the three FAISS index types + auto-selector + memmap).
5. Stage 4 (Union-Find clustering + annotation-aware resolution + leakage check).
6. Measurement harness (profiler report, leakage report, optional train-twice).
7. README, design-decisions doc, tests, end-to-end demo on the toy dataset.

Pause after the scaffold (step 1) and after step 4 to show me the structure and
the index-selector logic before continuing — I want to understand those parts
deeply, not just receive them.

---

## 7. What I Care About Most

Explain tradeoffs as you build. When you pick a default, tell me what you're
trading away and when I'd choose differently. The adaptive index selection
(Flat → IVF-Flat → IVF-PQ) and the cross-split leakage check are the two pieces
I most want to be correct and well-explained, because they're what make this a
senior-level project rather than a library wrapper.
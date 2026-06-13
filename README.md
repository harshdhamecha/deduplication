32# Scale-Adaptive Image Deduplication for Object Detection

A four-stage cascade that deduplicates object-detection image datasets and
**quantifies train/test leakage**. The thesis the code is built to demonstrate:
**the right index architecture changes shape with dataset size** — use exact
search when the data fits in memory, partition when it doesn't, and compress only
when forced. We never trade accuracy for speed we don't need.

Two pieces carry the project:
1. **The scale-adaptive ANN index selector** (`Flat → IVF-Flat → IVF-PQ`) — picks
   the cheapest index that still meets recall needs, from vector count + RAM, and
   logs the decision verbatim.
2. **The cross-split leakage check** — "X% of test images have a near-duplicate in
   train", the headline metric, because leakage silently inflates reported mAP.

## The cascade

| Stage | Catches | Method | Key idea |
|------:|---------|--------|----------|
| 1 | exact duplicates | SHA-256 over raw bytes | byte-identical ⇒ zero-risk drop; dict or LMDB backend by scale |
| 2 | near-duplicates | pHash + dHash + Hamming search | 3 search strategies (brute / multi-index / BK-tree), all exact, agree by test |
| 3 | semantic duplicates | DINOv2 embeddings + FAISS | **scale-adaptive index** + exact re-rank for PQ |
| 4 | resolution + leakage | Union-Find clustering | annotation-aware keep + cross-split leakage check |

## Scale-tier decision table

The selector (`dedup/indexing/selector.py`) chooses from estimated vector count
**and** available RAM, and the choice is overridable + logged:

| Vectors | Fits RAM? | Index | Approximation | Why |
|--------:|:---------:|-------|---------------|-----|
| < 1M | yes | `IndexFlatIP` | **none** (exact) | fits in memory — no reason to give up recall |
| 1M–10M | yes | `IndexIVFFlat` | cell-pruning only | full vectors kept; partition for speed (`nprobe` tunes recall) |
| 10M+ *or* won't fit RAM | — | `IndexIVFPQ` | PQ + cell-pruning | only compressed form fits; **exact re-rank** recovers precision |

The RAM gate, not the count, is the real boundary: `IndexIVFFlat` also stores
full vectors, so if the fp32 matrix won't fit the budget, the selector skips
straight to the compressed `IndexIVFPQ`.

## Setup

```bash
conda env create -f environment.yml
conda activate dedup

# Blackwell GPU (RTX 5070 Ti, sm_120) needs the CUDA 12.8 PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[embeddings]"

# IMPORTANT: torchvision must match the cu128 torch, or you'll hit
# "operator torchvision::nms does not exist" (and transformers' AutoImageProcessor
# will fail to import). Force the matching cu128 build:
pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cu128
```

Stages 1–2 and all index tests run CPU-only with just the core install; torch is
only needed from Stage 3 onward.

## Quickstart (end-to-end on a real COCO sample)

```bash
dedup fetch-data --num-images 120          # download a small COCO val2017 slice
dedup --config configs/default.yaml profile        # measure duplicates FIRST
dedup --config configs/default.yaml run-all        # run the full cascade
dedup --config configs/default.yaml report         # JSON + human-readable summary
```

Override any setting from the CLI (no file edit needed):

```bash
# Force a tier to see the selector log a different decision:
dedup --set stage3.index_type=ivfpq run --stage 3
# More conservative perceptual matching; CLIP instead of DINOv2:
dedup --set stage2.hamming_threshold=4 --set stage3.backbone=clip run-all
```

Use the locally-downloaded DINOv2-Large (1024-d, fully offline) instead of the
default `facebook/dinov2-base` (768-d, one-time download):

```bash
dedup --set stage3.backbone=dinov2_large --set stage3.model_path=models/dinov2_large run --stage 3
```

### See the leakage metric on real images

The raw COCO val slice has no duplicates (headline reads 0% — honest but dull).
To exercise the story on real imagery, plant a realistic train/val split with
recompressed+cropped near-duplicate leakage:

```bash
dedup make-leakage-demo                        # builds data/demo/ (train+val, planted leaks)
dedup --config configs/demo.yaml run-all
dedup --config configs/demo.yaml report
```

Example result (15 leaks planted into 45 val images):

```
 Removed        : 15  (11.11% redundant)
 Train/test leakage (HEADLINE):
   val        33.33% have a near-duplicate in train
```

### Measure recall with planted duplicates

`run-all` reports *how many* files it removed, but on raw data that count is
unfalsifiable — with no known duplicates you can't say what fraction the cascade
actually *caught*. `dedup plant` fixes that: it builds a self-contained dataset
(source images copied in + planted duplicates) and writes a **ground-truth
manifest**, so removals can be scored for recall/precision. It is non-destructive
— the original `data/images/` is left untouched.

It plants one tier per detection stage, each defeating the previous stage's
mechanism so all three paths get exercised:

| Tier | Transform | Caught by |
|------|-----------|-----------|
| `exact` | byte-identical copy | Stage 1 (SHA-256) |
| `near` | 2 px border crop + JPEG q35 | Stage 2 (pHash/dHash) |
| `semantic` | center-zoom + rotate + colour jitter | Stage 3 (embeddings) |

```bash
dedup fetch-data --num-images 120              # need some source images first
dedup plant --exact 10 --near 10 --semantic 10 # -> data/planted/ + plant_manifest.json
dedup --set io.image_root=data/planted/images run-all   # point a run at the planted set
dedup --set io.image_root=data/planted/images score     # grade the run vs the manifest
```

`plant` output:

```
planted dataset at data/planted/
  source images : 120 (copied into images/)
  exact   (S1)  : 10
  near    (S2)  : 10
  semantic(S3)  : 10
  manifest      : data/planted/plant_manifest.json
  -> score a run's removals against plant_manifest.json for recall/precision
```

The manifest maps each planted file to its `original`, `type`, and the
`expected_stage` that should flag it. The plant is reproducible: the seed comes
from config (`--set seed=...`). By default `--source` is taken from
`io.image_root`; override it to plant from any image directory.

```bash
# Smaller plant from a custom source, into a custom output dir:
dedup --set seed=7 plant --exact 5 --near 5 --semantic 0 \
      --source data/images/val2017 --out data/planted_small
```

**`dedup score`** does the grading for you — it joins the manifest's known
positives against the run's on-disk artifacts and reports the dedup *detector's*
own recall and precision (this is **not** mAP — that's `train-twice` — and not
the leakage headline). The key column is **on-time**: whether each tier died at
its *expected* stage, which is the test that the cascade design works — cheap
stages catch what they can so the embedding stage only sees what genuinely needs
it.

```bash
dedup --set io.output_dir=runs/latest score --manifest data/planted/plant_manifest.json
```

```
============================================================
 PLANT SCORE (dedup detector recall / precision)
============================================================
 planted  : 30  (seed 42)

 tier           planted  caught  on-time   recall
   exact   (S1)      10      10       10     1.00
   near    (S2)      10      10       10     1.00
   semantic(S3)      10       9        9     0.90
 ------------------------------------------------
 overall recall : 29/30  (0.967)
 precision      : 29/29  (1.000)   0 false removal(s)
```

* **recall** = planted dups the cascade caught / planted (a missed `semantic`
  here means a pair fell below `stage3.cosine_threshold`).
* **precision** = removals that were a real planted dup / total removals. A
  removed file that is neither a planted copy nor an original is a false merge —
  worth investigating (or a coincidental dup already in the source).

The full breakdown (per-tier misses, false-positive examples) is written to
`<output_dir>/score.json`.

### Optional: train-twice (leakage impact on a metric)

```bash
dedup train-twice --epochs 2 --max-images 200
```

Trains a compact detector on raw vs deduped data and reports the mAP delta — a
methodology demo (low absolute mAP on a tiny sample is expected; the *comparison*
is the point).

## Layout

```
dedup/
  config.py   cli.py
  io/         COCO parser (pluggable) · image enumeration · inter-stage state · video pre-filter
  hashing/    Stage 1 SHA-256 + storage backends · Stage 2 perceptual hashes + 3 search strategies
  embeddings/ Stage 3 extractor registry (DINOv2/CLIP/SSCD) · throughput loader · memmap store
  indexing/   VectorIndex interface · Flat / IVFFlat / IVFPQ · the auto-selector
  clustering/ Union-Find · cross-stage cluster builder
  resolution/ keep-strategies · Stage 4 orchestrator
  leakage/    cross-split leakage check + hard partition key
  profiling/  duplicate profiler · per-stage metrics · report renderer · train-twice · plant scorer
configs/default.yaml   scripts/fetch_coco_subset.py · plant_duplicates.py · make_leakage_demo.py   tests/
```

See [docs/design-decisions.md](docs/design-decisions.md) for the tradeoff
write-up.

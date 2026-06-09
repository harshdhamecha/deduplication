# Deduplication Pipeline — Project Context

## What This Is
Scale-adaptive image deduplication pipeline for object detection datasets.
Portfolio/resume project — optimize for learning depth and measurable impact,
not shortest path to working code.

## Hardware
- GPU: RTX 5070 Ti (16GB VRAM)
- System RAM: [your actual RAM]
- Storage: 1 TB SSD
- Must also run CPU-only for small datasets

## Module Layout
hashing/ | embeddings/ | indexing/ | clustering/ | resolution/ | leakage/ |
profiling/ | io/ | cli.py
Do not reorganize this structure without discussion.

## The Architectural Centerpiece
The FAISS index auto-selector (Flat → IVF-Flat → IVF-PQ) is the core design
idea. Every scale-tier decision must be logged — when the selector fires, print
which index was chosen and why (vector count, available RAM estimate, threshold).

## Scale Tiers (do not collapse these into one path)
- <1M vectors  → IndexFlatIP (exact, zero approximation)
- 1–10M        → IndexIVFFlat (partitioned, full vectors retained)
- 10M+         → IndexIVFPQ (compressed, disk-backed, re-rank top candidates)
Selector must be overridable by config even when auto-selection would choose
differently.

## Annotation Format
Primary: COCO JSON. Parser lives in io/. Keep it behind an interface so
YOLO/VOC can be added later without touching other modules.

## Dataset / Demo
[To be Added]

# Working Principles

## Evidence Before Claims
Never say "tests pass" without running them. Never say "bug is fixed" without
verifying. Show the output, not the assertion.

## Comment the Why, Not the What
Every non-obvious design decision needs a comment explaining the tradeoff, not
just what the code does. This is a learning project — the comments are part
of the deliverable.

## Tradeoff Visibility
When picking a default (threshold, model, index type), the code or its comment
must state: what is being traded away, and when someone would choose differently.

## No Silent Failures
If a stage produces zero removals on a real dataset, that is suspicious and
should warn loudly — not silently succeed.

## Resumability
Every stage must checkpoint progress to disk. A crash or preemption should
never lose more than one batch of work.

## Bottleneck Awareness
The embedding stage's real bottleneck is usually CPU-side JPEG decode, not
the GPU. Profile and log achieved img/sec and GPU utilization at the start
of every Stage 3 run.
"""Stage 3 embedding extraction lives here.

Planned contents (filled in Step 4):
  extractor.py  Extractor interface + registry. Adding a backbone is a ~10-line
                class + one decorator. Backbones: DINOv2 ViT-B/14 (default),
                CLIP, SSCD, plus the locally-downloaded DINOv2-Large.
  loader.py     A throughput-aware DataLoader. The real bottleneck is CPU-side
                JPEG decode, not the GPU — so multi-worker decode, an optional
                NVIDIA DALI GPU-decode path, and an optional pre-resize-to-disk
                pass, with a benchmark that prints achieved img/sec + GPU util.
  store.py      Memory-mapped numpy embedding store (fp16 option to halve size).
"""

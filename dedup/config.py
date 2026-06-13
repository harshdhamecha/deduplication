"""Configuration system for the deduplication pipeline.

Design goals (this is the single source of truth for every tunable):
  * Sane defaults baked into dataclasses, so the pipeline runs with zero config.
  * One YAML file can override any subset (deep-merged onto defaults).
  * The CLI can override any leaf via dotted keys (e.g. ``stage2.hamming_threshold=6``)
    so a demo never requires editing a file.

WHY dataclasses instead of a dict or pydantic: dataclasses give us typed,
discoverable fields with IDE autocomplete and zero runtime dependency, and the
defaults double as documentation. Every non-obvious default below carries a
comment stating what it trades away and when you'd change it — per the project's
"comment the why" rule.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Stage 1 — Exact duplicate detection
# --------------------------------------------------------------------------- #
@dataclass
class Stage1Config:
    enabled: bool = True
    # "auto" picks in-memory dict below the threshold, LMDB above it. We expose
    # the threshold because the right cutover depends on machine RAM, not a law.
    storage_backend: str = "auto"  # auto | memory | lmdb
    # Above this many items, byte-hash bookkeeping is pushed to disk (LMDB) so we
    # don't hold a dict of N hashes in RAM. 5M SHA-256 hex strings ~= a few GB.
    lmdb_threshold: int = 5_000_000


# --------------------------------------------------------------------------- #
# Stage 2 — Perceptual / near-duplicate detection
# --------------------------------------------------------------------------- #
@dataclass
class Stage2Config:
    enabled: bool = True
    # Default = pHash + dHash combined. pHash (DCT-based) is robust to JPEG
    # recompression; dHash (gradient-based) is robust to small spatial shifts.
    # Together they catch the two most common near-dup modes in scraped data.
    hashes: list[str] = field(default_factory=lambda: ["phash", "dhash"])
    hash_size: int = 8  # 8x8 -> 64-bit hash; the multi-index logic assumes 64 bits.
    # Conservative for DETECTION. A large threshold merges images that differ in
    # exactly the small objects detection cares about (a distant pedestrian, a
    # sign). 8 bits / 64 is a ~12.5% tolerance. Raise it for classification-style
    # dedup where global appearance is all that matters.
    hamming_threshold: int = 8
    # How to combine multiple hashes. "any": flag a pair if EITHER hash says it's
    # near (catches both recompression AND shift modes — the spec's intent, higher
    # recall). "all": require ALL hashes to agree (higher precision / more
    # conservative — fewer wrong merges of small-object-different frames). Default
    # "any" matches "pHash and dHash combined to catch both"; switch to "all" if
    # you find Stage 2 over-merging detection-relevant near-misses.
    combine: str = "any"  # any | all
    # "auto": brute-force for small N (exact, simplest), multi-index hashing for
    # large N (pigeonhole pruning). BK-tree is offered mainly for learning/compare.
    search_strategy: str = "auto"  # auto | bruteforce | multiindex | bktree


# --------------------------------------------------------------------------- #
# Stage 3 — Embedding-based semantic deduplication (the centerpiece)
# --------------------------------------------------------------------------- #
@dataclass
class Stage3Config:
    enabled: bool = True

    # --- backbone ---
    # DINOv2 ViT-B/14 default: self-supervised features that preserve spatial
    # structure, which suits detection better than CLIP's global semantic vector.
    # Swap to "clip" for semantic grouping, "sscd" for true copy-detection, or
    # "dinov2_large" to use the locally-downloaded 1024-d model.
    backbone: str = "dinov2_vitb14"  # dinov2_vitb14 | clip | sscd | dinov2_large
    model_path: str | None = None  # local dir override (e.g. models/dinov2_large)
    device: str = "auto"           # auto | cuda | cpu
    batch_size: int = 64
    num_workers: int = 8           # JPEG decode is the real bottleneck -> many workers
    use_dali: bool = False         # optional GPU decode path (NVIDIA DALI)
    pre_resize: bool = False       # optional resize-to-disk pre-pass to cut decode cost
    fp16_embeddings: bool = True   # store memmap as fp16 to halve disk/RAM footprint

    # --- ANN index (scale-adaptive) ---
    # "auto" runs the selector in dedup/indexing: picks Flat / IVFFlat / IVFPQ
    # from estimated vector count + available RAM, logs the decision, and can be
    # overridden by setting this to a concrete type.
    index_type: str = "auto"  # auto | flat | ivfflat | ivfpq
    # IVF params (used by ivfflat/ivfpq). nlist = #partitions; nprobe = #partitions
    # searched at query time (recall/speed knob). Defaults are placeholders; the
    # selector scales nlist with N (~sqrt(N) rule of thumb).
    nlist: int = 4096
    nprobe: int = 16
    # PQ params (ivfpq only). m = #subquantizers (must divide dim); nbits = bits
    # per subquantizer code. m=64,nbits=8 on a 768-d vector -> 64 bytes/vector,
    # a 24x compression vs fp32 — the price is approximation we claw back below.
    pq_m: int = 64
    pq_nbits: int = 8
    # Re-rank: after the approximate index returns top candidates, recompute exact
    # cosine on the original memmapped vectors to recover precision lost to PQ.
    rerank: bool = True
    rerank_k: int = 100
    # RAM the selector may assume is available for an in-memory index, in GB.
    ram_budget_gb: float = 24.0

    # --- similarity threshold ---
    # 0.92 cosine is conservative for detection: high enough that we only merge
    # genuinely near-identical scenes, not merely semantically similar ones.
    # Lower it to dedup more aggressively (risking loss of hard/rare examples).
    cosine_threshold: float = 0.92


# --------------------------------------------------------------------------- #
# Stage 4 — Cluster resolution + cross-split leakage
# --------------------------------------------------------------------------- #
@dataclass
class Stage4Config:
    enabled: bool = True
    # Detection-aware default: within a duplicate cluster, keep the image with the
    # richest supervision (most boxes / most class diversity) rather than an
    # arbitrary survivor. Alternatives: highest_res, central, weighted_sample.
    keep_strategy: str = "keep_most_annotated"  # + keep_highest_res|keep_central|weighted_sample
    # If set, frames sharing this metadata key (e.g. video_id) are never split
    # across train/val/test — the strongest defense against leakage.
    partition_key: str | None = None
    # Keep every k-th frame when sequence metadata is present (0/1 disables).
    video_subsample_k: int = 0


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #
@dataclass
class LeakageConfig:
    enabled: bool = True
    # Maps split name -> how to recognise membership. Filled in per-dataset; the
    # leakage stage reports any val/test image with a train neighbour above the
    # Stage 3 cosine threshold (the headline metric).
    train_split: str = "train"
    eval_splits: list[str] = field(default_factory=lambda: ["val", "test"])


# --------------------------------------------------------------------------- #
# IO / dataset
# --------------------------------------------------------------------------- #
@dataclass
class IOConfig:
    image_root: str = "data/images"
    annotations: str | None = None  # path to COCO JSON, if available
    annotation_format: str = "coco"  # coco | yolo | voc (only coco implemented now)
    output_dir: str = "runs/latest"  # checkpoints, embeddings memmap, reports


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    seed: int = 42
    log_level: str = "INFO"
    io: IOConfig = field(default_factory=IOConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)

    # ----------------------------- loading ----------------------------- #
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        """Build a Config from defaults, then a YAML file, then CLI overrides.

        Precedence (lowest to highest): dataclass defaults < YAML < CLI overrides.
        This ordering is deliberate: a demo can ship a YAML and still be tweaked
        from the command line without editing the file.
        """
        cfg = cls()
        if path is not None:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh) or {}
            cfg = _merge_into(cfg, data)
        if overrides:
            cfg = _apply_overrides(cfg, overrides)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Helpers: deep-merge a dict onto a dataclass, and apply dotted overrides.
# --------------------------------------------------------------------------- #
def _merge_into(obj: Any, data: dict[str, Any]) -> Any:
    """Recursively overlay ``data`` onto dataclass ``obj``, returning a new obj.

    Unknown keys raise — a typo in a config file should fail loudly, not be
    silently ignored (a debugging nightmare on a long run).
    """
    if not is_dataclass(obj):
        return data
    valid = {f.name: f for f in fields(obj)}
    updates: dict[str, Any] = {}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' (valid: {sorted(valid)})")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_into(current, value)
        else:
            updates[key] = value
    return dataclasses.replace(obj, **updates)


def _apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` strings (from the CLI) onto the config."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be of the form key.path=value")
        dotted, raw = ov.split("=", 1)
        cfg = _set_dotted(cfg, dotted.split("."), _coerce(raw))
    return cfg


def _set_dotted(obj: Any, path: list[str], value: Any) -> Any:
    head, *rest = path
    valid = {f.name for f in fields(obj)}
    if head not in valid:
        raise KeyError(f"Unknown config key '{head}' (valid: {sorted(valid)})")
    if rest:
        return dataclasses.replace(obj, **{head: _set_dotted(getattr(obj, head), rest, value)})
    return dataclasses.replace(obj, **{head: value})


def _coerce(raw: str) -> Any:
    """Best-effort scalar coercion for CLI override values (YAML scalar rules)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw

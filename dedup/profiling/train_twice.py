"""(Stretch goal) Train-twice harness: show that fixing leakage corrects an
OVERSTATED metric.

Methodology (the point — not SOTA numbers):
  1. RAW run     — train a small detector on the training split AS-IS (leaked
                   near-duplicates of test images still present), evaluate on test.
  2. DEDUPED run — train on the same split MINUS the images Stage 4 removed/flagged
                   as leaked, evaluate on the SAME test set.
  3. Report mAP_raw vs mAP_deduped. If raw > deduped, the raw number was inflated
     by test-set memorisation — exactly the harm the leakage check warns about.

Kept intentionally lightweight: a compact torchvision detector, few epochs,
capped image count. This is a demonstration of the methodology you'd run at scale,
not a benchmark. Everything here is optional and imported lazily so the core
pipeline never depends on a training stack.
"""

from __future__ import annotations

import json
from pathlib import Path

from dedup import get_logger
from dedup.config import Config
from dedup.leakage.check import infer_split_from_path

logger = get_logger()


def _split_paths(cfg: Config):
    """Partition the dataset's images into train/test by path inference."""
    from dedup.io.images import enumerate_images

    known = [cfg.leakage.train_split] + cfg.leakage.eval_splits
    train, test = [], []
    for p in enumerate_images(cfg.io.image_root):
        s = infer_split_from_path(str(p), known)
        if s == cfg.leakage.train_split:
            train.append(str(p))
        elif s in cfg.leakage.eval_splits:
            test.append(str(p))
    return train, test


def _removed_ids(cfg: Config) -> set[str]:
    """Images Stage 4 dropped (duplicates) + any flagged as leaked — the
    difference between the RAW and DEDUPED training sets."""
    out = Path(cfg.io.output_dir)
    removed: set[str] = set()
    rm_file = out / "stage4" / "removed.txt"
    if rm_file.exists():
        removed |= {x for x in rm_file.read_text().splitlines() if x}
    leak_file = out / "stage4" / "leakage.json"
    if leak_file.exists():
        rep = json.loads(leak_file.read_text())
        for split in rep.get("by_split", {}).values():
            removed |= set(split.get("examples", []))
    return removed


def run_train_twice(cfg: Config, epochs: int = 2, max_images: int = 200) -> dict:
    """Run both trainings and report the mAP delta. Returns the result dict."""
    if not cfg.io.annotations or not Path(cfg.io.annotations).exists():
        logger.error("train-twice needs COCO annotations (io.annotations). Aborting.")
        return {"error": "no annotations"}

    train_paths, test_paths = _split_paths(cfg)
    if not train_paths or not test_paths:
        logger.error("train-twice needs both a train and a test/val split inferable "
                     "from image paths (looked for %s). Found %d train, %d test. "
                     "Provide split-labelled directories (e.g. images/train, images/val).",
                     [cfg.leakage.train_split] + cfg.leakage.eval_splits,
                     len(train_paths), len(test_paths))
        return {"error": "missing split"}

    removed = _removed_ids(cfg)
    raw_train = train_paths[:max_images]
    deduped_train = [p for p in raw_train if p not in removed]
    logger.info("train-twice: raw train=%d, deduped train=%d (-%d), test=%d",
                len(raw_train), len(deduped_train), len(raw_train) - len(deduped_train),
                len(test_paths))

    # Lazy import the training stack so the rest of the project never needs it.
    from dedup.profiling._detector import evaluate_map, train_detector

    model_raw = train_detector(raw_train, cfg, epochs)
    map_raw = evaluate_map(model_raw, test_paths[:max_images], cfg)

    model_dd = train_detector(deduped_train, cfg, epochs)
    map_dd = evaluate_map(model_dd, test_paths[:max_images], cfg)

    result = {
        "map_raw": round(map_raw, 4),
        "map_deduped": round(map_dd, 4),
        "delta": round(map_raw - map_dd, 4),
        "n_train_raw": len(raw_train),
        "n_train_deduped": len(deduped_train),
        "n_test": len(test_paths[:max_images]),
        "epochs": epochs,
    }
    (Path(cfg.io.output_dir) / "train_twice.json").write_text(json.dumps(result, indent=2))

    print("\n=== Train-Twice (leakage impact on metric) ===")
    print(f"  mAP trained on RAW     : {result['map_raw']:.4f}")
    print(f"  mAP trained on DEDUPED : {result['map_deduped']:.4f}")
    print(f"  delta (raw - deduped)  : {result['delta']:+.4f}")
    if result["delta"] > 0:
        print("  -> RAW mAP was OVERSTATED: leaked test duplicates inflated it.")
    else:
        print("  -> no inflation detected at this scale (expected on tiny demos).")
    return result

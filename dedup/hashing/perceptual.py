"""Perceptual hash extraction for Stage 2.

Four algorithms behind one interface, all producing a fixed-width integer code:

  phash  DCT-based.   Robust to JPEG recompression & brightness — the workhorse.
  dhash  gradient.    Robust to small spatial shifts / crops.
  ahash  average.     Cheapest, weakest; good for a sanity baseline.
  whash  wavelet.     Multi-resolution; catches some scaling/blur cases.

Default is phash + dhash together: pHash covers the "same image, recompressed"
case, dHash covers the "same scene, nudged a few pixels" case — the two dominant
near-duplicate modes in scraped data.

>>> SMALL-OBJECT CAVEAT (read before raising the threshold) <<<
A perceptual hash downsamples the image to ~hash_size x hash_size (8x8 -> 64 bits)
before hashing. That throws away exactly the high-frequency detail that small
detection objects live in: a distant pedestrian, a traffic sign, a far-off
vehicle can vanish at 8x8, so two frames that differ ONLY in such an object can
hash identically and be wrongly merged. This is why the Stage 2 threshold default
is conservative (<=8) for detection, and why semantic Stage 3 (full-resolution
features) exists to catch what pixel hashing structurally cannot.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from dedup import get_logger

logger = get_logger()


def _imagehash_to_int(ih) -> int:
    """Pack an imagehash.ImageHash (a boolean grid) into an integer code."""
    val = 0
    for bit in ih.hash.flatten():
        val = (val << 1) | int(bit)
    return val


def _algo_fns():
    """Lazily build the {name: fn} registry — imagehash is only needed here, so
    the rest of the pipeline (and the search tests) import without it."""
    import imagehash  # noqa: PLC0415

    return {
        "phash": lambda img, n: imagehash.phash(img, hash_size=n),
        "dhash": lambda img, n: imagehash.dhash(img, hash_size=n),
        "ahash": lambda img, n: imagehash.average_hash(img, hash_size=n),
        # whash needs a power-of-two hash_size; imagehash enforces that.
        "whash": lambda img, n: imagehash.whash(img, hash_size=n),
    }


def compute_hashes(
    image_paths: Iterable[str | Path],
    algos: list[str],
    hash_size: int = 8,
) -> dict[str, dict[str, int]]:
    """Return ``{algo: {path: int_code}}`` for the requested algorithms.

    Images that fail to decode are skipped with a warning rather than crashing
    the whole run — but we warn (never silently drop) so a systematically broken
    input surfaces instead of quietly shrinking the dataset.
    """
    from PIL import Image  # noqa: PLC0415

    fns = _algo_fns()
    for a in algos:
        if a not in fns:
            raise ValueError(f"Unknown hash algorithm '{a}' (have: {sorted(fns)})")

    out: dict[str, dict[str, int]] = {a: {} for a in algos}
    n_failed = 0
    for path in image_paths:
        path = str(path)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                for a in algos:
                    out[a][path] = _imagehash_to_int(fns[a](img, hash_size))
        except Exception as exc:  # noqa: BLE001 - want to continue past one bad file
            n_failed += 1
            logger.warning("stage2: could not hash %s (%s)", path, exc)
    if n_failed:
        logger.warning("stage2: %d images failed to decode and were skipped", n_failed)
    return out

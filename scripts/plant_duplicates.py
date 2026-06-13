"""Plant labeled duplicates with a ground-truth manifest — make recall measurable.

The raw COCO subset has (almost) no duplicates, so the pipeline's removal count is
honest but unfalsifiable: with no known positives you can't say what fraction the
cascade actually *caught*. This builds a controlled dataset where we know exactly
which files are duplicates and which stage should flag each one, so a run can be
scored: recall = planted-and-removed / planted, precision = planted-removals / all-removals.

Three tiers are planted, one per detection stage, each defeating the prior stage's
mechanism so the tiers don't collapse into one path:

  * exact    — byte-identical copy. Same SHA-256 -> caught by Stage 1.
  * near     — tiny border crop + heavy JPEG recompression. Different bytes (defeats
               Stage 1) but pHash/dHash stay within Hamming threshold -> caught by Stage 2.
  * semantic — center-zoom + colour jitter + small rotation. Pixels shift enough that
               perceptual hashes diverge (defeats Stage 2) but the scene is preserved,
               so embedding cosine stays high -> caught by Stage 3.

NOTE on "semantic": these transforms are chosen to land *beyond* perceptual-hash
tolerance while staying *within* embedding tolerance. That boundary depends on the
backbone and thresholds, so the manifest records intent ("expected_stage"); scoring
against an actual run is what confirms it — we don't assert the catch here.

The output is non-destructive: source images are copied into ``<out>/images`` and
planted dups are added alongside, leaving the original dataset untouched.

Run after `dedup fetch-data`:
    dedup plant --exact 10 --near 10 --semantic 10
Then point a run at it and score against the manifest:
    dedup --set io.image_root=data/planted/images run-all
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

from dedup.io.images import enumerate_images


def _plant_near(img: Image.Image) -> tuple[Image.Image, dict]:
    """Border crop (spatial shift) + heavy recompression — the Stage-2 case."""
    w, h = img.size
    cropped = img.crop((2, 2, w - 2, h - 2))
    return cropped, {"save_kwargs": {"quality": 35}}


def _plant_semantic(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    """Center-zoom + colour jitter + small rotation — the Stage-3 case.

    Each op individually scrambles a perceptual hash (which keys on coarse spatial
    structure) while leaving the scene recognisable to a semantic embedding."""
    w, h = img.size
    # Center-zoom to 85%: changes every hash cell, preserves the subject.
    bx, by = int(w * 0.075), int(h * 0.075)
    out = img.crop((bx, by, w - bx, h - by)).resize((w, h))
    out = out.rotate(rng.uniform(-4, 4), expand=False)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(1.1, 1.25))
    out = ImageEnhance.Color(out).enhance(rng.uniform(1.1, 1.3))
    return out, {"save_kwargs": {"quality": 90}}


def plant(source_root: str = "data/images",
          out_root: str = "data/planted",
          exact: int = 10,
          near: int = 10,
          semantic: int = 10,
          seed: int = 42) -> str:
    """Build a planted dataset under ``out_root`` and write its ground-truth manifest.

    Returns the manifest path. Raises if there is nothing to plant — a planter that
    silently produces zero duplicates would defeat its own purpose (no-silent-failures).
    """
    if exact + near + semantic <= 0:
        raise ValueError("Nothing to plant: --exact/--near/--semantic are all 0.")

    sources = list(enumerate_images(source_root))
    if not sources:
        raise FileNotFoundError(
            f"No source images under {source_root!r}. Run `dedup fetch-data` first.")

    out = Path(out_root)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Copy the originals in so the planted set is self-contained and the dataset
    # the pipeline sees contains both the duplicate and its target.
    for src in sources:
        shutil.copy(src, img_dir / src.name)

    rng = random.Random(seed)  # deterministic: same seed -> same picks (resumability ethos)
    planted: list[dict] = []

    # If a tier asks for more dups than we have sources, sample with replacement and
    # warn — better than silently capping and reporting a smaller-than-requested set.
    def pick(n: int) -> list[Path]:
        if n <= len(sources):
            return rng.sample(sources, n)
        print(f"  ! requested {n} > {len(sources)} sources; sampling with replacement")
        return [rng.choice(sources) for _ in range(n)]

    def emit(src: Path, image: Image.Image, kind: str, stage: int,
             transform: str, save_kwargs: dict) -> None:
        name = f"dup_{kind}_{len(planted):05d}_{src.stem}.jpg"
        image.convert("RGB").save(img_dir / name, **save_kwargs)
        planted.append({"file": f"images/{name}", "original": f"images/{src.name}",
                        "type": kind, "expected_stage": stage, "transform": transform})

    # Stage 1: byte-identical copies (no decode/re-encode — bytes must match exactly).
    for src in pick(exact):
        name = f"dup_exact_{len(planted):05d}_{src.name}"
        shutil.copy(src, img_dir / name)
        planted.append({"file": f"images/{name}", "original": f"images/{src.name}",
                        "type": "exact", "expected_stage": 1, "transform": "byte-copy"})

    for src in pick(near):
        with Image.open(src) as im:
            image, meta = _plant_near(im.convert("RGB"))
        emit(src, image, "near", 2, "crop2px+jpeg35", meta["save_kwargs"])

    for src in pick(semantic):
        with Image.open(src) as im:
            image, meta = _plant_semantic(im.convert("RGB"), rng)
        emit(src, image, "semantic", 3, "zoom85+rotate+jitter", meta["save_kwargs"])

    manifest = {
        "seed": seed,
        "source_root": str(source_root),
        "out_root": str(out),
        "summary": {"exact": exact, "near": near, "semantic": semantic,
                    "total_planted": len(planted), "source_images": len(sources)},
        "planted": planted,
    }
    manifest_path = out / "plant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"planted dataset at {out}/")
    print(f"  source images : {len(sources)} (copied into images/)")
    print(f"  exact   (S1)  : {exact}")
    print(f"  near    (S2)  : {near}")
    print(f"  semantic(S3)  : {semantic}")
    print(f"  manifest      : {manifest_path}")
    print(f"  -> score a run's removals against {manifest_path.name} for recall/precision")
    return str(manifest_path)


if __name__ == "__main__":
    plant()

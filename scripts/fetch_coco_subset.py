"""Download a small COCO val2017 subset for the end-to-end demo.

WHY real COCO (not synthetic) for the demo: the resume story hinges on a
*defensible* leakage number on real-world data, and on real COCO JSON exercising
the actual annotation parser. The trade-off is a one-time ~241 MB annotations
download; we cache it and the images so re-runs are instant.

Strategy (kept resumable — every step skips work already on disk):
  1. Download the official annotations zip (train+val) once, cache it.
  2. Extract instances_val2017.json from the zip.
  3. Deterministically take the first N images (sorted by id).
  4. Download just those N images via their coco_url (avoids the 1 GB image zip).
  5. Write a subset COCO JSON referencing only those images + their annotations.

Run via the CLI (`dedup fetch-data`) or directly (`python -m scripts.fetch_coco_subset`).
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
INSTANCES_MEMBER = "annotations/instances_val2017.json"


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` unless it already exists (resumable-friendly)."""
    if dest.exists():
        print(f"  cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - trusted COCO host
    tmp.rename(dest)  # atomic-ish: a crash mid-download leaves a .part, not a corrupt file


def fetch(num_images: int = 200, out_dir: str = "data") -> Path:
    """Materialise a COCO val2017 subset under ``out_dir``. Returns the subset JSON path."""
    out = Path(out_dir)
    raw_dir = out / "raw"
    images_dir = out / "images" / "val2017"
    ann_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Get the full instances JSON (from cached zip).
    zip_path = raw_dir / "annotations_trainval2017.zip"
    print("[1/4] annotations zip")
    _download(ANNOTATIONS_ZIP_URL, zip_path)

    print("[2/4] extract instances_val2017.json")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(INSTANCES_MEMBER) as fh:
            coco = json.load(fh)

    # 3. Deterministic subset of images.
    print(f"[3/4] selecting first {num_images} images (by id)")
    images = sorted(coco["images"], key=lambda im: im["id"])[:num_images]
    keep_ids = {im["id"] for im in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    used_cats = {a["category_id"] for a in annotations}
    categories = [c for c in coco["categories"] if c["id"] in used_cats]

    # 4. Download just the selected images.
    print(f"[4/4] downloading {len(images)} images")
    for i, im in enumerate(images, 1):
        dest = images_dir / im["file_name"]
        _download(im["coco_url"], dest)
        if i % 25 == 0:
            print(f"    {i}/{len(images)}")

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    subset_path = ann_dir / "instances_val2017_subset.json"
    with open(subset_path, "w") as fh:
        json.dump(subset, fh)

    print(
        f"\nDone: {len(images)} images, {len(annotations)} boxes, "
        f"{len(categories)} categories\n"
        f"  images:      {images_dir}\n"
        f"  annotations: {subset_path}"
    )
    return subset_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download a small COCO val2017 subset.")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    fetch(num_images=args.num_images, out_dir=args.out)

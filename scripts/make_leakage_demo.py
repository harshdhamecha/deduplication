"""Build a realistic train/val split with PLANTED leakage from the COCO subset.

The raw COCO val2017 slice has no duplicates and no splits, so the headline
metric reads 0% — honest, but it doesn't exercise the story. This script
constructs the scenario the pipeline is designed to catch, on real imagery:

  * split the downloaded subset into images/train and images/val,
  * plant leakage: copy a fraction of TRAIN images into VAL as recompressed +
    slightly-cropped near-duplicates (the exact contamination that inflates mAP),
  * write a combined COCO json whose file_names live under train/ or val/ so the
    pipeline infers splits from the path and the leaked val copies inherit a
    sparse annotation (to also exercise keep_most_annotated).

Run after `dedup fetch-data`:
    python -m scripts.make_leakage_demo
Then:
    dedup --config configs/demo.yaml run-all && dedup --config configs/demo.yaml report
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image


def make_demo(subset_json: str = "data/annotations/instances_val2017_subset.json",
              src_images: str = "data/images/val2017",
              out_root: str = "data/demo",
              val_frac: float = 0.25,
              leak_count: int = 15) -> str:
    coco = json.loads(Path(subset_json).read_text())
    images = sorted(coco["images"], key=lambda im: im["id"])
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out = Path(out_root)
    train_dir = out / "images" / "train"
    val_dir = out / "images" / "val"
    for d in (train_dir, val_dir, out / "annotations"):
        d.mkdir(parents=True, exist_ok=True)

    n_val = int(len(images) * val_frac)
    val_imgs, train_imgs = images[:n_val], images[n_val:]

    new_images, new_anns = [], []
    next_img_id = max(im["id"] for im in images) + 1
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    def place(im, split_dir, split_name):
        nonlocal next_ann_id
        src = Path(src_images) / im["file_name"]
        if not src.exists():
            return False
        shutil.copy(src, split_dir / im["file_name"])
        rec = dict(im)
        rec["file_name"] = f"{split_name}/{im['file_name']}"  # split inferable from path
        new_images.append(rec)
        for a in anns_by_img.get(im["id"], []):
            na = dict(a); na["id"] = next_ann_id; next_ann_id += 1
            new_anns.append(na)
        return True

    for im in train_imgs:
        place(im, train_dir, "train")
    for im in val_imgs:
        place(im, val_dir, "val")

    # Plant leakage: recompressed + cropped near-dups of train images, dropped into val.
    planted = 0
    for im in train_imgs[:leak_count]:
        src = Path(src_images) / im["file_name"]
        if not src.exists():
            continue
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            img = img.crop((2, 2, w - 2, h - 2))             # tiny crop (spatial shift)
        name = f"leaked_{im['file_name']}"
        img.save(val_dir / name, quality=35)                 # heavy recompression
        leaked = {"id": next_img_id, "file_name": f"val/{name}",
                  "width": im.get("width"), "height": im.get("height")}
        next_img_id += 1
        new_images.append(leaked)
        # Sparse annotation on the leaked copy: 1 box, so keep_most_annotated would
        # prefer the richly-annotated train original.
        for a in anns_by_img.get(im["id"], [])[:1]:
            na = dict(a); na["id"] = next_ann_id; na["image_id"] = leaked["id"]
            next_ann_id += 1
            new_anns.append(na)
        planted += 1

    demo = {"images": new_images, "annotations": new_anns,
            "categories": coco["categories"]}
    ann_path = out / "annotations" / "instances.json"
    ann_path.write_text(json.dumps(demo))

    print(f"demo built at {out}/")
    print(f"  train images : {len(train_imgs)}")
    print(f"  val images   : {len(val_imgs)} + {planted} planted leaked near-dups")
    print(f"  annotations  : {ann_path}")
    print(f"  -> expect ~{planted}/{len(val_imgs) + planted} val images flagged as leaked")
    return str(ann_path)


if __name__ == "__main__":
    make_demo()

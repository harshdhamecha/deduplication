"""COCO JSON annotation parser.

We parse the JSON directly (stdlib ``json``) rather than going through
``pycocotools.COCO`` here. WHY: pycocotools builds heavy internal indices aimed
at evaluation (IoU matching, mAP) that we don't need just to count boxes and
collect class ids per image — direct parsing is faster to load, has no native
dependency, and is trivial to unit-test against a tiny hand-written JSON.
pycocotools is still used later, where it earns its keep: mAP in the optional
train-twice harness.

COCO schema essentials we rely on:
  images:      [{id, file_name, width, height, ...}]
  annotations: [{image_id, category_id, bbox, ...}]
  categories:  [{id, name, ...}]

Optional same-scene metadata: if an image record carries a ``partition_key``
field (some derived/video datasets add one), we surface it so frames from the
same source never split across train/val/test.
"""

from __future__ import annotations

import json

from dedup.io.annotations import (
    AnnotationParser,
    ImageAnnotations,
    register_parser,
)


@register_parser("coco")
class CocoParser(AnnotationParser):
    def __init__(self, annotations_path: str):
        self.annotations_path = annotations_path

    def parse(self) -> dict[int | str, ImageAnnotations]:
        with open(self.annotations_path, "r") as fh:
            data = json.load(fh)

        records: dict[int | str, ImageAnnotations] = {}
        for img in data.get("images", []):
            records[img["id"]] = ImageAnnotations(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img.get("width"),
                height=img.get("height"),
                # Non-standard but harmless if absent — supports the leakage
                # hard-partition story for video-derived datasets.
                partition_key=img.get("partition_key"),
            )

        # Fold annotations onto their image. Annotations can reference an image
        # id not present in `images` in malformed files; we skip those rather
        # than crash, but this is exactly the kind of thing the profiler should
        # later surface rather than swallow silently.
        for ann in data.get("annotations", []):
            rec = records.get(ann["image_id"])
            if rec is None:
                continue
            rec.num_boxes += 1
            if "category_id" in ann:
                rec.class_ids.add(ann["category_id"])

        return records

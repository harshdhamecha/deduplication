"""Minimal torchvision detection training/eval for the train-twice harness.

Compact on purpose: a from-scratch Faster R-CNN, a short SGD loop, and
pycocotools mAP. On a tiny demo the absolute mAP will be low (from-scratch, few
epochs) — the harness exists to demonstrate the RAW-vs-DEDUPED *comparison*
methodology, which is scale-independent, not to hit a benchmark number.

Everything torch/torchvision is imported here so importing the rest of the
project never drags in a training stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dedup import get_logger
from dedup.config import Config

logger = get_logger()


def _coco_index(annotations_path: str):
    """Return (by_basename->image_id, image_id->anns, cat_ids, cat->contiguous)."""
    data = json.loads(Path(annotations_path).read_text())
    by_basename = {os.path.basename(im["file_name"]): im["id"] for im in data["images"]}
    anns_by_img: dict[int, list] = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    cat_ids = sorted({c["id"] for c in data["categories"]})
    # torchvision labels are 1..K (0 = background); map COCO cat ids onto that.
    cat_to_contig = {c: i + 1 for i, c in enumerate(cat_ids)}
    contig_to_cat = {i + 1: c for i, c in enumerate(cat_ids)}
    return by_basename, anns_by_img, cat_ids, cat_to_contig, contig_to_cat


def _make_dataset(paths, annotations_path, require_boxes: bool):
    import torch  # noqa: PLC0415
    import torchvision.transforms.functional as F  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    by_basename, anns_by_img, _, cat_to_contig, _ = _coco_index(annotations_path)

    samples = []
    for p in paths:
        img_id = by_basename.get(os.path.basename(p))
        if img_id is None:
            continue
        boxes, labels = [], []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])           # xywh -> xyxy
            labels.append(cat_to_contig[a["category_id"]])
        if require_boxes and not boxes:
            continue                                      # training skips empty images
        samples.append((p, img_id, boxes, labels))

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            p, img_id, boxes, labels = samples[i]
            with Image.open(p) as im:
                img = F.to_tensor(im.convert("RGB"))
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }
            return img, target

    return _DS()


def _num_classes(annotations_path: str) -> int:
    _, _, cat_ids, _, _ = _coco_index(annotations_path)
    return len(cat_ids) + 1  # + background


def train_detector(paths, cfg: Config, epochs: int):
    import torch  # noqa: PLC0415
    from torchvision.models.detection import fasterrcnn_resnet50_fpn  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() and cfg.stage3.device != "cpu" else "cpu"
    ds = _make_dataset(paths, cfg.io.annotations, require_boxes=True)
    if len(ds) == 0:
        logger.warning("train-twice: no annotated training images; returning untrained model")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True,
                                         collate_fn=lambda b: tuple(zip(*b)))
    # weights=None: train from scratch (offline-friendly). num_classes from subset.
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    num_classes=_num_classes(cfg.io.annotations)).to(device)
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.005, momentum=0.9, weight_decay=0.0005)
    for ep in range(epochs):
        running = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
        logger.info("train-twice epoch %d/%d: loss=%.3f", ep + 1, epochs, running)
    return model


def _sanitized_gt(cfg: Config) -> str:
    """Write a COCO gt copy with iscrowd/area backfilled, so COCOeval works on
    minimal/custom COCO files too (real COCO already carries these fields)."""
    data = json.loads(Path(cfg.io.annotations).read_text())
    for a in data["annotations"]:
        a.setdefault("iscrowd", 0)
        if "area" not in a:
            _, _, w, h = a["bbox"]
            a["area"] = float(w) * float(h)
    out = Path(cfg.io.output_dir) / "_gt_sanitized.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    return str(out)


def evaluate_map(model, test_paths, cfg: Config) -> float:
    import torch  # noqa: PLC0415
    from pycocotools.coco import COCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval  # noqa: PLC0415

    device = next(model.parameters()).device
    ds = _make_dataset(test_paths, cfg.io.annotations, require_boxes=False)
    _, _, _, _, contig_to_cat = _coco_index(cfg.io.annotations)

    model.eval()
    results, img_ids = [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            img, target = ds[i]
            img_id = int(target["image_id"][0])
            img_ids.append(img_id)
            pred = model([img.to(device)])[0]
            for box, label, score in zip(pred["boxes"].cpu(), pred["labels"].cpu(),
                                         pred["scores"].cpu()):
                x1, y1, x2, y2 = [float(v) for v in box]
                results.append({
                    "image_id": img_id,
                    "category_id": contig_to_cat.get(int(label), int(label)),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],   # xyxy -> xywh
                    "score": float(score),
                })

    if not results:
        logger.warning("train-twice: model produced no detections; mAP=0")
        return 0.0

    gt = COCO(_sanitized_gt(cfg))
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = sorted(set(img_ids))
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])  # AP @ IoU=0.50:0.95 (primary COCO metric)

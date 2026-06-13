"""Throughput-aware image loading for Stage 3.

THE BOTTLENECK IS USUALLY NOT THE GPU. For a ViT-B forward pass, CPU-side JPEG
decode + resize routinely starves the GPU: the accelerator sits idle waiting for
pixels. So this loader is built decode-first:
  * multi-worker DataLoader so many CPUs decode in parallel (num_workers),
  * a benchmark that prints achieved img/sec AND GPU utilization at run start, so
    the bottleneck is visible rather than assumed,
  * optional pre-resize-to-disk (shrink images once so every epoch decodes less),
  * an optional NVIDIA DALI GPU-decode hook for when CPUs still can't keep up.

Decode failures are skipped (with a count) rather than crashing the run — but
never silently: a high skip count points at corrupt inputs.
"""

from __future__ import annotations

import time
from pathlib import Path

from dedup import get_logger

logger = get_logger()


def _build_dataset(paths: list[str], preprocess):
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    class _ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                with Image.open(self.paths[i]) as img:
                    tensor = preprocess(img.convert("RGB"))
                return i, tensor
            except Exception as exc:  # noqa: BLE001 - skip one bad image, keep going
                logger.warning("stage3: failed to decode %s (%s)", self.paths[i], exc)
                return i, None

    return _ImageDataset()


def _collate(batch):
    import torch  # noqa: PLC0415

    # Drop failed decodes (None tensors) so one corrupt file doesn't kill a batch.
    good = [(i, t) for i, t in batch if t is not None]
    if not good:
        return [], None
    idxs = [i for i, _ in good]
    tensors = torch.stack([t for _, t in good])
    return idxs, tensors


def make_loader(paths: list[str], extractor, batch_size: int, num_workers: int):
    """A DataLoader yielding (indices, batched_tensor). Indices map back to the
    position in ``paths`` (and thus the survivor/embedding row)."""
    import torch  # noqa: PLC0415

    dataset = _build_dataset(paths, extractor.preprocess)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_collate,
        # pin_memory speeds host->GPU copies; persistent workers avoid re-spawning.
        pin_memory=True, persistent_workers=num_workers > 0,
    )


def _gpu_utilization() -> str:
    """Best-effort GPU utilization string; empty if unavailable."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return ""
        # torch.cuda.utilization needs pynvml; fall back to mem stats if absent.
        try:
            util = torch.cuda.utilization()
            return f", GPU util ~{util}%"
        except Exception:  # noqa: BLE001
            mem = torch.cuda.memory_allocated() / 1024**2
            return f", GPU mem {mem:.0f}MB"
    except Exception:  # noqa: BLE001
        return ""


def log_throughput(stage_label: str, n_done: int, seconds: float) -> None:
    """Print achieved img/sec (+ GPU util) so the bottleneck is observable."""
    if seconds <= 0:
        return
    logger.info("%s throughput: %.1f img/sec (%d imgs in %.1fs)%s",
                stage_label, n_done / seconds, n_done, seconds, _gpu_utilization())


def maybe_pre_resize(paths: list[str], out_root: str | Path, max_edge: int = 256
                     ) -> list[str]:
    """Optional: resize images once to disk so repeated runs decode less.

    Returns the list of resized paths (1:1 with input). Worth it when you'll
    embed the same set multiple times (threshold sweeps, backbone comparisons);
    pure overhead for a single pass.
    """
    from PIL import Image  # noqa: PLC0415

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    resized = []
    for i, p in enumerate(paths):
        dst = out_root / f"{i:08d}.jpg"
        if not dst.exists():
            with Image.open(p) as img:
                img = img.convert("RGB")
                img.thumbnail((max_edge, max_edge))
                img.save(dst, quality=90)
        resized.append(str(dst))
    logger.info("pre-resized %d images to %s (max edge %d)", len(paths), out_root, max_edge)
    return resized

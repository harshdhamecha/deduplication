"""Embedding backbones behind one interface.

Adding a backbone is a small class + one `@register_backbone` line — the rest of
Stage 3 (loader, store, index) is backbone-agnostic because everything downstream
only sees an (N, dim) array of normalized vectors.

Backbone choices and WHY each exists:
  dinov2_vitb14 (default) — DINOv2 is self-supervised and its features preserve
    SPATIAL structure (it was trained with dense objectives), which suits
    detection: we want "same scene / same objects", not just "same vibe". 768-d.
  dinov2_large            — the locally-downloaded ViT-L/14 (1024-d); higher
    capacity, slower, no download needed. Same family as the default.
  clip                    — contrastive image-text features = GLOBAL semantic
    similarity. Great for "two photos of a beach", which is often TOO loose for
    detection dedup; offered for comparison so the difference is visible.
  sscd                    — Meta's self-supervised COPY-detection model: tuned to
    catch edited/re-encoded copies rather than semantic siblings. The "true copy
    vs semantic" contrast. Stubbed here (needs separate weights) — see below.

torch/transformers are imported lazily so Stages 1-2 and the index tests run
without them installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dedup import get_logger
from dedup.config import Stage3Config

logger = get_logger()


class Extractor(ABC):
    dim: int

    @abstractmethod
    def preprocess(self, pil_image):
        """PIL.Image -> a CHW float tensor ready to batch (runs in DataLoader workers)."""

    @abstractmethod
    def embed(self, batch) -> np.ndarray:
        """A stacked batch tensor -> (B, dim) float32 embeddings (un-normalized;
        the store normalizes)."""


_BACKBONES: dict[str, type[Extractor]] = {}


def register_backbone(name: str):
    def deco(cls):
        _BACKBONES[name] = cls
        return cls

    return deco


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch  # noqa: PLC0415

    return "cuda" if torch.cuda.is_available() else "cpu"


@register_backbone("dinov2_vitb14")
@register_backbone("dinov2_large")
class DinoV2Extractor(Extractor):
    """HuggingFace Dinov2Model. Uses pooler_output (layernormed CLS token) as the
    image embedding. Default repo is facebook/dinov2-base (768-d); pass a local
    model_path (e.g. models/dinov2_large, 1024-d) to use downloaded weights."""

    # Default HF repos per backbone name; overridden by cfg.model_path if set.
    _DEFAULT_REPO = {
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2_large": "facebook/dinov2-large",
    }

    def __init__(self, cfg: Stage3Config):
        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        self.device = _resolve_device(cfg.device)
        source = cfg.model_path or self._DEFAULT_REPO[cfg.backbone]
        logger.info("loading DINOv2 backbone '%s' from %s on %s",
                    cfg.backbone, source, self.device)
        self.processor = AutoImageProcessor.from_pretrained(source)
        self.model = AutoModel.from_pretrained(source).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        self._torch = torch

    def preprocess(self, pil_image):
        # Return a CHW tensor; the HF processor handles resize/crop/normalize.
        return self.processor(pil_image, return_tensors="pt")["pixel_values"][0]

    def embed(self, batch) -> np.ndarray:
        with self._torch.inference_mode():
            out = self.model(pixel_values=batch.to(self.device))
            # pooler_output is the layernormed CLS embedding; fall back to CLS
            # token of last_hidden_state if a model lacks a pooler.
            emb = getattr(out, "pooler_output", None)
            if emb is None:
                emb = out.last_hidden_state[:, 0]
        return emb.float().cpu().numpy()


@register_backbone("clip")
class ClipExtractor(Extractor):
    """open_clip image encoder (global semantic features)."""

    def __init__(self, cfg: Stage3Config):
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self.device = _resolve_device(cfg.device)
        model_name = "ViT-B-32"
        pretrained = cfg.model_path or "laion2b_s34b_b79k"
        logger.info("loading CLIP '%s' (%s) on %s", model_name, pretrained, self.device)
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.dim = self.model.visual.output_dim
        self._torch = torch

    def preprocess(self, pil_image):
        return self._preprocess(pil_image)

    def embed(self, batch) -> np.ndarray:
        with self._torch.inference_mode():
            emb = self.model.encode_image(batch.to(self.device))
        return emb.float().cpu().numpy()


@register_backbone("sscd")
class SscdExtractor(Extractor):
    """Meta SSCD copy-detection backbone. Intentionally a stub: SSCD ships as a
    separate TorchScript/checkpoint download, not via transformers/open_clip.

    To enable it: download an SSCD checkpoint (e.g. sscd_disc_mixup.torchscript.pt),
    point cfg.model_path at it, and fill in __init__/embed below — the resize-320
    + normalize preprocess and a forward pass are ~10 lines. Left unwired so the
    demo doesn't depend on an out-of-band weight download."""

    def __init__(self, cfg: Stage3Config):
        raise NotImplementedError(
            "SSCD backbone is stubbed. Download an SSCD checkpoint, set "
            "stage3.model_path to it, and implement SscdExtractor (see docstring)."
        )

    def preprocess(self, pil_image):  # pragma: no cover - unreachable until wired
        raise NotImplementedError

    def embed(self, batch) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def build_extractor(cfg: Stage3Config) -> Extractor:
    if cfg.backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{cfg.backbone}' (have: {sorted(_BACKBONES)})")
    return _BACKBONES[cfg.backbone](cfg)

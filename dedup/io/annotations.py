"""Annotation parsing interface.

We deliberately put a thin abstraction in front of the concrete COCO parser so
that adding YOLO / Pascal-VOC support later is a new class + a registry entry,
touching nothing downstream. Stage 4's annotation-aware resolution only needs a
handful of fields per image (box count, class diversity, dimensions, an optional
partition key) — so that minimal contract is what the interface exposes, not the
full annotation richness of any one format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ImageAnnotations:
    """The detection metadata Stage 4 needs to resolve a duplicate cluster.

    Kept format-agnostic on purpose: every parser maps its native schema onto
    this. ``partition_key`` carries same-scene grouping info (video id, capture
    session, source URL) when the dataset provides it — the leakage stage uses
    it as a hard split boundary.
    """

    image_id: int | str
    file_name: str
    width: int | None = None
    height: int | None = None
    num_boxes: int = 0
    class_ids: set[int] = field(default_factory=set)
    partition_key: str | None = None

    @property
    def num_classes(self) -> int:
        """Class diversity — a tiebreak for ``keep_most_annotated``."""
        return len(self.class_ids)

    @property
    def area(self) -> int:
        """Pixel area; backs the ``keep_highest_res`` strategy."""
        if self.width is None or self.height is None:
            return 0
        return self.width * self.height


class AnnotationParser(ABC):
    """Maps a dataset's annotations onto ``{image_id: ImageAnnotations}``."""

    @abstractmethod
    def parse(self) -> dict[int | str, ImageAnnotations]:
        ...


# Registry so the format string in config selects a parser without imports
# leaking across the codebase. New formats register here.
_PARSERS: dict[str, type[AnnotationParser]] = {}


def register_parser(name: str):
    def deco(cls: type[AnnotationParser]) -> type[AnnotationParser]:
        _PARSERS[name] = cls
        return cls

    return deco


def get_parser(fmt: str, annotations_path: str) -> AnnotationParser:
    if fmt not in _PARSERS:
        raise NotImplementedError(
            f"No annotation parser for format '{fmt}'. Available: {sorted(_PARSERS)}"
        )
    return _PARSERS[fmt](annotations_path)

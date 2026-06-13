"""IO layer: annotation parsing (pluggable) and image enumeration.

This package is named ``dedup.io`` rather than a top-level ``io`` on purpose —
a top-level ``io`` package would shadow Python's standard-library ``io`` module
and break imports across the interpreter. Nesting under ``dedup`` keeps the
spec's intended module name while staying safe.
"""

from dedup.io.annotations import AnnotationParser, ImageAnnotations, get_parser
from dedup.io.coco import CocoParser
from dedup.io.images import enumerate_images

__all__ = [
    "AnnotationParser",
    "ImageAnnotations",
    "CocoParser",
    "get_parser",
    "enumerate_images",
]

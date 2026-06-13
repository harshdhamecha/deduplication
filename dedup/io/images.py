"""Image enumeration.

A generator over image paths under a root. WHY a generator (not a list): the
pipeline must stream — at large scale we cannot hold every path, let alone every
image, in memory. Callers that genuinely need a list (small data) can wrap it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# Lower-cased extensions we treat as images. Kept explicit so a stray .txt or
# .json sitting next to the images never enters the pipeline.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def enumerate_images(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    """Yield image file paths under ``root`` in sorted order per directory.

    Sorted ordering makes runs deterministic (important for resumability — a
    checkpoint that says "processed up to index N" must mean the same N on
    restart).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    yield Path(dirpath) / name
    else:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

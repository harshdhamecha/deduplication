"""Per-stage metrics: items in/removed, wall time, peak memory, and free-form
extras (e.g. the chosen index type).

Every stage emits one StageMetrics record. Collected together they become the
final JSON + human-readable summary the project is measured by, so the schema is
deliberately uniform across stages.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dedup import get_logger

logger = get_logger()


@dataclass
class StageMetrics:
    stage: str
    items_in: int = 0
    items_removed: int = 0
    seconds: float = 0.0
    peak_mem_mb: float = 0.0
    # Stage-specific facts worth surfacing: index_type, backend, n_groups, etc.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def items_out(self) -> int:
        return self.items_in - self.items_removed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items_out"] = self.items_out
        return d

    def save(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / self.stage / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


def _peak_rss_mb() -> float:
    """Process peak resident memory in MB.

    ru_maxrss is in KB on Linux but BYTES on macOS — normalise so the number is
    comparable wherever the demo runs.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return maxrss / divisor


@contextmanager
def track_stage(stage: str, items_in: int):
    """Time a stage and capture its peak memory.

    Usage:
        with track_stage("stage1", n) as m:
            ...                       # do work
            m.items_removed = removed
            m.extra["backend"] = "lmdb"
    The wall time and peak memory are filled in automatically on exit.
    """
    m = StageMetrics(stage=stage, items_in=items_in)
    start = time.perf_counter()
    try:
        yield m
    finally:
        m.seconds = round(time.perf_counter() - start, 3)
        m.peak_mem_mb = round(_peak_rss_mb(), 1)
        # No silent failures: a stage that removed nothing on real input is
        # suspicious enough to warn about — it usually means a misconfiguration.
        if items_in > 0 and m.items_removed == 0:
            logger.warning(
                "%s removed 0 of %d items — verify thresholds/backbone; this is "
                "often a misconfiguration, not a clean dataset.", stage, items_in,
            )
        logger.info(
            "%s done: in=%d removed=%d out=%d  %.2fs  peakRSS=%.0fMB",
            stage, m.items_in, m.items_removed, m.items_out, m.seconds, m.peak_mem_mb,
        )

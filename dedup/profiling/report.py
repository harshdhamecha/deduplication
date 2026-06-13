"""Final report: aggregate every stage's metrics into the project's payload.

Two outputs, same content: a machine-readable report.json and a human-readable
summary printed to the terminal. The summary leads with the two numbers that
make the resume story — total redundancy removed (broken down by exact / near /
semantic) and the train/test leakage fraction — because those are the defensible
"impact" claims the whole pipeline exists to produce.

All numbers come from the on-disk stage artifacts (metrics.json, leakage.json),
so the report is reproducible from a finished run with no recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path

from dedup import get_logger
from dedup.config import Config

logger = get_logger()

_STAGE_LABELS = {
    "stage1": "exact (SHA-256)",
    "stage2": "near-duplicate (perceptual)",
    "stage3": "semantic (embedding)",
    "stage4": "cluster resolution",
}


def _load_metrics(output_dir: Path, stage: str) -> dict | None:
    path = output_dir / stage / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_report(cfg: Config) -> dict:
    output_dir = Path(cfg.io.output_dir)
    stages = {s: _load_metrics(output_dir, s)
              for s in ("stage1", "stage2", "stage3", "stage4")}
    ran = {s: m for s, m in stages.items() if m is not None}
    if not ran:
        logger.warning("report: no stage metrics found under %s — run the pipeline "
                       "first (dedup run-all).", output_dir)
        return {"error": "no stage metrics found", "output_dir": str(output_dir)}

    # Initial count = first stage that ran; final survivors = last stage that ran.
    first = next(iter(ran.values()))
    last = list(ran.values())[-1]
    initial = first["items_in"]
    final = last["items_out"]
    total_removed = initial - final

    distribution = {
        s: {"label": _STAGE_LABELS[s], "removed": m["items_removed"],
            "fraction_of_initial": round(m["items_removed"] / initial, 4) if initial else 0.0}
        for s, m in ran.items()
    }

    report = {
        "dataset": {"image_root": cfg.io.image_root, "annotations": cfg.io.annotations},
        "totals": {
            "initial_images": initial,
            "final_images": final,
            "removed": total_removed,
            "reduction_fraction": round(total_removed / initial, 4) if initial else 0.0,
        },
        "removal_distribution": distribution,
        "index_type": ran.get("stage3", {}).get("extra", {}).get("index_type"),
        "leakage": ran.get("stage4", {}).get("extra", {}).get("leakage", {}),
        "per_stage": {s: {k: m[k] for k in ("items_in", "items_removed", "items_out",
                                            "seconds", "peak_mem_mb", "extra")}
                      for s, m in ran.items()},
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _print_summary(r: dict) -> None:
    t = r["totals"]
    print("\n" + "=" * 60)
    print(" DEDUPLICATION REPORT")
    print("=" * 60)
    print(f" Initial images : {t['initial_images']:,}")
    print(f" Final images   : {t['final_images']:,}")
    print(f" Removed        : {t['removed']:,}  ({t['reduction_fraction'] * 100:.2f}% redundant)")
    if r.get("index_type"):
        print(f" Stage-3 index  : {r['index_type']}")
    # Stages 1-3 flag duplicates by TYPE (provisional, greedy). Stage 4 re-clusters
    # across all of them and makes the authoritative removal — so it is shown
    # separately, not as another addend (the per-type rows are not meant to sum).
    print("\n Flagged by detector type (provisional):")
    for s in ("stage1", "stage2", "stage3"):
        d = r["removal_distribution"].get(s)
        if d:
            print(f"   {d['label']:<28} {d['removed']:>8,}  ({d['fraction_of_initial'] * 100:.2f}%)")
    s4 = r["removal_distribution"].get("stage4")
    if s4:
        print(f"\n Final removed (after cross-stage cluster resolution): "
              f"{s4['removed']:,} ({s4['fraction_of_initial'] * 100:.2f}%)")
    if r["leakage"]:
        print("\n Train/test leakage (HEADLINE):")
        for split, frac in r["leakage"].items():
            print(f"   {split:<10} {frac * 100:.2f}% have a near-duplicate in train")
    print("\n Per-stage timing / memory:")
    for s, m in r["per_stage"].items():
        print(f"   {s}: {m['seconds']:.2f}s, peak {m['peak_mem_mb']:.0f}MB")
    print("=" * 60)

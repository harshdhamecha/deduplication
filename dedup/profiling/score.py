"""Score a pipeline run against a planted ground-truth manifest.

`dedup plant` knows the truth — which files are duplicates and which stage *should*
catch each one. After a run, the stages record their own opinion on disk. This
module joins the two and reports the metrics that make the impact claim defensible:

  * recall    — of the duplicates we planted, how many did the cascade catch?
  * precision — of everything the cascade removed, how much was a real planted dup?
  * per-tier  — did each tier die at its *expected* stage (exact@1, near@2, semantic@3)?
                That is the test that the cascade design works: cheap stages should
                catch what they can so the expensive embedding stage only sees what
                genuinely needs it.

This is NOT mAP (that's `train-twice`, a downstream-model question) and NOT the
leakage headline (an absolute count with no ground truth). It is the dedup
*detector's* own precision/recall on known positives.

How the join works (all on file basename, since the manifest stores repo-relative
paths and the run stores image_root-relative paths):
  * A planted pair (planted-copy, original) is "caught at stage S" iff both basenames
    co-occur in one of stage S's duplicate_groups. Because each stage runs on the
    prior's survivors, a caught pair appears in exactly one stage's groups — the one
    that caught it — so the earliest match is unambiguous.
  * Removal precision uses Stage 4's authoritative removed.txt (the resolver may keep
    the planted copy and drop the original; either counts, since both are "planted-
    related"). Anything removed that is neither a planted copy nor an original is a
    false positive — a wrong merge (or a coincidental dup already in the source).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dedup import get_logger
from dedup.config import Config

logger = get_logger()

# expected_stage (from the manifest) -> human label, mirroring the cascade tiers.
_TIER_LABEL = {1: "exact   (S1)", 2: "near    (S2)", 3: "semantic(S3)"}


def _load_groups(output_dir: Path, stage: str) -> list[set[str]]:
    """Return each duplicate group at ``stage`` as a set of file basenames."""
    path = output_dir / stage / "duplicate_groups.json"
    if not path.exists():
        return []
    groups: list[set[str]] = []
    for g in json.loads(path.read_text()):
        members = {os.path.basename(g["kept"])}
        members.update(os.path.basename(r) for r in g["removed"])
        groups.append(members)
    return groups


def _removed_basenames(output_dir: Path) -> set[str]:
    """The authoritative set of removed file basenames.

    Prefer Stage 4's removed.txt (post cross-stage resolution); if Stage 4 didn't
    run, fall back to the union of the provisional per-stage removals so the score
    still works for a stages-1..3-only run."""
    rt = output_dir / "stage4" / "removed.txt"
    if rt.exists():
        return {os.path.basename(line) for line in rt.read_text().splitlines() if line}
    removed: set[str] = set()
    for stage in ("stage1", "stage2", "stage3"):
        path = output_dir / stage / "duplicate_groups.json"
        if path.exists():
            for g in json.loads(path.read_text()):
                removed.update(os.path.basename(r) for r in g["removed"])
    return removed


def score_plant(cfg: Config, manifest_path: str) -> dict:
    """Score the run under ``cfg.io.output_dir`` against the plant manifest."""
    mpath = Path(manifest_path)
    if not mpath.exists():
        # No silent failure: scoring without ground truth is meaningless.
        raise FileNotFoundError(
            f"No plant manifest at {mpath}. Run `dedup plant` first (it writes "
            f"plant_manifest.json next to the planted images).")

    manifest = json.loads(mpath.read_text())
    planted = manifest["planted"]
    output_dir = Path(cfg.io.output_dir)

    stage_groups = {s: _load_groups(output_dir, f"stage{s}") for s in (1, 2, 3)}
    removed = _removed_basenames(output_dir)

    def caught_at(planted_bn: str, original_bn: str) -> int | None:
        for s in (1, 2, 3):
            for members in stage_groups[s]:
                if planted_bn in members and original_bn in members:
                    return s
        return None

    # --- per-entry join ---
    entries = []
    seen_any = False  # did any planted basename surface in the run at all?
    for p in planted:
        pb, ob = os.path.basename(p["file"]), os.path.basename(p["original"])
        caught = caught_at(pb, ob)
        if caught is not None or pb in removed or ob in removed:
            seen_any = True
        entries.append({**p, "planted_bn": pb, "original_bn": ob, "caught_stage": caught,
                        "on_time": caught == p["expected_stage"]})

    # --- per-tier aggregation (keyed by the stage that *should* catch it) ---
    tiers: dict[int, dict] = {}
    for e in entries:
        t = tiers.setdefault(e["expected_stage"],
                             {"total": 0, "caught": 0, "on_time": 0, "missed": []})
        t["total"] += 1
        if e["caught_stage"] is not None:
            t["caught"] += 1
            if e["on_time"]:
                t["on_time"] += 1
        else:
            t["missed"].append(e)

    total = len(entries)
    total_caught = sum(t["caught"] for t in tiers.values())
    total_on_time = sum(t["on_time"] for t in tiers.values())

    # --- precision over authoritative removals ---
    planted_related = {e["planted_bn"] for e in entries} | {e["original_bn"] for e in entries}
    true_pos = removed & planted_related
    false_pos = removed - planted_related
    precision = (len(true_pos) / len(removed)) if removed else 1.0

    result = {
        "manifest": str(mpath),
        "output_dir": str(output_dir),
        "seed": manifest.get("seed"),
        "planted_total": total,
        "recall": {
            "caught": total_caught,
            "recall": round(total_caught / total, 4) if total else 0.0,
            "caught_on_expected_stage": total_on_time,
        },
        "precision": {
            "removed_total": len(removed),
            "true_positive_removals": len(true_pos),
            "false_positive_removals": len(false_pos),
            "precision": round(precision, 4),
            "false_positive_examples": sorted(false_pos)[:10],
        },
        "by_tier": {
            _TIER_LABEL[s]: {"total": t["total"], "caught": t["caught"],
                             "on_expected_stage": t["on_time"],
                             "recall": round(t["caught"] / t["total"], 4) if t["total"] else 0.0,
                             "missed": [m["planted_bn"] for m in t["missed"]]}
            for s, t in sorted(tiers.items())
        },
    }

    (output_dir / "score.json").write_text(json.dumps(result, indent=2))
    _warn(cfg, manifest, total, total_caught, seen_any)
    _print_summary(result, entries)
    return result


def _warn(cfg: Config, manifest: dict, total: int, total_caught: int, seen_any: bool) -> None:
    # If none of the planted files turn up in the run at all, the run almost
    # certainly wasn't pointed at the planted dataset — the usual mistake.
    if total and not seen_any:
        expected_root = os.path.join(manifest.get("out_root", "data/planted"), "images")
        logger.warning("score: not one planted file appears in the run under %s. Did the "
                       "run use the planted set? Re-run with --set io.image_root=%s.",
                       cfg.io.output_dir, expected_root)
    elif total and total_caught == 0:
        # Planted files are present but nothing was caught — suspicious, warn loudly.
        logger.warning("score: %d duplicates planted but the cascade caught 0 — this is "
                       "almost certainly a bug or a far-too-tight threshold.", total)


def _print_summary(r: dict, entries: list[dict]) -> None:
    rec, prec = r["recall"], r["precision"]
    print("\n" + "=" * 60)
    print(" PLANT SCORE (dedup detector recall / precision)")
    print("=" * 60)
    print(f" manifest : {r['manifest']}")
    print(f" planted  : {r['planted_total']}  (seed {r['seed']})\n")
    print(f" {'tier':<14}{'planted':>8}{'caught':>8}{'on-time':>9}{'recall':>9}")
    for tier, t in r["by_tier"].items():
        print(f"   {tier:<12}{t['total']:>8}{t['caught']:>8}"
              f"{t['on_expected_stage']:>9}{t['recall']:>9.2f}")
    print(" " + "-" * 48)
    print(f" overall recall : {rec['caught']}/{r['planted_total']}  ({rec['recall']:.3f})")
    late = rec["caught"] - rec["caught_on_expected_stage"]
    if late:
        print(f"   ! {late} caught LATE (a stage after the one expected) — see score.json")
    fp = prec["false_positive_removals"]
    print(f" precision      : {prec['true_positive_removals']}/{prec['removed_total']}  "
          f"({prec['precision']:.3f})   {fp} false removal(s)")

    missed = [e for e in entries if e["caught_stage"] is None]
    if missed:
        print(f"\n missed ({len(missed)}):")
        for e in missed[:10]:
            print(f"   {e['type']:<9} {e['planted_bn']}  (orig {e['original_bn']})")
    print("=" * 60)

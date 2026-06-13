"""Tests for the plant scorer — recall/precision joins against synthetic run artifacts.

These build the on-disk artifacts the stages would write (duplicate_groups.json,
stage4/removed.txt) by hand, so the scorer is tested in isolation from the heavy
pipeline. The manifest is produced by the real planter for fidelity.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from PIL import Image

from dedup.config import Config
from dedup.profiling.score import score_plant
from scripts.plant_duplicates import plant


def _sources(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 64), (i * 8 % 256, 64, 128)).save(root / f"src_{i:03d}.jpg")


def _manifest(tmp_path: Path) -> dict:
    mpath = plant(source_root=str(tmp_path / "src"), out_root=str(tmp_path / "planted"),
                  exact=2, near=2, semantic=2, seed=3)
    return json.loads(Path(mpath).read_text())


def _write_groups(output_dir: Path, stage: str, groups: list[dict]) -> None:
    d = output_dir / stage
    d.mkdir(parents=True, exist_ok=True)
    (d / "duplicate_groups.json").write_text(json.dumps(groups))


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    io = dataclasses.replace(cfg.io, output_dir=str(tmp_path / "runs"),
                             image_root=str(tmp_path / "planted/images"))
    return dataclasses.replace(cfg, io=io)


def _entries_by_stage(manifest: dict, stage: int) -> list[dict]:
    return [p for p in manifest["planted"] if p["expected_stage"] == stage]


def test_perfect_run_scores_full_recall_and_precision(tmp_path):
    _sources(tmp_path / "src", 6)
    manifest = _manifest(tmp_path)
    cfg = _cfg(tmp_path)
    out = Path(cfg.io.output_dir)

    removed = []
    # Each tier caught at its expected stage: group {original, planted-copy}.
    for stage in (1, 2, 3):
        groups = []
        for p in _entries_by_stage(manifest, stage):
            groups.append({"kept": p["original"], "removed": [p["file"]],
                           "reason": "test", "key": None})
            removed.append(Path(p["file"]).name)
        _write_groups(out, f"stage{stage}", groups)
    (out / "stage4").mkdir(parents=True, exist_ok=True)
    (out / "stage4" / "removed.txt").write_text("\n".join(removed) + "\n")

    res = score_plant(cfg, manifest["out_root"] + "/plant_manifest.json")
    assert res["recall"]["recall"] == 1.0
    assert res["recall"]["caught_on_expected_stage"] == 6
    assert res["precision"]["precision"] == 1.0
    assert res["precision"]["false_positive_removals"] == 0


def test_missed_semantic_lowers_recall(tmp_path):
    _sources(tmp_path / "src", 6)
    manifest = _manifest(tmp_path)
    cfg = _cfg(tmp_path)
    out = Path(cfg.io.output_dir)

    # Catch exact + near, but NOT the semantic tier (Stage 3 groups absent).
    removed = []
    for stage in (1, 2):
        groups = []
        for p in _entries_by_stage(manifest, stage):
            groups.append({"kept": p["original"], "removed": [p["file"]],
                           "reason": "test", "key": None})
            removed.append(Path(p["file"]).name)
        _write_groups(out, f"stage{stage}", groups)
    (out / "stage4").mkdir(parents=True, exist_ok=True)
    (out / "stage4" / "removed.txt").write_text("\n".join(removed) + "\n")

    res = score_plant(cfg, manifest["out_root"] + "/plant_manifest.json")
    assert res["recall"]["caught"] == 4          # 2 exact + 2 near
    assert res["by_tier"]["semantic(S3)"]["recall"] == 0.0


def test_false_positive_removal_lowers_precision(tmp_path):
    _sources(tmp_path / "src", 6)
    manifest = _manifest(tmp_path)
    cfg = _cfg(tmp_path)
    out = Path(cfg.io.output_dir)

    # Catch one exact pair correctly, but also wrongly remove an unrelated source.
    p = _entries_by_stage(manifest, 1)[0]
    _write_groups(out, "stage1", [{"kept": p["original"], "removed": [p["file"]],
                                   "reason": "test", "key": None}])
    (out / "stage4").mkdir(parents=True, exist_ok=True)
    (out / "stage4" / "removed.txt").write_text(
        Path(p["file"]).name + "\nsrc_999.jpg\n")  # src_999 is not planted-related

    res = score_plant(cfg, manifest["out_root"] + "/plant_manifest.json")
    assert res["precision"]["false_positive_removals"] == 1
    assert res["precision"]["precision"] == 0.5
    assert "src_999.jpg" in res["precision"]["false_positive_examples"]


def test_missing_manifest_errors(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        score_plant(cfg, str(tmp_path / "nope.json"))

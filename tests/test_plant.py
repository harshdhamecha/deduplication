"""Tests for the duplicate planter — manifest correctness + ground-truth invariants."""

import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from scripts.plant_duplicates import plant


def _make_sources(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # Distinct, non-trivial content so perceptual hashes are meaningful.
        Image.new("RGB", (64, 64), (i * 8 % 256, 64, 128)).save(root / f"src_{i:03d}.jpg")


def test_plant_writes_manifest_and_files(tmp_path):
    src = tmp_path / "images"
    _make_sources(src, 6)
    manifest_path = plant(source_root=str(src), out_root=str(tmp_path / "planted"),
                          exact=2, near=2, semantic=2, seed=1)

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["summary"]["total_planted"] == 6
    img_dir = tmp_path / "planted" / "images"
    # Every planted file and its named original must exist on disk.
    for rec in manifest["planted"]:
        assert (img_dir / Path(rec["file"]).name).exists()
        assert (img_dir / Path(rec["original"]).name).exists()


def test_exact_plant_is_byte_identical(tmp_path):
    src = tmp_path / "images"
    _make_sources(src, 4)
    manifest_path = plant(source_root=str(src), out_root=str(tmp_path / "planted"),
                          exact=4, near=0, semantic=0, seed=7)
    manifest = json.loads(Path(manifest_path).read_text())
    img_dir = tmp_path / "planted" / "images"

    for rec in manifest["planted"]:
        assert rec["expected_stage"] == 1
        dup = (img_dir / Path(rec["file"]).name).read_bytes()
        orig = (img_dir / Path(rec["original"]).name).read_bytes()
        assert hashlib.sha256(dup).hexdigest() == hashlib.sha256(orig).hexdigest()


def test_deterministic_with_seed(tmp_path):
    src = tmp_path / "images"
    _make_sources(src, 8)
    m1 = json.loads(Path(plant(source_root=str(src), out_root=str(tmp_path / "a"),
                               exact=3, near=3, semantic=2, seed=42)).read_text())
    m2 = json.loads(Path(plant(source_root=str(src), out_root=str(tmp_path / "b"),
                               exact=3, near=3, semantic=2, seed=42)).read_text())
    # Same seed -> same originals chosen, in the same order.
    assert [r["original"] for r in m1["planted"]] == [r["original"] for r in m2["planted"]]


def test_refuses_empty_plant(tmp_path):
    src = tmp_path / "images"
    _make_sources(src, 2)
    with pytest.raises(ValueError):
        plant(source_root=str(src), exact=0, near=0, semantic=0)


def test_errors_on_no_sources(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        plant(source_root=str(empty), out_root=str(tmp_path / "out"))

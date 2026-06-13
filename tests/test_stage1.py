"""Tests for Stage 1: SHA-256 hashing, storage backends, grouping, profiler."""

import pytest

from dedup.config import Config
from dedup.hashing.exact import run_stage1, sha256_file
from dedup.hashing.storage import InMemoryBackend, select_backend
from dedup.profiling.profiler import profile_exact


def _make_dataset(tmp_path):
    """3 files: a.jpg == b.jpg (exact dup), c.jpg unique."""
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"IDENTICAL-CONTENT")
    (imgs / "b.jpg").write_bytes(b"IDENTICAL-CONTENT")
    (imgs / "c.jpg").write_bytes(b"something-else")
    return imgs


def test_sha256_is_deterministic_and_content_based(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"xyz")
    b.write_bytes(b"xyz")
    assert sha256_file(a) == sha256_file(b)


def test_inmemory_backend_groups_by_key():
    be = InMemoryBackend()
    be.add("h1", "a")
    be.add("h1", "b")
    be.add("h2", "c")
    groups = dict(be.groups())
    assert groups["h1"] == ["a", "b"]
    assert groups["h2"] == ["c"]
    assert len(be) == 2


def test_select_backend_auto_uses_memory_below_threshold():
    be = select_backend("auto", estimated_count=10, threshold=100, lmdb_path=None)
    assert isinstance(be, InMemoryBackend)


def test_select_backend_explicit_memory_ignores_count():
    be = select_backend("memory", estimated_count=10_000_000, threshold=1, lmdb_path=None)
    assert isinstance(be, InMemoryBackend)


def test_select_backend_auto_chooses_lmdb_above_threshold(tmp_path):
    pytest.importorskip("lmdb")
    be = select_backend("auto", estimated_count=1000, threshold=100,
                        lmdb_path=tmp_path / "db.lmdb")
    assert type(be).__name__ == "LmdbBackend"
    be.close()


def test_run_stage1_removes_exact_duplicate(tmp_path):
    imgs = _make_dataset(tmp_path)
    out = tmp_path / "out"
    cfg = Config.load(None, [f"io.image_root={imgs}", f"io.output_dir={out}"])

    m = run_stage1(cfg)

    assert m.items_in == 3
    assert m.items_removed == 1          # one of the identical pair dropped
    assert m.items_out == 2
    assert m.extra["n_duplicate_groups"] == 1
    survivors = (out / "stage1" / "survivors.txt").read_text().split()
    assert len(survivors) == 2
    assert any(s.endswith("c.jpg") for s in survivors)   # unique file always kept


def test_profile_exact_reports_fraction(tmp_path):
    imgs = _make_dataset(tmp_path)
    paths = [str(imgs / n) for n in ("a.jpg", "b.jpg", "c.jpg")]
    rep = profile_exact(paths)
    assert rep["total_files"] == 3
    assert rep["distinct_files"] == 2
    assert rep["exact_duplicate_files"] == 1
    assert rep["exact_duplicate_fraction"] == pytest.approx(1 / 3, abs=1e-3)

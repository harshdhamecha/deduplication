"""Tests for the memory-mapped embedding store: normalization, fp16 round-trip,
non-contiguous row writes, and resume (reopen preserves written rows)."""

import numpy as np

from dedup.embeddings.store import EmbeddingStore


def test_store_normalizes_on_write(tmp_path):
    store = EmbeddingStore(tmp_path, dim=4, fp16=False)
    store.create(2)
    store.write_rows([0, 1], np.array([[3.0, 0, 0, 0], [0, 0, 0, 5.0]], dtype=np.float32))
    store.flush()
    mm = store.open_read()
    # Rows are L2-normalized so cosine == inner product downstream.
    assert np.allclose(np.linalg.norm(mm, axis=1), 1.0, atol=1e-4)


def test_non_contiguous_row_writes(tmp_path):
    store = EmbeddingStore(tmp_path, dim=3, fp16=False)
    store.create(3)
    # Workers deliver rows out of order; write by explicit index.
    store.write_rows([2, 0], np.array([[0, 0, 1.0], [1.0, 0, 0]], dtype=np.float32))
    store.flush()
    mm = store.open_read()
    assert np.allclose(mm[0], [1, 0, 0])
    assert np.allclose(mm[2], [0, 0, 1])


def test_reopen_preserves_written_rows(tmp_path):
    store = EmbeddingStore(tmp_path, dim=2, fp16=True)
    store.create(4)
    store.write_rows([0, 1], np.array([[1.0, 0], [0, 1.0]], dtype=np.float32))
    store.flush()

    # Simulate a resumed run: a fresh store object reopens the same files r+.
    store2 = EmbeddingStore(tmp_path, dim=2, fp16=True)
    store2.reopen_write()
    store2.write_rows([2, 3], np.array([[1.0, 0], [0, 1.0]], dtype=np.float32))
    store2.flush()
    mm = store2.open_read()
    assert np.allclose(mm[0], [1, 0], atol=1e-3)   # earlier rows survived
    assert np.allclose(mm[3], [0, 1], atol=1e-3)


def test_ids_round_trip(tmp_path):
    store = EmbeddingStore(tmp_path, dim=2)
    store.save_ids(["a/1.jpg", "b/2.jpg"])
    assert store.load_ids() == ["a/1.jpg", "b/2.jpg"]

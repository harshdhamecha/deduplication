"""Stage 1 (exact, SHA-256) and Stage 2 (perceptual hashing) live here.

Planned contents (filled in Steps 2 & 3):
  exact.py      SHA-256 byte hashing + grouping of identical files.
  storage.py    StorageBackend interface; in-memory dict + LMDB backends,
                auto-selected by estimated item count.
  perceptual.py phash/dhash/ahash/whash behind one hash interface.
  search.py     three candidate-retrieval strategies behind one interface:
                brute-force Hamming, multi-index hashing, BK-tree.
"""

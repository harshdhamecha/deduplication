"""Scale-adaptive image deduplication pipeline for object-detection datasets.

The pipeline is a cascade of four independent, individually-runnable stages:

    Stage 1  exact duplicates        (SHA-256 over raw bytes)
    Stage 2  near-duplicates         (perceptual hashes + Hamming search)
    Stage 3  semantic duplicates     (embeddings + scale-adaptive ANN index)
    Stage 4  cluster resolution      (Union-Find) + cross-split leakage check

Each stage consumes the survivors of the previous one and writes its state to
disk, so any stage can be run, re-run, or resumed in isolation.
"""

import logging

__version__ = "0.1.0"


def get_logger(name: str = "dedup", level: str = "INFO") -> logging.Logger:
    """Return a configured logger.

    Centralised so every stage logs the same way — and so the scale-tier and
    index-selection decisions (which we *require* to be visible) land in a
    consistent, greppable format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

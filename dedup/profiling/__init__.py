"""Measurement harness (filled in Steps 2 & 6).

Profiling-first discipline: we measure the duplicate distribution BEFORE removing
anything, so the impact number is defensible.

Planned contents:
  profiler.py  Duplicate-distribution profiler: fraction exact / near-dup /
               semantic-dup, with a histogram.
  metrics.py   Per-stage metrics record (items in, items removed, time elapsed,
               index type chosen, peak memory) -> final JSON + human summary.
  report.py    Renders the JSON + a human-readable summary, including the
               leakage headline metric.
"""

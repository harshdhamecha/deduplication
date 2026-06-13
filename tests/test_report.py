"""Integration tests for the cluster builder and the report aggregator."""

import dataclasses
import json

from dedup.clustering.cluster import build_clusters
from dedup.config import Config
from dedup.profiling.report import build_report


def _write_metrics(out, stage, items_in, removed, extra=None):
    d = out / stage
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({
        "stage": stage, "items_in": items_in, "items_removed": removed,
        "items_out": items_in - removed, "seconds": 0.1, "peak_mem_mb": 50.0,
        "extra": extra or {},
    }))


def test_build_clusters_unions_stage2_and_stage3_pairs(tmp_path):
    out = tmp_path / "out"
    (out / "stage1").mkdir(parents=True)
    (out / "stage1" / "survivors.txt").write_text("a\nb\nc\nd\n")
    (out / "stage2").mkdir()
    (out / "stage2" / "pairs.json").write_text(json.dumps([["a", "b"]]))
    (out / "stage3").mkdir()
    (out / "stage3" / "pairs.json").write_text(json.dumps([["b", "c"]]))

    comps = build_clusters(image_root="unused", output_dir=out)
    # a-b (stage2) and b-c (stage3) union transitively -> {a,b,c}; d singleton.
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 3]


def test_report_aggregates_totals_and_leakage(tmp_path, capsys):
    out = tmp_path / "out"
    _write_metrics(out, "stage1", 100, 10)
    _write_metrics(out, "stage2", 90, 5)
    _write_metrics(out, "stage3", 85, 15, extra={"index_type": "FlatIndex"})
    _write_metrics(out, "stage4", 100, 30, extra={"leakage": {"val": 0.12}})

    cfg = dataclasses.replace(Config(), io=dataclasses.replace(Config().io,
                                                               output_dir=str(out)))
    rep = build_report(cfg)
    assert rep["totals"]["initial_images"] == 100
    assert rep["totals"]["final_images"] == 70          # stage4 items_out
    assert rep["totals"]["removed"] == 30
    assert rep["index_type"] == "FlatIndex"
    assert rep["leakage"] == {"val": 0.12}
    assert (out / "report.json").exists()
    assert "HEADLINE" in capsys.readouterr().out

"""Tests for the COCO parser: box counting, class diversity, partition key."""

import json

from dedup.io import get_parser
from dedup.io.coco import CocoParser


def _write_coco(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100,
             "partition_key": "video_7"},
            {"id": 3, "file_name": "c.jpg", "width": 200, "height": 200},  # no boxes
        ],
        "annotations": [
            {"image_id": 1, "category_id": 10, "bbox": [0, 0, 5, 5]},
            {"image_id": 1, "category_id": 10, "bbox": [1, 1, 5, 5]},
            {"image_id": 1, "category_id": 20, "bbox": [2, 2, 5, 5]},
            {"image_id": 2, "category_id": 10, "bbox": [0, 0, 9, 9]},
            {"image_id": 999, "category_id": 30, "bbox": [0, 0, 1, 1]},  # dangling ref
        ],
        "categories": [{"id": 10, "name": "x"}, {"id": 20, "name": "y"}],
    }
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(coco))
    return p


def test_box_count_and_class_diversity(tmp_path):
    recs = CocoParser(str(_write_coco(tmp_path))).parse()
    assert recs[1].num_boxes == 3
    assert recs[1].class_ids == {10, 20}
    assert recs[1].num_classes == 2
    assert recs[1].area == 640 * 480


def test_partition_key_and_zero_box_image(tmp_path):
    recs = CocoParser(str(_write_coco(tmp_path))).parse()
    assert recs[2].partition_key == "video_7"
    assert recs[3].num_boxes == 0          # image with no annotations still present
    assert recs[3].class_ids == set()


def test_dangling_annotation_is_skipped_not_crashed(tmp_path):
    recs = CocoParser(str(_write_coco(tmp_path))).parse()
    assert 999 not in recs                 # annotation referenced a missing image -> skipped


def test_registry_resolves_coco(tmp_path):
    parser = get_parser("coco", str(_write_coco(tmp_path)))
    assert isinstance(parser, CocoParser)

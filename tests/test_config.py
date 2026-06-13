"""Tests for the config system: defaults, YAML merge, CLI overrides, error cases."""

import pytest

from dedup.config import Config


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.stage2.hamming_threshold == 8
    assert cfg.stage3.backbone == "dinov2_vitb14"
    assert cfg.stage3.index_type == "auto"
    assert cfg.stage3.cosine_threshold == pytest.approx(0.92)


def test_yaml_merge_overlays_only_specified_keys(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "stage2:\n  hamming_threshold: 4\nstage3:\n  backbone: clip\n"
    )
    cfg = Config.load(yaml_path)
    assert cfg.stage2.hamming_threshold == 4          # overridden
    assert cfg.stage2.hashes == ["phash", "dhash"]    # untouched default preserved
    assert cfg.stage3.backbone == "clip"


def test_cli_overrides_take_precedence_and_coerce_types():
    cfg = Config.load(
        None,
        ["stage2.hamming_threshold=6", "stage3.fp16_embeddings=false",
         "stage3.cosine_threshold=0.85"],
    )
    assert cfg.stage2.hamming_threshold == 6           # int coercion
    assert cfg.stage3.fp16_embeddings is False         # bool coercion
    assert cfg.stage3.cosine_threshold == pytest.approx(0.85)  # float coercion


def test_unknown_yaml_key_fails_loudly(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("stage2:\n  not_a_real_key: 3\n")
    with pytest.raises(KeyError):
        Config.load(yaml_path)


def test_unknown_override_key_fails_loudly():
    with pytest.raises(KeyError):
        Config.load(None, ["stage9.nope=1"])

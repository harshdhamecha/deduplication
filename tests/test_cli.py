"""CLI smoke tests via click's CliRunner — commands wire up and config flows."""

import json

from click.testing import CliRunner

from dedup.cli import main


def test_show_config_emits_valid_json_with_overrides():
    res = CliRunner().invoke(main, ["--set", "stage2.hamming_threshold=5", "show-config"])
    assert res.exit_code == 0
    cfg = json.loads(res.output)
    assert cfg["stage2"]["hamming_threshold"] == 5


def test_help_lists_all_commands():
    res = CliRunner().invoke(main, ["--help"])
    assert res.exit_code == 0
    for cmd in ("profile", "run", "run-all", "report", "fetch-data", "train-twice",
                "plant", "score", "make-leakage-demo"):
        assert cmd in res.output


def test_run_rejects_out_of_range_stage():
    res = CliRunner().invoke(main, ["run", "--stage", "9"])
    assert res.exit_code != 0          # click IntRange(1,4) rejects 9


def test_bad_override_format_errors():
    res = CliRunner().invoke(main, ["--set", "no_equals_sign", "show-config"])
    assert res.exit_code != 0

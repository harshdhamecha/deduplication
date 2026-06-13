"""Command-line interface.

Subcommands mirror the demo story in the spec:

    dedup profile               profile the duplicate distribution (measure first)
    dedup run --stage N         run one stage in isolation (1..4)
    dedup run-all               run the full cascade
    dedup report                render the JSON + human-readable summary
    dedup fetch-data            download the small COCO val2017 demo subset
    dedup show-config           print the effective config (defaults+YAML+overrides)

Global options ``--config FILE`` and repeatable ``--set key.path=value`` feed the
config system, so every threshold/model/index choice is overridable without
editing a file. Stages not yet implemented print a clear, honest message rather
than pretending to succeed (the project's "no silent failures" rule).
"""

from __future__ import annotations

import json

import click

from dedup import __version__, get_logger
from dedup.config import Config

def _run_stage(cfg: Config, stage: int, force: bool) -> None:
    """Dispatch a single stage. Imports are local so a CPU-only / no-torch
    environment can still run Stages 1-2 and the index tests."""
    if stage == 1:
        from dedup.hashing.exact import run_stage1
        run_stage1(cfg, force=force)
    elif stage == 2:
        from dedup.hashing.near import run_stage2
        run_stage2(cfg, force=force)
    elif stage == 3:
        from dedup.embeddings.semantic import run_stage3
        run_stage3(cfg, force=force)
    elif stage == 4:
        from dedup.resolution.stage4 import run_stage4
        run_stage4(cfg, force=force)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="dedup")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="YAML config file (overrides built-in defaults).")
@click.option("--set", "overrides", multiple=True, metavar="key.path=value",
              help="Override any config leaf, repeatable. E.g. --set stage2.hamming_threshold=6")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, overrides: tuple[str, ...]) -> None:
    """Scale-adaptive image deduplication for object-detection datasets."""
    cfg = Config.load(config_path, list(overrides))
    ctx.obj = cfg
    get_logger(level=cfg.log_level)


@main.command("show-config")
@click.pass_obj
def show_config(cfg: Config) -> None:
    """Print the effective configuration (defaults < YAML < --set overrides)."""
    click.echo(json.dumps(cfg.to_dict(), indent=2, default=str))


@main.command()
@click.pass_obj
def profile(cfg: Config) -> None:
    """Profile the duplicate distribution BEFORE removing anything."""
    from dedup.profiling.profiler import profile as run_profile

    run_profile(cfg)


@main.command()
@click.option("--stage", type=click.IntRange(1, 4), required=True,
              help="Which stage to run in isolation (1..4).")
@click.option("--force", is_flag=True, help="Ignore checkpoints and recompute from scratch.")
@click.pass_obj
def run(cfg: Config, stage: int, force: bool) -> None:
    """Run a single stage on the survivors of the previous stage."""
    _run_stage(cfg, stage, force)


@main.command("run-all")
@click.option("--force", is_flag=True, help="Ignore checkpoints and recompute from scratch.")
@click.pass_obj
def run_all(cfg: Config, force: bool) -> None:
    """Run the full four-stage cascade, each stage on the prior's survivors."""
    for stage in (1, 2, 3, 4):
        _run_stage(cfg, stage, force)


@main.command()
@click.pass_obj
def report(cfg: Config) -> None:
    """Render the final JSON + human-readable summary (incl. leakage metric)."""
    from dedup.profiling.report import build_report

    build_report(cfg)


@main.command("train-twice")
@click.option("--epochs", type=int, default=2, show_default=True)
@click.option("--max-images", type=int, default=200, show_default=True,
              help="Cap images per run to keep the demo fast.")
@click.pass_obj
def train_twice(cfg: Config, epochs: int, max_images: int) -> None:
    """(Stretch) Train a small detector on raw vs deduped data, report mAP delta."""
    from dedup.profiling.train_twice import run_train_twice

    run_train_twice(cfg, epochs=epochs, max_images=max_images)


@main.command()
@click.option("--exact", type=int, default=10, show_default=True,
              help="Byte-identical copies to plant (Stage 1 ground truth).")
@click.option("--near", type=int, default=10, show_default=True,
              help="Crop+recompress near-dups to plant (Stage 2 ground truth).")
@click.option("--semantic", type=int, default=10, show_default=True,
              help="Zoom/jitter/rotate dups to plant (Stage 3 ground truth).")
@click.option("--source", type=click.Path(file_okay=False), default=None,
              help="Source image root [default: io.image_root from config].")
@click.option("--out", type=click.Path(file_okay=False), default="data/planted",
              show_default=True, help="Destination for the planted dataset + manifest.")
@click.pass_obj
def plant(cfg: Config, exact: int, near: int, semantic: int,
          source: str | None, out: str) -> None:
    """Plant labeled duplicates with a ground-truth manifest, to measure recall.

    Builds a self-contained dataset (source copies + planted exact/near/semantic
    duplicates) so a run's removals can be scored against the known positives.
    Seed comes from the config so the plant is reproducible.
    """
    from scripts.plant_duplicates import plant as run_plant  # local: keeps CLI import light

    run_plant(source_root=source or cfg.io.image_root, out_root=out,
              exact=exact, near=near, semantic=semantic, seed=cfg.seed)


@main.command()
@click.option("--manifest", type=click.Path(dir_okay=False),
              default="data/planted/plant_manifest.json", show_default=True,
              help="Ground-truth manifest written by `dedup plant`.")
@click.pass_obj
def score(cfg: Config, manifest: str) -> None:
    """Score a run against a plant manifest: recall, precision, per-tier breakdown.

    Reads the run artifacts under io.output_dir and the ground-truth manifest, then
    reports what fraction of planted duplicates the cascade caught (and whether each
    tier died at its expected stage). NOT mAP — this is the dedup detector's own
    precision/recall on known positives. Run after `dedup plant` + `run-all`.
    """
    from dedup.profiling.score import score_plant

    score_plant(cfg, manifest)


@main.command("make-leakage-demo")
@click.option("--val-frac", type=float, default=0.25, show_default=True,
              help="Fraction of the subset held out as the val split.")
@click.option("--leak-count", type=int, default=15, show_default=True,
              help="How many train images to plant into val as near-dup leaks.")
@click.option("--subset-json", type=click.Path(dir_okay=False),
              default="data/annotations/instances_val2017_subset.json", show_default=True,
              help="COCO subset JSON from `dedup fetch-data`.")
@click.option("--src-images", type=click.Path(file_okay=False),
              default="data/images/val2017", show_default=True,
              help="Where the fetched source images live.")
@click.option("--out", type=click.Path(file_okay=False), default="data/demo",
              show_default=True, help="Destination for the train/val demo dataset.")
def make_leakage_demo(val_frac: float, leak_count: int, subset_json: str,
                      src_images: str, out: str) -> None:
    """Build a train/val split with planted cross-split leakage (for the demo).

    Splits the fetched COCO subset and plants recompressed+cropped near-duplicates
    of train images into val, so the leakage headline has something real to catch.
    Run after `dedup fetch-data`, then `run-all` + `report` on the result.
    """
    from scripts.make_leakage_demo import make_demo  # local: keeps CLI import light

    make_demo(subset_json=subset_json, src_images=src_images, out_root=out,
              val_frac=val_frac, leak_count=leak_count)


@main.command("fetch-data")
@click.option("--num-images", type=int, default=200, show_default=True,
              help="How many COCO val2017 images to download for the demo.")
@click.option("--out", type=click.Path(file_okay=False), default="data",
              show_default=True, help="Destination directory.")
def fetch_data(num_images: int, out: str) -> None:
    """Download a small COCO val2017 subset for the end-to-end demo."""
    # Delegates to the standalone script so it's also runnable without the CLI.
    from scripts.fetch_coco_subset import fetch  # local import: keeps CLI import light

    fetch(num_images=num_images, out_dir=out)


if __name__ == "__main__":
    main()

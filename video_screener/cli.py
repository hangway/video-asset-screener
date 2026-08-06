"""`pipeline` CLI (typer): run stages individually or the whole pipeline.

    pipeline run ingest --config configs/default.yaml
    pipeline run all --config configs/default.yaml
    pipeline generate image     # run a local ComfyUI workflow
    pipeline samples            # (re)synthesize the sample clips
    pipeline validate           # validate an annotation/inference file
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .generation.cli import app as generation_app

app = typer.Typer(add_completion=False, help="AI video asset usability screening pipeline")
run_app = typer.Typer(help="Run one stage or the full pipeline")
app.add_typer(run_app, name="run")
app.add_typer(generation_app, name="generate")
console = Console()

STAGES = ["ingest", "prelabel", "annotate", "dataset", "train", "evaluate", "screen"]


def _load(config: Optional[str], workdir: Optional[str], video_dir: Optional[list[str]]):
    overrides: dict = {}
    if workdir:
        overrides["workdir"] = workdir
    if video_dir:
        overrides["video_dirs"] = list(video_dir)
    return load_config(config, **overrides)


def _dispatch(stage: str, cfg) -> dict:
    import importlib

    mod = importlib.import_module(f"video_screener.stages.{stage}")
    return mod.run(cfg)


def _print_summary(summary: dict) -> None:
    table = Table(show_header=False, box=None)
    for k, v in summary.items():
        table.add_row(f"[cyan]{k}[/cyan]", str(v))
    console.print(table)


@run_app.command("ingest")
def run_ingest(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None,
               video_dir: Optional[list[str]] = typer.Option(None)):
    cfg = _load(config, workdir, video_dir)
    _print_summary(_dispatch("ingest", cfg))


@run_app.command("prelabel")
def run_prelabel(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None):
    cfg = _load(config, workdir, None)
    _print_summary(_dispatch("prelabel", cfg))


@run_app.command("annotate")
def run_annotate(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None,
                 auto: bool = typer.Option(False, help="non-interactive: accept prelabels")):
    cfg = _load(config, workdir, None)
    from .stages import annotate

    _print_summary(annotate.run(cfg, auto=auto))


@run_app.command("dataset")
def run_dataset(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None):
    cfg = _load(config, workdir, None)
    _print_summary(_dispatch("dataset", cfg))


@run_app.command("train")
def run_train(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None):
    cfg = _load(config, workdir, None)
    _print_summary(_dispatch("train", cfg))


@run_app.command("evaluate")
def run_evaluate(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None):
    cfg = _load(config, workdir, None)
    _print_summary(_dispatch("evaluate", cfg))


@run_app.command("screen")
def run_screen(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None,
               video_dir: Optional[list[str]] = typer.Option(None),
               out: Optional[str] = typer.Option(None, help="output dir for screen report"),
               reference_dir: Optional[str] = typer.Option(
                   None, help="reference images (one subdir per subject) for "
                              "consistency checks")):
    cfg = _load(config, workdir, video_dir)
    if reference_dir:
        cfg.consistency.reference_dir = reference_dir
    from .stages import screen

    _print_summary(screen.run(cfg, out=out))


@run_app.command("all")
def run_all(config: Optional[str] = typer.Option(None), workdir: Optional[str] = None,
            video_dir: Optional[list[str]] = typer.Option(None),
            reference_dir: Optional[str] = typer.Option(
                None, help="reference images (one subdir per subject) for "
                           "consistency checks")):
    """Run the full pipeline end to end on the configured video dirs."""
    cfg = _load(config, workdir, video_dir)
    if reference_dir:
        cfg.consistency.reference_dir = reference_dir
    from .stages import annotate

    for stage in STAGES:
        console.rule(f"[bold]stage: {stage}")
        if stage == "annotate":
            summary = annotate.run(cfg, auto=True)
        else:
            summary = _dispatch(stage, cfg)
        _print_summary(summary)
    # build the dashboard from the run artifacts
    from dashboard.build import build_dashboard

    dash = build_dashboard(cfg.workdir, Path(cfg.workdir) / "dashboard.html")
    console.rule("[bold green]pipeline complete")
    console.print(f"artifacts in [cyan]{cfg.workdir}[/cyan]")
    console.print(f"dashboard: [cyan]{dash}[/cyan]")


@app.command("samples")
def make_samples(out: str = typer.Option("samples", help="output directory")):
    """(Re)synthesize the ffmpeg sample clips + sidecars."""
    from samples.make_samples import build_samples

    built = build_samples(out)
    console.print(f"built {len(built)} clips into {out}")


@app.command("validate")
def validate(path: str = typer.Argument(..., help="annotation or inference JSON/JSONL"),
             kind: str = typer.Option("annotation", help="annotation|inference")):
    """Validate a records file against the taxonomy schema."""
    from .schema import AnnotationRecord, InferenceRecord
    from .utils.io import read_json, read_jsonl

    model = AnnotationRecord if kind == "annotation" else InferenceRecord
    p = Path(path)
    rows = read_jsonl(p) if p.suffix == ".jsonl" else read_json(p)
    if isinstance(rows, dict):
        rows = rows.get("records", [rows])
    n_ok = 0
    for r in rows:
        model.model_validate(r)
        n_ok += 1
    console.print(f"[green]OK[/green] {n_ok} {kind} records valid")


@app.command("dashboard")
def dashboard(workdir: str = typer.Option("runs/default"),
              out: Optional[str] = typer.Option(None)):
    """Build the single-file HTML dashboard from run artifacts."""
    from dashboard.build import build_dashboard

    dest = build_dashboard(workdir, out)
    console.print(f"dashboard written to [cyan]{dest}[/cyan]")


if __name__ == "__main__":
    app()

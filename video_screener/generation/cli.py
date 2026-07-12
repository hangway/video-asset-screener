"""Typer commands for local ComfyUI image and video generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .comfyui import ComfyUIBackend, ComfyUIConnection
from .contracts import (
    GenerationResult,
    GenerationStatus,
    ImageGenerationRequest,
    ModelSpec,
    ReferenceAsset,
    VideoGenerationRequest,
    WorkflowSpec,
)


app = typer.Typer(help="Generate image or video candidates with local ComfyUI")


def _parse_references(values: Optional[list[str]]) -> tuple[ReferenceAsset, ...]:
    references: list[ReferenceAsset] = []
    for value in values or []:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise typer.BadParameter(
                f"reference {value!r} must use ROLE=PATH format"
            )
        references.append(ReferenceAsset(path=Path(raw_path), role=role))
    return tuple(references)


def _backend(
    workflow: Path,
    bindings: Path,
    output_dir: Path,
    endpoint: str,
    api_prefix: str,
    timeout_seconds: float,
) -> ComfyUIBackend:
    connection = ComfyUIConnection(
        base_url=endpoint,
        api_prefix=api_prefix,
        job_timeout_seconds=timeout_seconds,
    )
    return ComfyUIBackend.from_files(
        workflow, bindings, output_dir, connection=connection
    )


def _model_spec(model_id: str, model_revision: str, model_license: str) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        revision=model_revision,
        license=model_license,
    )


def _workflow_spec(workflow: Path, workflow_id: Optional[str], version: str):
    return WorkflowSpec(
        workflow_id=workflow_id or workflow.stem,
        version=version,
    )


def _print_result(result: GenerationResult) -> None:
    typer.echo(f"status: {result.status.value}")
    typer.echo(f"manifest: {result.manifest_path}")
    for artifact in result.artifacts:
        typer.echo(
            f"candidate {artifact.candidate_index}: {artifact.path} "
            f"(seed {artifact.seed})"
        )
    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)
    if result.error:
        typer.echo(f"error: {result.error}", err=True)


@app.command("image")
def generate_image(
    workflow: Path = typer.Option(..., exists=True, dir_okay=False),
    bindings: Path = typer.Option(..., exists=True, dir_okay=False),
    prompt: str = typer.Option(...),
    model_id: str = typer.Option(...),
    model_revision: str = typer.Option(...),
    model_license: str = typer.Option(..., "--model-license"),
    output_dir: Path = typer.Option(Path("runs/generation"), "--out"),
    workflow_id: Optional[str] = typer.Option(None),
    workflow_version: str = typer.Option("1"),
    negative_prompt: str = typer.Option(""),
    width: int = typer.Option(1024, min=64, max=16384),
    height: int = typer.Option(1024, min=64, max=16384),
    seed: Optional[int] = typer.Option(None, min=0),
    candidates: int = typer.Option(1, min=1, max=8),
    reference: Optional[list[str]] = typer.Option(
        None, help="Repeat ROLE=PATH for bound reference images"
    ),
    endpoint: str = typer.Option("http://127.0.0.1:8188"),
    api_prefix: str = typer.Option(""),
    timeout_seconds: float = typer.Option(1800.0, min=1.0),
) -> None:
    """Generate one or more image candidates with a local workflow."""
    values = {
        "model": _model_spec(model_id, model_revision, model_license),
        "workflow": _workflow_spec(workflow, workflow_id, workflow_version),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "candidate_count": candidates,
        "references": _parse_references(reference),
        "metadata": {
            "workflow_path": str(workflow.resolve()),
            "bindings_path": str(bindings.resolve()),
        },
    }
    if seed is not None:
        values["seed"] = seed
    request = ImageGenerationRequest.model_validate(values)
    result = _backend(
        workflow,
        bindings,
        output_dir,
        endpoint,
        api_prefix,
        timeout_seconds,
    ).generate_image(request)
    _print_result(result)
    if result.status == GenerationStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("video")
def generate_video(
    workflow: Path = typer.Option(..., exists=True, dir_okay=False),
    bindings: Path = typer.Option(..., exists=True, dir_okay=False),
    prompt: str = typer.Option(...),
    model_id: str = typer.Option(...),
    model_revision: str = typer.Option(...),
    model_license: str = typer.Option(..., "--model-license"),
    output_dir: Path = typer.Option(Path("runs/generation"), "--out"),
    workflow_id: Optional[str] = typer.Option(None),
    workflow_version: str = typer.Option("1"),
    negative_prompt: str = typer.Option(""),
    width: int = typer.Option(1280, min=64, max=16384),
    height: int = typer.Option(720, min=64, max=16384),
    duration_seconds: float = typer.Option(5.0, min=0.1),
    fps: float = typer.Option(24.0, min=1.0, max=240.0),
    seed: Optional[int] = typer.Option(None, min=0),
    candidates: int = typer.Option(1, min=1, max=8),
    first_frame: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False
    ),
    last_frame: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False
    ),
    reference: Optional[list[str]] = typer.Option(
        None, help="Repeat ROLE=PATH for additional bound references"
    ),
    endpoint: str = typer.Option("http://127.0.0.1:8188"),
    api_prefix: str = typer.Option(""),
    timeout_seconds: float = typer.Option(1800.0, min=1.0),
) -> None:
    """Generate one or more video candidates with a local workflow."""
    values = {
        "model": _model_spec(model_id, model_revision, model_license),
        "workflow": _workflow_spec(workflow, workflow_id, workflow_version),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "candidate_count": candidates,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "references": _parse_references(reference),
        "metadata": {
            "workflow_path": str(workflow.resolve()),
            "bindings_path": str(bindings.resolve()),
        },
    }
    if seed is not None:
        values["seed"] = seed
    request = VideoGenerationRequest.model_validate(values)
    result = _backend(
        workflow,
        bindings,
        output_dir,
        endpoint,
        api_prefix,
        timeout_seconds,
    ).generate_video(request)
    _print_result(result)
    if result.status == GenerationStatus.FAILED:
        raise typer.Exit(code=1)

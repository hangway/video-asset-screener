"""Typer commands for validating and inspecting portable workflow graphs."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .schema import load_workflow


app = typer.Typer(help="Validate and inspect multi-engine workflow graphs")
console = Console()


def _load(path: Path):
    try:
        return load_workflow(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"invalid workflow: {exc}", param_hint="path") from None


@app.command("validate")
def validate_workflow(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Validate graph connections, types, inputs, and execution order."""
    workflow = _load(path)
    console.print(f"[green]OK[/green] {workflow.name} v{workflow.version}")
    console.print(f"nodes: {len(workflow.nodes)}  edges: {len(workflow.edges)}")
    console.print(f"execution: {' -> '.join(workflow.execution_order())}")
    engines = workflow.required_engines()
    models = workflow.required_models()
    console.print(f"engines: {', '.join(engines) if engines else 'none'}")
    console.print(f"models: {', '.join(models) if models else 'declared at runtime'}")


@app.command("inspect")
def inspect_workflow(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Show the validated execution plan and publication-facing slots."""
    workflow = _load(path)
    nodes = {node.id: node for node in workflow.nodes}
    table = Table(
        title=f"{workflow.name} v{workflow.version}",
        box=None,
        pad_edge=False,
    )
    table.add_column("Step", justify="right")
    table.add_column("Node", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Engine", no_wrap=True)
    for step, node_id in enumerate(workflow.execution_order(), start=1):
        node = nodes[node_id]
        table.add_row(
            str(step),
            f"{node.label}\n[dim]{node.id}\n{node.type}[/dim]",
            node.category.value,
            node.engine or "-",
        )
    console.print(table)

    if workflow.input_slots:
        console.print("[bold]Inputs[/bold]")
        for slot in workflow.input_slots:
            console.print(
                f"  {slot.label} ({slot.type.value}) -> "
                f"{slot.target_node}.{slot.target_port}"
            )
    if workflow.output_slots:
        console.print("[bold]Outputs[/bold]")
        for slot in workflow.output_slots:
            console.print(
                f"  {slot.label} ({slot.type.value}) <- "
                f"{slot.source_node}.{slot.source_port}"
            )

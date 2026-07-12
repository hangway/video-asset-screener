"""Workflow command registration and output smoke tests."""

from pathlib import Path

from typer.testing import CliRunner

from video_screener.cli import app


runner = CliRunner()
WORKFLOW = (
    Path(__file__).parents[1]
    / "configs"
    / "workflows"
    / "vimax-screen-openmontage.json"
)


def test_workflow_validate_command_reports_execution_contract():
    result = runner.invoke(app, ["workflow", "validate", str(WORKFLOW)])

    assert result.exit_code == 0
    assert "Consistent AI Video Production" in result.stdout
    assert "nodes: 9" in result.stdout
    assert "vimax" in result.stdout
    assert "openmontage" in result.stdout


def test_workflow_inspect_command_shows_nodes_and_slots():
    result = runner.invoke(app, ["workflow", "inspect", str(WORKFLOW)])

    assert result.exit_code == 0
    assert "ViMax Story Plan" in result.stdout
    assert "OpenMontage" in result.stdout
    assert "Assembly" in result.stdout
    assert "Creative brief" in result.stdout
    assert "Approved video" in result.stdout


def test_workflow_validation_error_is_presented_as_a_command_error(tmp_path: Path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["workflow", "validate", str(invalid)])

    assert result.exit_code == 2
    assert "invalid workflow" in result.stderr

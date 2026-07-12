"""Generation command registration smoke tests."""

from typer.testing import CliRunner

from video_screener.cli import app


runner = CliRunner()


def test_image_generation_command_is_exposed():
    result = runner.invoke(app, ["generate", "image", "--help"])

    assert result.exit_code == 0
    assert "--workflow" in result.stdout
    assert "--model-license" in result.stdout
    assert "--candidates" in result.stdout


def test_video_generation_command_is_exposed():
    result = runner.invoke(app, ["generate", "video", "--help"])

    assert result.exit_code == 0
    assert "--first-frame" in result.stdout
    assert "--last-frame" in result.stdout
    assert "--duration-seconds" in result.stdout

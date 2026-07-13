"""CLI smoke tests through the installed console script (audit A1+A2).

``pipeline samples`` imports the ``samples`` package at runtime; before the
fix ``samples`` was missing from ``[tool.setuptools] packages``, so the
installed script crashed with ModuleNotFoundError — but only when run from
outside the repo checkout (at the repo root, the cwd shadowed the missing
package). cli.py had zero test coverage, which is how this shipped. These
tests therefore run the real console script from a temp cwd.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.conftest import requires_ffmpeg


def _cli() -> str:
    exe = shutil.which("pipeline")
    if exe is None:
        pytest.skip("console script 'pipeline' not installed (pip install -e .)")
    return exe


def test_cli_help_from_non_repo_cwd(tmp_path):
    res = subprocess.run([_cli(), "--help"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "samples" in res.stdout


@requires_ffmpeg
def test_cli_samples_builds_clips_from_non_repo_cwd(tmp_path):
    out = tmp_path / "built"
    res = subprocess.run([_cli(), "samples", "--out", str(out)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert len(list(out.glob("*.mp4"))) == 8


def test_cli_samples_refuses_overwrite_without_force(tmp_path):
    """Guard fires before any encoding, so this needs no ffmpeg (audit A3)."""
    out = tmp_path / "existing"
    out.mkdir()
    (out / "watermark.mp4").write_bytes(b"\x00")
    res = subprocess.run([_cli(), "samples", "--out", str(out)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode != 0
    assert "calibrated" in (res.stdout + res.stderr)
    assert (out / "watermark.mp4").read_bytes() == b"\x00"

"""Shared pytest fixtures + path helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_MD = REPO_ROOT / "taxonomy.md"
SAMPLES_DIR = REPO_ROOT / "samples"


def have_ffmpeg() -> bool:
    from shutil import which

    return which("ffmpeg") is not None and which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(
    not have_ffmpeg(), reason="ffmpeg/ffprobe not available"
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture(scope="session")
def synth_samples(tmp_path_factory) -> Path:
    """Ensure synthetic samples exist; build into a temp dir if absent.

    Returns a directory containing *.mp4 clips + sidecar JSONs.
    """
    if SAMPLES_DIR.exists() and list(SAMPLES_DIR.glob("*.mp4")):
        return SAMPLES_DIR
    out = tmp_path_factory.mktemp("samples")
    from samples.make_samples import build_samples  # type: ignore

    build_samples(out)
    return out

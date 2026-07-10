"""Dashboard: builds a self-contained HTML from run artifacts."""

from __future__ import annotations

import json
import re

from video_screener.config import PipelineConfig
from video_screener.stages import annotate, dataset, evaluate, ingest, prelabel, screen, train
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_dashboard_builds_self_contained(tmp_path, synth_samples):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(synth_samples)])
    cfg.train.epochs = 8
    ingest.run(cfg); prelabel.run(cfg); annotate.run(cfg, auto=True)
    dataset.run(cfg); train.run(cfg); evaluate.run(cfg); screen.run(cfg)

    from dashboard.build import build_dashboard

    dest = build_dashboard(cfg.workdir, tmp_path / "dashboard.html")
    html = dest.read_text()
    # self-contained: no external resource references
    assert "http://" not in html and "https://" not in html
    # embedded data blob parses and has all stages + assets
    m = re.search(r"/\*__DATA__\*/(.*?)/\*__END__\*/", html, re.S)
    data = json.loads(m.group(1))
    assert len(data["assets"]) == 8
    assert all(data["stages"][s]["done"] for s in
               ["ingest", "prelabel", "annotate", "dataset", "train", "evaluate", "screen"])
    assert data["evaluate"] is not None
    assert data["train"]["history"]
    # dataset viewer + confusion present in the rendered shell
    assert "Dataset viewer" in html
    assert "confusion" in html.lower()


@requires_ffmpeg
def test_dashboard_handles_partial_run(tmp_path, synth_samples):
    """Dashboard builds even when only early stages have run."""
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(synth_samples)])
    ingest.run(cfg)
    from dashboard.build import build_dashboard

    dest = build_dashboard(cfg.workdir, tmp_path / "d.html")
    html = dest.read_text()
    m = re.search(r"/\*__DATA__\*/(.*?)/\*__END__\*/", html, re.S)
    data = json.loads(m.group(1))
    assert data["stages"]["ingest"]["done"] is True
    assert data["stages"]["train"]["done"] is False
    assert data["evaluate"] is None

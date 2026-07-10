"""Stage 7 screen: §7.2 output contract, objective overrides, HTML report."""

from __future__ import annotations

import json

from video_screener.config import PipelineConfig
from video_screener.schema import InferenceRecord
from video_screener.stages import annotate, dataset, ingest, prelabel, screen, train
from tests.conftest import requires_ffmpeg


def _prep(tmp_path, samples_dir) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    cfg.train.epochs = 15
    ingest.run(cfg); prelabel.run(cfg); annotate.run(cfg, auto=True)
    dataset.run(cfg); train.run(cfg)
    return cfg


@requires_ffmpeg
def test_screen_output_contract(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    out = tmp_path / "screened"
    summary = screen.run(cfg, out=str(out), video_dir=[str(synth_samples)])
    assert summary["n_screened"] == 8
    rows = [json.loads(l) for l in (out / "screen_results.jsonl").read_text().splitlines()]
    # every record validates against the §7.2 inference schema
    for r in rows:
        InferenceRecord.model_validate(r)
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["verdict"] in ("PASS", "FIX", "REJECT")
        # §1: a PASS never carries a hard-fail flag
        if r["verdict"] == "PASS":
            assert r["hard_fail_flags"] == []


@requires_ffmpeg
def test_screen_objective_delivery_failure(tmp_path, synth_samples):
    """corrupted + tooshort are REJECTed by the objective rule (before model)."""
    cfg = _prep(tmp_path, synth_samples)
    out = tmp_path / "screened"
    screen.run(cfg, out=str(out), video_dir=[str(synth_samples)])
    by = {json.loads(l)["asset_id"]: json.loads(l)
          for l in (out / "screen_results.jsonl").read_text().splitlines()}
    for aid in ("corrupted", "tooshort"):
        assert by[aid]["verdict"] == "REJECT"
        assert "delivery_failure" in by[aid]["hard_fail_flags"]
        assert "delivery_failure" in by[aid]["primary_reasons"]


@requires_ffmpeg
def test_screen_clean_pass_and_routing(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    out = tmp_path / "screened"
    summary = screen.run(cfg, out=str(out), video_dir=[str(synth_samples)])
    by = {json.loads(l)["asset_id"]: json.loads(l)
          for l in (out / "screen_results.jsonl").read_text().splitlines()}
    # a clean, sharp, well-exposed clip routes to PASS
    assert by["clean_pass"]["verdict"] == "PASS"
    # routing.json counts sum to n
    routing = json.loads((out / "routing.json").read_text())
    assert sum(routing["counts"].values()) == summary["n_screened"]
    assert set(routing["queues"]) == {"PASS", "FIX", "REJECT"}


@requires_ffmpeg
def test_screen_fix_carries_actions(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    out = tmp_path / "screened"
    screen.run(cfg, out=str(out), video_dir=[str(synth_samples)])
    rows = [json.loads(l) for l in (out / "screen_results.jsonl").read_text().splitlines()]
    for r in rows:
        if r["verdict"] == "FIX":
            assert r["fix_actions"], f"{r['asset_id']} FIX must suggest fix_actions"


@requires_ffmpeg
def test_screen_html_report_self_contained(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    out = tmp_path / "screened"
    screen.run(cfg, out=str(out), video_dir=[str(synth_samples)])
    html = (out / "screen_report.html").read_text()
    assert "<!doctype html>" in html.lower()
    assert "PASS" in html and "REJECT" in html
    # single-file: no external resource references
    assert "http://" not in html and "https://" not in html


@requires_ffmpeg
def test_screen_single_corrupted_folder(tmp_path, synth_samples):
    """Screening a folder with only a corrupted clip -> objective REJECT."""
    cfg = _prep(tmp_path, synth_samples)
    onlydir = tmp_path / "only"
    onlydir.mkdir()
    import shutil
    shutil.copy(synth_samples / "corrupted.mp4", onlydir / "corrupted.mp4")
    out = tmp_path / "screened_one"
    summary = screen.run(cfg, out=str(out), video_dir=[str(onlydir)])
    assert summary["n_screened"] == 1
    rec = json.loads((out / "screen_results.jsonl").read_text().splitlines()[0])
    assert rec["verdict"] == "REJECT"
    assert rec["hard_fail_flags"] == ["delivery_failure"]

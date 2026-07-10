"""Stage 3 annotate: session guardrails, auto mode, headless TUI drive."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from video_screener.config import PipelineConfig
from video_screener.schema import AnnotationRecord
from video_screener.stages import annotate, ingest, prelabel
from tests.conftest import requires_ffmpeg


def _prep(tmp_path, samples_dir) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    ingest.run(cfg)
    prelabel.run(cfg)
    return cfg


# ------------------------- pure session guardrails ------------------------
@requires_ffmpeg
def test_toggle_flag_forces_reject_with_evidence(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    s = annotate.AnnotationSession(cfg)
    s.go_to(0)
    s.toggle_flag("watermark_contamination")
    rec = s.current()
    assert rec["verdict"] == "REJECT"
    assert "watermark_contamination" in rec["hard_fail_flags"]
    assert rec["reject_reason"]  # non-empty
    assert any(e["flag_or_dimension"] == "watermark_contamination"
               for e in rec["frame_evidence"])
    ok, err = s.validate()
    assert ok, err


@requires_ffmpeg
def test_toggle_flag_off_removes_evidence(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    s = annotate.AnnotationSession(cfg)
    s.toggle_flag("severe_ai_artifact")
    s.toggle_flag("severe_ai_artifact")  # off again
    assert s.current()["hard_fail_flags"] == []
    assert all(e["flag_or_dimension"] != "severe_ai_artifact"
               for e in s.current()["frame_evidence"])


@requires_ffmpeg
def test_invalid_flag_rejected(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    s = annotate.AnnotationSession(cfg)
    with pytest.raises(ValueError):
        s.toggle_flag("not_a_real_flag")


@requires_ffmpeg
def test_score_bounds(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    s = annotate.AnnotationSession(cfg)
    with pytest.raises(ValueError):
        s.set_score("sharpness_focus", 7)
    with pytest.raises(ValueError):
        s.set_score("not_a_dim", 2)
    s.set_score("sharpness_focus", 0)
    assert s.current()["scores"]["sharpness_focus"] == 0


@requires_ffmpeg
def test_navigation_bounds(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    s = annotate.AnnotationSession(cfg)
    for _ in range(100):
        s.go_prev()
    assert s.idx == 0
    for _ in range(100):
        s.go_next()
    assert s.idx == len(s) - 1


# ------------------------- auto mode --------------------------------------
@requires_ffmpeg
def test_auto_applies_sidecars_and_all_valid(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    summary = annotate.run(cfg, auto=True)
    assert summary["n_from_sidecar"] == 8
    assert summary["n_invalid"] == 0
    assert summary["n_saved"] == 8
    rows = [json.loads(l) for l in
            (tmp_path / "run" / "annotate" / "annotations.jsonl").read_text().splitlines()]
    by = {r["asset_id"]: r for r in rows}
    # ground truth from sidecars
    assert by["watermark"]["verdict"] == "REJECT"
    assert "watermark_contamination" in by["watermark"]["hard_fail_flags"]
    assert by["clean_pass"]["verdict"] == "PASS"
    assert by["flicker"]["verdict"] == "FIX"
    # every saved record validates against the taxonomy schema
    for r in rows:
        AnnotationRecord.model_validate(r)


# ------------------------- headless TUI -----------------------------------
@requires_ffmpeg
def test_tui_launches_navigates_displays_and_saves(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    session = annotate.AnnotationSession(cfg)
    from tui.annotate_app import AnnotateApp

    async def drive():
        app = AnnotateApp(session)
        async with app.run_test() as pilot:
            await pilot.press("right")       # navigate
            await pilot.press("right")
            assert session.idx == 2
            # thumbnail widget shows rendered content (frames displayed)
            from textual.widgets import Static
            thumb = app.query_one("#thumb", Static)
            assert thumb.render() is not None
            # toggle a flag on current asset -> valid REJECT
            await pilot.press("1")           # watermark_contamination
            assert session.current()["verdict"] == "REJECT"
            await pilot.press("w")           # save
        return True

    assert asyncio.run(drive())

    path = tmp_path / "run" / "annotate" / "annotations.jsonl"
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 8
    for r in rows:                            # all saved records valid
        AnnotationRecord.model_validate(r)


@requires_ffmpeg
def test_thumbnail_renders_non_empty(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    import json as _j
    idx = _j.loads((tmp_path / "run" / "ingest" / "index.json").read_text())
    frame = idx["assets"][0]["sampled_frames"][0]["path"]
    from tui.annotate_app import render_thumbnail

    txt = render_thumbnail(frame, cols=32)
    assert len(str(txt)) > 0
    assert "▀" in str(txt)


@requires_ffmpeg
def test_contact_sheet_builds(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    import json as _j
    idx = _j.loads((tmp_path / "run" / "ingest" / "index.json").read_text())
    a = idx["assets"][0]
    from tui.annotate_app import build_contact_sheet

    dest = tmp_path / "sheet.html"
    build_contact_sheet({"asset_id": a["asset_id"], "verdict": "PASS"},
                        a["sampled_frames"], dest)
    assert dest.exists()
    assert "img" in dest.read_text()

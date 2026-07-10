"""Stage 7 screen: §7.2 output contract, objective overrides, HTML report,
and the unified confidence definition across the three routing paths."""

from __future__ import annotations

import json

import numpy as np
import torch

from video_screener.config import PipelineConfig
from video_screener.schema import InferenceRecord
from video_screener.stages import annotate, dataset, ingest, prelabel, screen, train
from video_screener.stages.screen import _screen_one
from video_screener.taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, VERDICTS
from video_screener.utils.video import FrameInfo, SampleResult, VideoMeta
from tests.conftest import requires_ffmpeg


def _prep(tmp_path, samples_dir) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    cfg.train.epochs = 15
    ingest.run(cfg); prelabel.run(cfg); annotate.run(cfg, auto=True)
    dataset.run(cfg); train.run(cfg)
    return cfg


# ------------------- unified confidence definition (unit) -------------------
# confidence = probability the deciding source assigns to the emitted verdict:
# delivery rule -> 1.0; flag-forced REJECT -> strongest triggered flag sigmoid;
# head routing -> head softmax of the emitted verdict.

class _StubEncoder:
    def encode_paths(self, paths):
        return np.zeros((len(paths), 8), dtype=np.float32)


class _StubModel:
    """Fixed-output stand-in for MultiTaskScreener, one clip per call."""

    def __init__(self, verdict_logits, flag_probs, thresh_probs):
        self._out = {
            "verdict_logits": torch.tensor([verdict_logits]),
            "dim_thresh_probs": {d: torch.tensor([thresh_probs]) for d in DIMENSIONS},
            "flag_probs": torch.tensor([flag_probs]),
        }

    def __call__(self, x, mask):
        return self._out


def _ok_sample(n: int = 4) -> SampleResult:
    frames = [FrameInfo(index=i, time_sec=0.5 * i, path=f"f{i}.jpg") for i in range(n)]
    return SampleResult(frames=frames, decode_ok=True, n_decoded=n)


def _meta(duration: float = 5.0) -> VideoMeta:
    return VideoMeta(path="x.mp4", duration_sec=duration, probe_ok=True)


LEVEL4 = [0.99, 0.98, 0.97, 0.96]  # all thresholds cleared -> dim score 4
NO_FLAGS = [0.05] * len(HARD_FAIL_FLAGS)


def test_confidence_delivery_rule_is_one(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    bad = SampleResult(frames=[], decode_ok=False, n_decoded=0)
    rec = _screen_one(None, None, _meta(), bad, cfg, "corrupt")
    assert rec.verdict == "REJECT"
    assert rec.confidence == 1.0
    assert rec.needs_human_review is False


def test_confidence_flag_reject_is_strongest_flag_sigmoid(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    flags = list(NO_FLAGS)
    flags[0], flags[1] = 0.7, 0.9  # two triggered flags; strongest = 0.9
    # head is (wrongly) confident in PASS: flag path must NOT use head probs
    model = _StubModel([8.0, 0.0, 0.0], flags, LEVEL4)
    rec = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "flagged")
    assert rec.verdict == "REJECT"
    assert abs(rec.confidence - 0.9) < 1e-6
    assert set(rec.hard_fail_flags) == {HARD_FAIL_FLAGS[0], HARD_FAIL_FLAGS[1]}


def test_confidence_head_routing_is_softmax_of_emitted_verdict(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    logits = [2.0, 1.0, 0.0]
    model = _StubModel(logits, NO_FLAGS, LEVEL4)
    rec = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "clean")
    assert rec.verdict == "PASS"
    expected = float(torch.softmax(torch.tensor(logits), dim=-1)[VERDICTS.index("PASS")])
    assert abs(rec.confidence - expected) < 1e-6


def test_confidence_pass_gate_downgrade_uses_emitted_verdict(tmp_path):
    """When the §1 PASS gate downgrades the head's PASS, confidence is the
    head's softmax for the *emitted* (downgraded) verdict."""
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    logits = [2.0, 1.0, 0.0]                 # head argmax = PASS
    sub_gate = [0.9, 0.4, 0.3, 0.1]          # dim level 1: below every gate
    model = _StubModel(logits, NO_FLAGS, sub_gate)
    rec = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "flawed")
    assert rec.verdict == "FIX"              # downgraded, no zero dims
    expected = float(torch.softmax(torch.tensor(logits), dim=-1)[VERDICTS.index("FIX")])
    assert abs(rec.confidence - expected) < 1e-6
    assert rec.needs_human_review is True


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

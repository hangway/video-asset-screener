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
    rec, _ = _screen_one(None, None, _meta(), bad, cfg, "corrupt")
    assert rec.verdict == "REJECT"
    assert rec.confidence == 1.0
    assert rec.needs_human_review is False


def test_confidence_flag_reject_is_strongest_flag_sigmoid(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    flags = list(NO_FLAGS)
    flags[0], flags[1] = 0.7, 0.9  # two triggered flags; strongest = 0.9
    # head is (wrongly) confident in PASS: flag path must NOT use head probs
    model = _StubModel([8.0, 0.0, 0.0], flags, LEVEL4)
    rec, _ = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "flagged")
    assert rec.verdict == "REJECT"
    assert abs(rec.confidence - 0.9) < 1e-6
    assert set(rec.hard_fail_flags) == {HARD_FAIL_FLAGS[0], HARD_FAIL_FLAGS[1]}


def test_confidence_head_routing_is_softmax_of_emitted_verdict(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    logits = [2.0, 1.0, 0.0]
    model = _StubModel(logits, NO_FLAGS, LEVEL4)
    rec, _ = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "clean")
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
    rec, _ = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "flawed")
    assert rec.verdict == "FIX"              # downgraded, no zero dims
    expected = float(torch.softmax(torch.tensor(logits), dim=-1)[VERDICTS.index("FIX")])
    assert abs(rec.confidence - expected) < 1e-6
    assert rec.needs_human_review is True


# ----------------- reference consistency -> flag routing --------------------
from video_screener.consistency import ReferenceIndex, SubjectReferences


class _EmbeddingEncoder:
    """Stub encoder that returns one fixed embedding per requested path."""

    def __init__(self, embeddings):
        self._emb = np.asarray(embeddings, dtype=np.float32)

    def encode_paths(self, paths):
        return self._emb[: len(paths)]


def _ref_index(**subjects) -> ReferenceIndex:
    return ReferenceIndex(encoder_name="stub", subjects={
        name: SubjectReferences(name, [f"{name}.png"], np.asarray(emb, np.float32))
        for name, emb in subjects.items()
    })


def test_consistency_below_threshold_triggers_reference_inconsistency(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))  # threshold 0.5
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    # last frame nearly orthogonal to the reference: best-match cosine 0.3
    c = 0.3
    emb = [[1, 0, 0, 0], [1, 0, 0, 0], [c, np.sqrt(1 - c * c), 0, 0]]
    model = _StubModel([8.0, 0.0, 0.0], NO_FLAGS, LEVEL4)  # head says PASS
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(),
                            _ok_sample(3), cfg, "offmodel", idx)
    assert rec.verdict == "REJECT"
    assert "reference_inconsistency" in rec.hard_fail_flags
    # confidence = similarity deficit of the deciding consistency check
    assert abs(rec.confidence - (1.0 - 0.3)) < 1e-4
    InferenceRecord.model_validate(rec.model_dump())      # §7.2 still valid
    assert cons["below_threshold"] is True
    assert cons["subject"] == "hero"
    assert abs(cons["score"] - 0.3) < 1e-3
    # worst offender is the off-model frame (index 2)
    assert cons["worst_frames"][0]["frame_index"] == 2
    assert abs(cons["worst_frames"][0]["time_sec"] - 1.0) < 1e-6


def test_consistency_above_threshold_leaves_routing_alone(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    emb = [[1, 0, 0, 0], [0.95, 0.1, 0, 0], [0.9, 0.2, 0, 0]]  # all close
    logits = [2.0, 1.0, 0.0]
    model = _StubModel(logits, NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(),
                            _ok_sample(3), cfg, "onmodel", idx)
    assert rec.verdict == "PASS"
    assert rec.hard_fail_flags == []
    expected = float(torch.softmax(torch.tensor(logits), dim=-1)[VERDICTS.index("PASS")])
    assert abs(rec.confidence - expected) < 1e-6           # head still decides
    assert cons["below_threshold"] is False
    InferenceRecord.model_validate(rec.model_dump())


def test_drift_flags_morphing_candidate_for_review_not_reject(tmp_path):
    """High frame-to-frame drift -> needs_human_review, but the verdict and
    flags are untouched (drift never auto-REJECTs)."""
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))  # drift threshold 0.35
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    # both frames similar enough to the reference (0.6 >= 0.5) but the jump
    # between them has cosine distance 0.4 > 0.35 -> morphing candidate
    emb = [[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]]
    model = _StubModel([2.0, 1.0, 0.0], NO_FLAGS, LEVEL4)  # head: PASS
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(),
                            _ok_sample(2), cfg, "morphy", idx)
    assert rec.verdict == "PASS"                  # unchanged
    assert rec.hard_fail_flags == []              # no flag from drift
    assert rec.needs_human_review is True         # review only
    assert cons["drift_exceeds_threshold"] is True
    assert abs(cons["max_drift"] - 0.4) < 1e-3
    assert cons["max_drift_between"]["frame_index"] == 0
    assert abs(cons["max_drift_between"]["time_sec"] - 0.0) < 1e-6
    InferenceRecord.model_validate(rec.model_dump())


def test_low_drift_does_not_request_review(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    emb = [[1, 0, 0, 0], [0.98, 0.05, 0, 0], [0.97, 0.08, 0, 0]]
    model = _StubModel([2.0, 1.0, 0.0], NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(),
                            _ok_sample(3), cfg, "steady", idx)
    assert rec.verdict == "PASS"
    assert rec.needs_human_review is False
    assert cons["drift_exceeds_threshold"] is False
    assert cons["max_drift"] < 0.05


def test_consistency_entry_absent_without_references(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    model = _StubModel([2.0, 1.0, 0.0], NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _StubEncoder(), _meta(), _ok_sample(), cfg, "noref")
    assert cons is None and rec.verdict == "PASS"


def test_consistency_report_doc_structure(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]], villain=[[0.0, 1.0, 0.0, 0.0]])
    entries = [
        {"asset_id": "a", "subject": "hero", "score": 0.9, "per_subject": {},
         "per_frame": [0.9], "worst_frames": [], "below_threshold": False},
        {"asset_id": "b", "subject": "villain", "score": 0.1, "per_subject": {},
         "per_frame": [0.1], "worst_frames": [], "below_threshold": True},
    ]
    doc = screen._build_consistency_doc(entries, idx, cfg)
    assert doc["n_clips"] == 2 and doc["n_below_threshold"] == 1
    assert doc["threshold"] == cfg.consistency.min_reference_similarity
    assert doc["subjects"] == {"hero": 1, "villain": 1}
    assert doc["encoder"] == "stub"
    json.dumps(doc)  # must be JSON-serializable as written


# ---------------- head/tail edge stability -> trim FIX ----------------------
def test_unstable_head_routes_fix_with_trim_head(tmp_path):
    """A statistically unstable head second downgrades PASS -> FIX with a
    trim_head suggestion — never REJECT (§1: trims are allowed fixes)."""
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    # frames at t=0,0.5,...,2.5 (duration 5): head = first two frames, off
    # the reference (sim 0.6, still above the 0.5 flag threshold); body clean
    off, on = [0.6, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    emb = [off, off, on, on, on, on]
    logits = [2.0, 1.0, 0.0]                       # head verdict: PASS
    model = _StubModel(logits, NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(5.0),
                            _ok_sample(6), cfg, "shakyhead", idx)
    assert rec.verdict == "FIX"                    # downgraded, NOT rejected
    assert "trim_head" in rec.fix_actions
    assert "edge_instability_head" in rec.primary_reasons
    assert rec.hard_fail_flags == []               # no flag from edges
    es = cons["edge_stability"]
    assert es["head"]["outlier"] is True and es["tail"]["outlier"] is False
    assert es["trim_suggestions"][0] == {"action": "trim_head",
                                         "suggested_trim_sec": 1.0}
    # confidence: head softmax of the emitted (downgraded) verdict
    expected = float(torch.softmax(torch.tensor(logits), dim=-1)[VERDICTS.index("FIX")])
    assert abs(rec.confidence - expected) < 1e-6
    InferenceRecord.model_validate(rec.model_dump())  # §7.2 (FIX has actions)


def test_unstable_tail_routes_fix_with_trim_tail(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    off, on = [0.6, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    emb = [on, on, on, on, on, off]                # last frame (t=2.5) is off
    model = _StubModel([2.0, 1.0, 0.0], NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(3.0),
                            _ok_sample(6), cfg, "shakytail", idx)
    assert rec.verdict == "FIX"
    assert rec.fix_actions == ["trim_tail"]
    es = cons["edge_stability"]
    assert es["tail"]["outlier"] is True and es["head"]["outlier"] is False
    assert es["trim_suggestions"][0]["action"] == "trim_tail"
    # tail window anchored at the last sample (2.5s): frames {2.0, 2.5} are
    # the clip's last second of content, so the last body frame is 1.5 and
    # the trim runs from there to the 3.0s clip end.
    assert es["tail"]["n_frames"] == 2
    assert abs(es["trim_suggestions"][0]["suggested_trim_sec"] - 1.5) < 1e-6
    InferenceRecord.model_validate(rec.model_dump())


def test_stable_edges_leave_pass_untouched(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    idx = _ref_index(hero=[[1.0, 0.0, 0.0, 0.0]])
    emb = [[1.0, 0.0, 0.0, 0.0]] * 6
    model = _StubModel([2.0, 1.0, 0.0], NO_FLAGS, LEVEL4)
    rec, cons = _screen_one(model, _EmbeddingEncoder(emb), _meta(5.0),
                            _ok_sample(6), cfg, "steadyclip", idx)
    assert rec.verdict == "PASS" and rec.fix_actions == []
    assert cons["edge_stability"]["trim_suggestions"] == []


def test_report_ranking_and_reference_gallery(tmp_path):
    """With a consistency doc, screen_report.html gains a reference gallery
    and a per-subject ranking table ordered best -> worst; without one, the
    report is unchanged (no consistency section)."""
    from PIL import Image

    ref_png = tmp_path / "hero.png"
    Image.new("RGB", (32, 32), (200, 40, 40)).save(ref_png)
    idx = ReferenceIndex(encoder_name="stub", subjects={
        "hero": SubjectReferences("hero", [str(ref_png)],
                                  np.ones((1, 4), np.float32)),
    })
    entries = [
        {"asset_id": "worst_clip", "subject": "hero", "score": 0.2,
         "per_subject": {}, "per_frame": [0.2], "worst_frames": [],
         "below_threshold": True, "drift": [], "max_drift": 0.5,
         "max_drift_between": None, "drift_exceeds_threshold": True,
         "edge_stability": {"trim_suggestions": [
             {"action": "trim_head", "suggested_trim_sec": 1.0}]}},
        {"asset_id": "best_clip", "subject": "hero", "score": 0.9,
         "per_subject": {}, "per_frame": [0.9], "worst_frames": [],
         "below_threshold": False, "drift": [], "max_drift": 0.01,
         "max_drift_between": None, "drift_exceeds_threshold": False},
    ]
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    doc = screen._build_consistency_doc(entries, idx, cfg)
    rows = [{"asset_id": "best_clip", "verdict": "PASS", "confidence": 0.9,
             "hard_fail_flags": [], "scores": {}, "fix_actions": [],
             "primary_reasons": [], "needs_human_review": False, "thumb": ""}]
    routing = {"n": 1, "n_needs_review": 0,
               "counts": {"PASS": 1, "FIX": 0, "REJECT": 0}}

    html = screen._build_report(rows, routing, doc, idx)
    assert "Reference consistency" in html
    assert "data:image/jpeg;base64," in html          # embedded ref gallery
    assert html.index("best_clip</td>") < html.index("worst_clip</td>")  # ranked
    assert "⚑ inconsistent" in html and "⚠ drift" in html
    assert "✂ trim_head 1.0s" in html            # head/tail edge marker
    assert "http://" not in html and "https://" not in html  # self-contained

    plain = screen._build_report(rows, routing)        # no references -> no section
    assert "Reference consistency" not in plain


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

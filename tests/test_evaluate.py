"""Stage 6 evaluate: metric math (synthetic) + consistency check + e2e report."""

from __future__ import annotations

import json

from video_screener.config import PipelineConfig
from video_screener.eval_metrics import (
    consistency_rate,
    dimension_metrics,
    flag_metrics,
    stratified_verdict_accuracy,
    verdict_confusion,
)
from video_screener.stages import annotate, dataset, evaluate, ingest, prelabel, train
from tests.conftest import requires_ffmpeg


# ------------------------------ metric math -------------------------------
def test_verdict_confusion_math():
    y_true = ["PASS", "PASS", "FIX", "REJECT", "REJECT"]
    y_pred = ["PASS", "FIX", "FIX", "REJECT", "PASS"]
    r = verdict_confusion(y_true, y_pred)
    # matrix rows = true (PASS,FIX,REJECT), cols = pred
    assert r["matrix"][0] == [1, 1, 0]   # true PASS -> 1 PASS, 1 FIX
    assert r["matrix"][1] == [0, 1, 0]   # true FIX -> 1 FIX
    assert r["matrix"][2] == [1, 0, 1]   # true REJECT -> 1 PASS, 1 REJECT
    assert abs(r["accuracy"] - 3 / 5) < 1e-9
    # PASS: tp=1, fp=1 (REJECT->PASS), fn=1 -> P=0.5 R=0.5 F1=0.5
    assert abs(r["per_class"]["PASS"]["precision"] - 0.5) < 1e-9
    assert abs(r["per_class"]["PASS"]["recall"] - 0.5) < 1e-9


def test_dimension_metrics_math():
    true = {"sharpness_focus": [4, 2, None, 0]}
    pred = {"sharpness_focus": [4, 3, 2, 1]}
    # skip the None pair -> pairs: (4,4)=0, (2,3)=1, (0,1)=1
    m = dimension_metrics(true, pred)["sharpness_focus"]
    assert m["n"] == 3
    assert abs(m["mae"] - (0 + 1 + 1) / 3) < 1e-9
    assert abs(m["exact_acc"] - 1 / 3) < 1e-9
    assert abs(m["within1_acc"] - 3 / 3) < 1e-9


def test_flag_metrics_math():
    true = [["watermark_contamination"], [], ["delivery_failure"]]
    pred = [["watermark_contamination"], ["watermark_contamination"], []]
    fm = flag_metrics(true, pred)
    wm = fm["per_flag"]["watermark_contamination"]
    assert wm["tp"] == 1 and wm["fp"] == 1 and wm["fn"] == 0
    assert abs(wm["precision"] - 0.5) < 1e-9
    assert wm["recall"] == 1.0
    df = fm["per_flag"]["delivery_failure"]
    assert df["tp"] == 0 and df["fn"] == 1
    assert df["recall"] == 0.0
    assert fm["micro"]["tp"] == 1 and fm["micro"]["fp"] == 1 and fm["micro"]["fn"] == 1


def test_stratified_accuracy():
    strata = ["anime_2d", "anime_2d", "photorealistic"]
    yt = ["PASS", "FIX", "REJECT"]
    yp = ["PASS", "REJECT", "REJECT"]
    s = stratified_verdict_accuracy(strata, yt, yp)
    assert s["anime_2d"]["n"] == 2
    assert abs(s["anime_2d"]["accuracy"] - 0.5) < 1e-9
    assert s["photorealistic"]["accuracy"] == 1.0


def test_consistency_rate_flags_pass_with_zero_and_flag():
    verdicts = ["PASS", "PASS", "REJECT", "PASS"]
    scores = [
        {"sharpness_focus": 3},                  # clean PASS -> consistent
        {"sharpness_focus": 0},                  # PASS with dim=0 -> INCONSISTENT
        {"sharpness_focus": 0},                  # REJECT w/ dim 0 -> consistent
        {"sharpness_focus": 3},                  # PASS but has a flag -> INCONSISTENT
    ]
    flags = [[], [], ["delivery_failure"], ["watermark_contamination"]]
    r = consistency_rate(verdicts, scores, flags)
    assert r["n"] == 4
    assert r["n_inconsistent"] == 2
    assert abs(r["rate"] - 0.5) < 1e-9


# ------------------------------ e2e ---------------------------------------
def _prep(tmp_path, samples_dir) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    cfg.train.epochs = 12
    ingest.run(cfg); prelabel.run(cfg); annotate.run(cfg, auto=True)
    dataset.run(cfg); train.run(cfg)
    return cfg


@requires_ffmpeg
def test_evaluate_e2e_report_structure(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    summary = evaluate.run(cfg, split="test")
    rep = json.loads((tmp_path / "run" / "evaluate" / "eval_report.json").read_text())
    for key in ("verdict", "dimensions", "flags", "stratified", "consistency",
                "worst_failures", "per_clip"):
        assert key in rep
    assert rep["verdict"]["labels"] == ["PASS", "FIX", "REJECT"]
    assert 0.0 <= rep["consistency"]["rate"] <= 1.0
    assert 0.0 <= summary["verdict_accuracy"] <= 1.0
    # dimensions report all 6 canonical dims
    assert set(rep["dimensions"].keys()) == {
        "sharpness_focus", "exposure_dynamic_range", "composition_framing",
        "prompt_fidelity_coherence", "temporal_stability", "motion_quality"}


@requires_ffmpeg
def test_evaluate_all_split_richer(tmp_path, synth_samples):
    """Evaluating on the full set exercises multi-class confusion + strata."""
    cfg = _prep(tmp_path, synth_samples)
    # write an 'all' split file = union of splits for a richer diagnostic view
    ds = tmp_path / "run" / "dataset"
    rows = []
    for s in ("train", "val", "test"):
        rows += [json.loads(l) for l in (ds / f"{s}.jsonl").read_text().splitlines() if l.strip()]
    (ds / "all.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = evaluate.run(cfg, split="all")
    assert summary["n_evaluated"] >= 5   # most clips have frames
    rep = json.loads((tmp_path / "run" / "evaluate" / "eval_report.json").read_text())
    # consistency of the model's own predictions is measured and bounded
    assert 0.0 <= rep["consistency"]["rate"] <= 1.0

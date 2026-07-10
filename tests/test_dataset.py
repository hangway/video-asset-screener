"""Stage 4 dataset: split integrity + leakage checks + balance report."""

from __future__ import annotations

import json

from video_screener.config import PipelineConfig
from video_screener.data.splits import Asset, leakage_pairs, make_splits
from video_screener.stages import annotate, dataset, ingest, prelabel
from tests.conftest import requires_ffmpeg


# --------------------------- unit: split logic ----------------------------
def _assets(n_clusters=10, per_cluster=1, verdicts=None):
    assets = []
    k = 0
    for c in range(n_clusters):
        for _ in range(per_cluster):
            v = (verdicts[k % len(verdicts)] if verdicts else "PASS")
            assets.append(Asset(asset_id=f"a{k}", cluster_id=c, verdict=v))
            k += 1
    return assets


def test_no_cluster_spans_splits():
    # 6 clusters of 3 near-dups each -> a cluster must never split.
    assets = _assets(n_clusters=6, per_cluster=3)
    res = make_splits(assets, 0.6, 0.2, 0.2, seed=7)
    # every cluster's assets share one split
    by_cluster: dict[int, set[str]] = {}
    for a in assets:
        by_cluster.setdefault(a.cluster_id, set()).add(res.assignment[a.asset_id])
    for c, splits in by_cluster.items():
        assert len(splits) == 1, f"cluster {c} leaked across splits {splits}"
    assert leakage_pairs(assets, res.assignment) == []


def test_same_source_id_never_splits():
    assets = [
        Asset("x1", cluster_id=1, source_id="scene_A"),
        Asset("x2", cluster_id=2, source_id="scene_A"),  # diff cluster, same source
        Asset("x3", cluster_id=3, source_id="scene_B"),
    ]
    res = make_splits(assets, 0.6, 0.2, 0.2, seed=1)
    assert res.assignment["x1"] == res.assignment["x2"]  # source-grouped
    assert leakage_pairs(assets, res.assignment) == []


def test_fractions_approx_honored_on_larger_set():
    assets = _assets(n_clusters=50, per_cluster=1)
    res = make_splits(assets, 0.6, 0.2, 0.2, seed=3)
    counts = {"train": 0, "val": 0, "test": 0}
    for s in res.assignment.values():
        counts[s] += 1
    assert 25 <= counts["train"] <= 35        # ~30
    assert counts["val"] >= 5 and counts["test"] >= 5
    assert sum(counts.values()) == 50


def test_deterministic_with_seed():
    assets = _assets(n_clusters=20)
    a = make_splits(assets, seed=42).assignment
    b = make_splits(assets, seed=42).assignment
    assert a == b


# --------------------------- e2e on samples -------------------------------
def _prep(tmp_path, samples_dir) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    ingest.run(cfg)
    prelabel.run(cfg)
    annotate.run(cfg, auto=True)
    return cfg


@requires_ffmpeg
def test_dataset_e2e_no_leakage(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    summary = dataset.run(cfg)
    assert summary["leakage_pairs"] == 0
    doc = json.loads((tmp_path / "run" / "dataset" / "splits.json").read_text())
    # clean_pass and its byte-duplicate must share a split (no near-dup leak)
    assert doc["assignment"]["clean_pass"] == doc["assignment"]["duplicate"]
    # every asset assigned exactly one split
    assert set(doc["assignment"].values()) <= {"train", "val", "test"}
    assert len(doc["assignment"]) == 8


@requires_ffmpeg
def test_balance_report_sums(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    dataset.run(cfg)
    rep = json.loads((tmp_path / "run" / "dataset" / "balance_report.json").read_text())
    # per-split verdict counts sum to overall
    overall = rep["overall"]["verdicts"]
    summed = {"PASS": 0, "FIX": 0, "REJECT": 0}
    for split in ("train", "val", "test"):
        for v, c in rep["per_split"][split]["verdicts"].items():
            summed[v] += c
    assert summed == overall
    # overall n == 8
    assert rep["overall"]["n"] == 8


@requires_ffmpeg
def test_dataset_export_format(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples)
    dataset.run(cfg)
    ds = tmp_path / "run" / "dataset"
    total = 0
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in (ds / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:
            assert "verdict" in r and "frames" in r  # training record shape
        total += len(rows)
    assert total == 8

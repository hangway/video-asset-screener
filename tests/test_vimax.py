"""ViMax adapter: shot discovery against the source-verified layout
(docs/audits/vimax-layout.md), config/CLI plumbing, manifest entries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_screener.config import PipelineConfig, load_config
from video_screener.vimax import VimaxShot, discover_shots, manifest_entry


def _shot_dir(root, idx, scene=None, desc=True, video=True):
    d = root / (f"{scene}/shots/{idx}" if scene else f"shots/{idx}")
    d.mkdir(parents=True, exist_ok=True)
    if video:
        (d / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    if desc:
        (d / "shot_description.json").write_text(json.dumps({
            "idx": idx, "is_last": False, "cam_idx": 0,
            "visual_desc": f"<Alice> waves in shot {idx}",
            "variation_type": "medium",
            "variation_reason": "x",
            "ff_desc": "first frame", "lf_desc": "last frame",
            "ff_vis_char_idxs": [0], "lf_vis_char_idxs": [0],
            "motion_desc": f"Alice waves slowly ({idx})",
            "audio_desc": "[Sound Effect] wind",
        }))
    return d


@pytest.fixture
def vimax_dir(tmp_path):
    """Mimics the audited layout: script2video shots at the root plus an
    idea2video scene_0 nesting, decoys that must be ignored."""
    root = tmp_path / "working_dir"
    _shot_dir(root, 0)
    _shot_dir(root, 2)                        # gap in numbering is legal
    _shot_dir(root, 1, desc=False)            # missing description -> empty prompt
    _shot_dir(root, 0, scene="scene_0")       # idea2video nesting
    (root / "shots" / "not_a_shot").mkdir()   # non-numeric dir ignored
    _shot_dir(root, 7, video=False)           # no video.mp4 -> ignored
    (root / "final_video.mp4").write_bytes(b"x")   # top-level concat ignored
    (root / "shots" / "3").mkdir()            # empty numeric dir ignored
    return root


def test_discover_shots_layout(vimax_dir):
    shots = discover_shots(vimax_dir)
    assert [(s.scene, s.shot_idx) for s in shots] == [
        (None, 0), (None, 1), (None, 2), ("scene_0", 0),
    ]
    by = {s.asset_id: s for s in shots}
    assert set(by) == {"shot_000", "shot_001", "shot_002", "scene_0_shot_000"}
    s0 = by["shot_000"]
    assert s0.prompt == "<Alice> waves in shot 0"
    assert s0.motion_desc.startswith("Alice waves")
    assert s0.variation_type == "medium"
    assert s0.description["cam_idx"] == 0
    assert by["shot_001"].prompt == ""        # missing sidecar tolerated
    assert all(s.video_path.endswith("video.mp4") for s in shots)


def test_discover_shots_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_shots(tmp_path / "nope")


def test_discover_shots_empty_dir_returns_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    assert discover_shots(tmp_path / "empty") == []


def test_corrupt_description_tolerated(tmp_path):
    d = _shot_dir(tmp_path / "wd", 0, desc=False)
    (d / "shot_description.json").write_text("{not json")
    shots = discover_shots(tmp_path / "wd")
    assert len(shots) == 1 and shots[0].prompt == ""


def test_manifest_entry_carries_context():
    s = VimaxShot(shot_idx=4, scene="scene_1", video_path="v.mp4",
                  prompt="<Bob> runs", motion_desc="run", audio_desc="steps",
                  variation_type="large", description={"cam_idx": 2})
    e = manifest_entry(s, "scene_1_shot_004")
    assert e["asset_id"] == "scene_1_shot_004"
    assert e["shot_idx"] == 4 and e["scene"] == "scene_1"
    assert e["prompt"] == "<Bob> runs" and e["description"]["cam_idx"] == 2
    json.dumps(e)                              # sidecar must be serializable


def test_config_vimax_key(tmp_path):
    assert PipelineConfig().vimax.working_dir is None      # default off
    p = tmp_path / "cfg.yaml"
    p.write_text("vimax:\n  working_dir: /runs/vimax\n")
    assert load_config(p).vimax.working_dir == "/runs/vimax"
    p.write_text("vimax:\n  shots_dir: oops\n")            # closed section
    with pytest.raises(Exception):
        load_config(p)


# ----------------- portrait registry -> ReferenceIndex ----------------------
import numpy as np
from PIL import Image

from video_screener.consistency import index_references
from video_screener.models.encoder import DeterministicEncoder
from video_screener.stages.screen import _resolve_reference_dir
from video_screener.vimax import portrait_subjects


def _img(path, seed):
    rng = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.randint(0, 255, (24, 24, 3), dtype=np.uint8)).save(path)


def _vimax_registry_dir(root, stale_paths=False):
    """working_dir with character_portraits/0_hero/{front,side,back}.png and
    the registry json exactly as script2video writes it."""
    pdir = root / "character_portraits" / "0_hero"
    for i, view in enumerate(("front", "side", "back")):
        _img(pdir / f"{view}.png", seed=10 + i)
    base = "/moved/away" if stale_paths else str(root)
    reg = {
        "hero": {
            view: {"path": f"{base}/character_portraits/0_hero/{view}.png",
                   "description": f"A {view} view portrait of hero."}
            for view in ("front", "side", "back")
        }
    }
    (root / "character_portraits_registry.json").write_text(json.dumps(reg))
    return root


def test_registry_maps_to_same_index_as_flat_dir(tmp_path):
    wd = _vimax_registry_dir(tmp_path / "wd")
    flat = tmp_path / "flat" / "hero"
    for i, view in enumerate(("front", "side", "back")):
        _img(flat / f"{view}.png", seed=10 + i)      # same bytes as registry

    enc = DeterministicEncoder(feature_dim=64)
    idx_reg = index_references(wd, enc, tmp_path / "c1")
    idx_flat = index_references(tmp_path / "flat", enc, tmp_path / "c2")
    assert set(idx_reg.subjects) == set(idx_flat.subjects) == {"hero"}
    assert np.allclose(idx_reg.subjects["hero"].embeddings,
                       idx_flat.subjects["hero"].embeddings)
    assert [Path(p).name for p in idx_reg.subjects["hero"].paths] == \
           [Path(p).name for p in idx_flat.subjects["hero"].paths]


def test_registry_stale_paths_rerooted(tmp_path):
    wd = _vimax_registry_dir(tmp_path / "wd", stale_paths=True)
    subjects = portrait_subjects(wd)
    assert set(subjects) == {"hero"} and len(subjects["hero"]) == 3
    assert all(p.is_file() for p in subjects["hero"])


def test_structural_fallback_without_registry_json(tmp_path):
    wd = tmp_path / "wd"
    _img(wd / "character_portraits" / "0_hero" / "front.png", seed=1)
    _img(wd / "character_portraits" / "1_Dr_Vex" / "front.png", seed=2)
    subjects = portrait_subjects(wd)
    assert set(subjects) == {"hero", "Dr_Vex"}       # idx prefix stripped


def test_flat_dirs_not_misdetected_as_vimax(tmp_path):
    flat = tmp_path / "flat"
    _img(flat / "hero" / "a.png", seed=1)
    assert portrait_subjects(flat) is None            # existing behaviour holds
    enc = DeterministicEncoder(feature_dim=64)
    idx = index_references(flat, enc, tmp_path / "cache")
    assert set(idx.subjects) == {"hero"}


def test_screen_auto_uses_vimax_registry(tmp_path):
    wd = _vimax_registry_dir(tmp_path / "wd")
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    cfg.vimax.working_dir = str(wd)
    assert _resolve_reference_dir(cfg) == str(wd)     # auto-detected
    cfg.consistency.reference_dir = str(tmp_path / "explicit")
    assert _resolve_reference_dir(cfg) == str(tmp_path / "explicit")  # wins
    plain = PipelineConfig(workdir=str(tmp_path / "run2"))
    assert _resolve_reference_dir(plain) is None


# --------------------------- vimax preset config ----------------------------
def test_vimax_preset_loads_and_scopes_changes(repo_root):
    """configs/vimax.yaml only tunes sampling density + edge sensitivity;
    every gate/threshold that encodes taxonomy policy matches the defaults
    (global defaults themselves are untouched by the preset's existence)."""
    preset = load_config(repo_root / "configs" / "vimax.yaml")
    default = PipelineConfig()
    # the intended deltas
    assert preset.ingest.short_clip_threshold_sec == 9.0
    assert preset.consistency.edge_outlier_sigma == 2.5
    # policy-bearing settings stay identical to the defaults
    assert preset.gate_min == default.gate_min
    assert preset.consistency.min_reference_similarity == \
        default.consistency.min_reference_similarity
    assert preset.consistency.max_frame_drift == default.consistency.max_frame_drift
    assert preset.prelabel.still_coverage_reject_frac == \
        default.prelabel.still_coverage_reject_frac
    assert preset.screen.review_confidence_threshold == \
        default.screen.review_confidence_threshold
    assert preset.metrics_backend == default.metrics_backend
    # and the global defaults were not silently changed by this batch
    assert default.ingest.short_clip_threshold_sec == 4.0
    assert default.consistency.edge_outlier_sigma == 3.0

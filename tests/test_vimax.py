"""ViMax adapter: shot discovery against the source-verified layout
(docs/audits/vimax-layout.md), config/CLI plumbing, manifest entries."""

from __future__ import annotations

import json

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

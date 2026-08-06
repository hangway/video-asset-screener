"""ViMax workspace discovery and cross-shot continuity tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from video_screener.vimax import load_vimax_project, score_vimax_boundaries


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _portrait(path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _shot(root, idx: int, camera: int, visible, rendered: bool = True):
    shot_dir = root / "shots" / str(idx)
    shot_dir.mkdir(parents=True, exist_ok=True)
    _write_json(shot_dir / "shot_description.json", {
        "idx": idx,
        "cam_idx": camera,
        "ff_vis_char_idxs": visible,
        "lf_vis_char_idxs": list(reversed(visible)),
        "visual_desc": f"shot {idx}",
        "motion_desc": f"motion {idx}",
    })
    _portrait(shot_dir / "first_frame.png", (20 + idx, 30, 40))
    if idx == 0:
        _portrait(shot_dir / "last_frame.png", (40, 50, 60))
    if rendered:
        (shot_dir / "video.mp4").write_bytes(b"rendered")
    return shot_dir


def test_load_vimax_project_discovers_nested_shots_and_assigned_characters(tmp_path):
    root = tmp_path / "idea2video"
    scene = root / "scene_0"
    alice = root / "character_portraits" / "0_Alice" / "front.png"
    bob = root / "character_portraits" / "1_Bob" / "front.png"
    _portrait(alice, (200, 30, 30))
    _portrait(bob, (30, 30, 200))
    _write_json(root / "characters.json", [
        {"idx": 0, "identifier_in_scene": "Alice"},
        {"idx": 1, "identifier_in_scene": "Bob"},
    ])
    _write_json(root / "character_portraits_registry.json", {
        "Alice": {"front": {"path": str(alice), "description": "Alice"}},
        "Bob": {"front": {"path": str(bob), "description": "Bob"}},
    })
    _shot(scene, 0, camera=2, visible=[0, 1])
    _shot(scene, 1, camera=2, visible=[1], rendered=False)

    project = load_vimax_project(root)

    assert [shot.asset_id for shot in project.shots] == [
        "scene_0__shot_0000", "scene_0__shot_0001"
    ]
    assert project.shots[0].expected_subjects == ("Alice", "Bob")
    assert project.shots[0].camera_idx == 2
    assert project.shots[0].first_frame_path.is_file()
    assert project.shots[0].last_frame_path.is_file()
    assert [shot.shot_idx for shot in project.rendered_shots] == [0]
    assert [shot.shot_idx for shot in project.missing_video_shots] == [1]
    assert set(project.reference_paths) == {"Alice", "Bob"}
    assert project.shot_for_video(project.shots[0].video_path) == project.shots[0]


def test_load_vimax_project_requires_shot_artifacts(tmp_path):
    with pytest.raises(ValueError, match="shot_description.json"):
        load_vimax_project(tmp_path)


def test_missing_character_artifact_keeps_an_explicit_review_target(tmp_path):
    root = tmp_path / "script2video"
    _shot(root, 0, camera=0, visible=[4])

    project = load_vimax_project(root)

    assert project.shots[0].expected_subjects == ("character_4",)
    assert any("unknown character index 4" in warning for warning in project.warnings)


def test_same_camera_boundary_is_flagged_but_camera_cut_is_not(tmp_path):
    root = tmp_path / "script2video"
    _write_json(root / "characters.json", [
        {"idx": 0, "identifier_in_scene": "Alice"}
    ])
    _shot(root, 0, camera=0, visible=[0])
    _shot(root, 1, camera=0, visible=[0])
    _shot(root, 2, camera=1, visible=[0])
    project = load_vimax_project(root)
    by_idx = {shot.shot_idx: shot for shot in project.shots}

    endpoints = {
        by_idx[0].asset_id: {"head": np.array([1.0, 0.0]),
                             "tail": np.array([1.0, 0.0])},
        by_idx[1].asset_id: {"head": np.array([0.2, np.sqrt(0.96)]),
                             "tail": np.array([1.0, 0.0])},
        by_idx[2].asset_id: {"head": np.array([0.0, 1.0]),
                             "tail": np.array([0.0, 1.0])},
    }
    boundaries = score_vimax_boundaries(project.shots, endpoints, threshold=0.5)

    assert len(boundaries) == 2
    assert boundaries[0]["same_camera"] is True
    assert boundaries[0]["similarity"] == pytest.approx(0.2)
    assert boundaries[0]["below_threshold"] is True
    assert boundaries[1]["same_camera"] is False
    assert boundaries[1]["below_threshold"] is False

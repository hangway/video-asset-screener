"""Adapter for screening rendered shots from a ViMax working directory.

ViMax records production intent that a generic video folder loses: shot order,
camera assignment, visible character indices, generated keyframes, and the
character portrait registry. This module reads those artifacts without
importing ViMax or depending on its runtime.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ViMaxShot:
    """One planned ViMax shot and the artifacts needed to screen it."""

    sequence_id: str
    shot_idx: int
    camera_idx: int | None
    asset_id: str
    video_path: Path
    first_frame_path: Path | None
    last_frame_path: Path | None
    expected_subjects: tuple[str, ...] = ()
    visual_description: str = ""
    motion_description: str = ""

    @property
    def rendered(self) -> bool:
        return self.video_path.is_file()

    def report_metadata(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "shot_idx": self.shot_idx,
            "camera_idx": self.camera_idx,
            "expected_subjects": list(self.expected_subjects),
            "visual_description": self.visual_description,
            "motion_description": self.motion_description,
            "first_frame_path": (
                str(self.first_frame_path) if self.first_frame_path else None
            ),
            "last_frame_path": (
                str(self.last_frame_path) if self.last_frame_path else None
            ),
        }


@dataclass
class ViMaxProject:
    """Screening view of one ViMax run or session directory."""

    root: Path
    shots: list[ViMaxShot] = field(default_factory=list)
    reference_paths: dict[str, list[Path]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def rendered_shots(self) -> list[ViMaxShot]:
        return [shot for shot in self.shots if shot.rendered]

    @property
    def missing_video_shots(self) -> list[ViMaxShot]:
        return [shot for shot in self.shots if not shot.rendered]

    def shot_for_video(self, path: str | Path) -> ViMaxShot | None:
        key = _path_key(Path(path))
        return next(
            (shot for shot in self.shots if _path_key(shot.video_path) == key),
            None,
        )


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ViMax artifact {path}: {exc}") from exc


def _find_upwards(start: Path, stop: Path, name: str) -> Path | None:
    current = start.resolve()
    stop = stop.resolve()
    while True:
        candidate = current / name
        if candidate.is_file():
            return candidate
        if current == stop or stop not in current.parents:
            return None
        current = current.parent


def _sequence_id(sequence_root: Path, project_root: Path) -> str:
    try:
        relative = sequence_root.relative_to(project_root).as_posix()
    except ValueError:
        relative = sequence_root.name
    if relative in ("", "."):
        relative = sequence_root.name
    value = relative.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")
    return value or "vimax"


def _character_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"ViMax characters artifact must be a list: {path}")
    names: dict[int, str] = {}
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        idx = item.get("idx", item.get("index", position))
        identifier = (
            item.get("identifier_in_scene")
            or item.get("identifier_in_event")
            or item.get("identifier_in_novel")
        )
        if identifier is not None:
            names[int(idx)] = str(identifier)
    return names


def _resolve_artifact_path(raw: str, base: Path, root: Path) -> Path | None:
    supplied = Path(raw)
    candidates = [supplied] if supplied.is_absolute() else [
        base / supplied,
        root / supplied,
        Path.cwd() / supplied,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Registry paths are often relative to ViMax's original launch directory.
    parts = supplied.parts
    for count in range(min(6, len(parts)), 0, -1):
        candidate = root.joinpath(*parts[-count:])
        if candidate.is_file():
            return candidate.resolve()

    matches = list(root.rglob(supplied.name)) if supplied.name else []
    files = [match.resolve() for match in matches if match.is_file()]
    return files[0] if len(files) == 1 else None


def _registry_references(root: Path, warnings: list[str]) -> dict[str, list[Path]]:
    references: dict[str, list[Path]] = {}
    for registry_path in sorted(root.rglob("character_portraits_registry.json")):
        raw = _load_json(registry_path)
        if not isinstance(raw, dict):
            warnings.append(f"ignored non-object portrait registry: {registry_path}")
            continue
        for subject, views in raw.items():
            if not isinstance(views, dict):
                continue
            for payload in views.values():
                if not isinstance(payload, dict) or not payload.get("path"):
                    continue
                resolved = _resolve_artifact_path(
                    str(payload["path"]), registry_path.parent, root
                )
                if resolved and resolved.suffix.lower() in IMAGE_EXTS:
                    references.setdefault(str(subject), []).append(resolved)
                else:
                    warnings.append(
                        f"portrait path not found for {subject!r}: {payload['path']}"
                    )
    return references


def _fallback_portraits(root: Path) -> dict[str, list[Path]]:
    """Recover portrait references when a nested ViMax run lacks a registry."""
    references: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if "character_portraits" not in {part.lower() for part in path.parts}:
            continue
        subject = None
        parent_match = re.match(r"^\d+_(.+)$", path.parent.name)
        file_match = re.match(r"^character_\d+_(.+)$", path.stem)
        if parent_match:
            subject = parent_match.group(1)
        elif file_match:
            subject = file_match.group(1)
        if subject:
            references.setdefault(subject, []).append(path.resolve())
    return references


def _merge_paths(*maps: dict[str, list[Path]]) -> dict[str, list[Path]]:
    merged: dict[str, list[Path]] = {}
    for mapping in maps:
        for subject, paths in mapping.items():
            seen = {_path_key(path) for path in merged.get(subject, [])}
            for path in paths:
                key = _path_key(path)
                if key not in seen:
                    merged.setdefault(subject, []).append(path)
                    seen.add(key)
    return {subject: sorted(paths) for subject, paths in sorted(merged.items())}


def load_vimax_project(workdir: str | Path) -> ViMaxProject:
    """Discover rendered shots and continuity targets in a ViMax workspace.

    ``workdir`` may be a direct ``script2video`` directory or a higher-level
    session containing nested Idea2Video/Novel2Video scenes.
    """
    root = Path(workdir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ViMax workdir not found: {root}")

    description_paths = sorted(
        path for path in root.rglob("shot_description.json")
        if path.parent.parent.name == "shots"
    )
    if not description_paths:
        raise ValueError(
            f"no ViMax shots/*/shot_description.json artifacts found under {root}"
        )

    warnings: list[str] = []
    shots: list[ViMaxShot] = []
    character_cache: dict[Path | None, dict[int, str]] = {}

    for description_path in description_paths:
        shot_dir = description_path.parent
        sequence_root = shot_dir.parent.parent
        sequence_id = _sequence_id(sequence_root, root)
        characters_path = _find_upwards(sequence_root, root, "characters.json")
        if characters_path not in character_cache:
            character_cache[characters_path] = _character_names(characters_path)
        names = character_cache[characters_path]

        raw = _load_json(description_path)
        if not isinstance(raw, dict):
            raise ValueError(
                f"ViMax shot description must be an object: {description_path}"
            )
        try:
            shot_idx = int(raw.get("idx", shot_dir.name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ViMax shot index in {description_path}") from exc

        visible_indices: list[int] = []
        for field_name in ("ff_vis_char_idxs", "lf_vis_char_idxs"):
            for value in raw.get(field_name, []) or []:
                idx = int(value)
                if idx not in visible_indices:
                    visible_indices.append(idx)
        expected: list[str] = []
        for idx in visible_indices:
            if idx in names:
                expected.append(names[idx])
            else:
                expected.append(f"character_{idx}")
                warnings.append(
                    f"shot {sequence_id}/{shot_idx} references unknown character index {idx}"
                )

        first = shot_dir / "first_frame.png"
        last = shot_dir / "last_frame.png"
        shots.append(ViMaxShot(
            sequence_id=sequence_id,
            shot_idx=shot_idx,
            camera_idx=(
                int(raw["cam_idx"]) if raw.get("cam_idx") is not None else None
            ),
            asset_id=f"{sequence_id}__shot_{shot_idx:04d}",
            video_path=(shot_dir / "video.mp4").resolve(),
            first_frame_path=first.resolve() if first.is_file() else None,
            last_frame_path=last.resolve() if last.is_file() else None,
            expected_subjects=tuple(expected),
            visual_description=str(raw.get("visual_desc", "")),
            motion_description=str(raw.get("motion_desc", "")),
        ))

    references = _merge_paths(
        _registry_references(root, warnings),
        _fallback_portraits(root),
    )
    shots.sort(key=lambda shot: (shot.sequence_id, shot.shot_idx))
    return ViMaxProject(
        root=root, shots=shots, reference_paths=references, warnings=warnings
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def score_vimax_boundaries(
    shots: list[ViMaxShot],
    endpoint_embeddings: dict[str, dict[str, np.ndarray]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Score adjacent shot boundaries; gate only same-camera transitions."""
    boundaries: list[dict[str, Any]] = []
    by_sequence: dict[str, list[ViMaxShot]] = {}
    for shot in shots:
        by_sequence.setdefault(shot.sequence_id, []).append(shot)

    for sequence_id, sequence_shots in sorted(by_sequence.items()):
        ordered = sorted(sequence_shots, key=lambda shot: shot.shot_idx)
        for previous, current in zip(ordered, ordered[1:]):
            left = endpoint_embeddings.get(previous.asset_id)
            right = endpoint_embeddings.get(current.asset_id)
            if not left or not right or "tail" not in left or "head" not in right:
                continue
            similarity = _cosine(left["tail"], right["head"])
            same_camera = (
                previous.camera_idx is not None
                and previous.camera_idx == current.camera_idx
            )
            boundaries.append({
                "sequence_id": sequence_id,
                "from_asset_id": previous.asset_id,
                "to_asset_id": current.asset_id,
                "from_shot_idx": previous.shot_idx,
                "to_shot_idx": current.shot_idx,
                "from_camera_idx": previous.camera_idx,
                "to_camera_idx": current.camera_idx,
                "same_camera": same_camera,
                "shared_subjects": sorted(
                    set(previous.expected_subjects) & set(current.expected_subjects)
                ),
                "similarity": round(similarity, 4),
                "below_threshold": bool(same_camera and similarity < threshold),
            })
    return boundaries

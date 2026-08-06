"""Reference-consistency support (taxonomy v0.3.1, wired to the existing
``reference_inconsistency`` flag — no new flags, dimensions, or schema fields).

References are still images of the subjects a clip is supposed to depict,
laid out on disk as one subdirectory per subject::

    reference_dir/
      subject_a/ img1.png img2.jpg ...
      subject_b/ ...
      loose.png          # files directly in reference_dir -> subject "default"

``index_references`` embeds every reference image through the SAME pluggable
frozen encoder the pipeline uses for frames (deterministic or CLIP — whatever
``build_encoder`` resolves), so clip frames and references live in one
embedding space. Embeddings are cached on disk keyed by encoder name + a
content digest of the subject's images; editing, adding, or removing an image
invalidates only that subject's cache entry.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# subject name for images placed directly in reference_dir (no subdirectory)
DEFAULT_SUBJECT = "default"


@dataclass
class SubjectReferences:
    subject: str
    paths: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )


@dataclass
class ReferenceIndex:
    """Per-subject reference embeddings, all produced by one encoder."""

    encoder_name: str
    subjects: dict[str, SubjectReferences] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return sum(len(s.paths) for s in self.subjects.values())

    def __bool__(self) -> bool:  # truthy iff there is anything to compare to
        return self.n_images > 0


def _list_subject_images(reference_dir: Path) -> dict[str, list[Path]]:
    """Map subject -> sorted image paths. Subdir name = subject; loose files
    in reference_dir itself belong to ``DEFAULT_SUBJECT``."""
    subjects: dict[str, list[Path]] = {}
    for entry in sorted(reference_dir.iterdir()):
        if entry.is_dir():
            imgs = sorted(
                p for p in entry.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if imgs:
                subjects[entry.name] = imgs
        elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            subjects.setdefault(DEFAULT_SUBJECT, []).append(entry)
    if DEFAULT_SUBJECT in subjects:
        subjects[DEFAULT_SUBJECT].sort()
    return subjects


def _content_digest(paths: list[Path]) -> str:
    """Digest over file names + bytes: any change re-keys the cache entry."""
    h = hashlib.sha1()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class ClipConsistency:
    """Clip-vs-reference consistency for one clip.

    ``score`` follows §5 worst-frame aggregation: per frame take the
    best-match cosine similarity against the subject's references (a frame is
    consistent if it matches ANY reference view), then take the MIN over
    frames (one off-model frame makes the clip inconsistent).
    """

    subject: str                 # best-matching subject
    score: float                 # worst-frame best-match cosine, in [-1, 1]
    per_frame: list[float]       # best-match cosine per (sampled) frame
    per_subject: dict[str, float]  # worst-frame score against every subject
    best_subject: str | None = None  # unconstrained best subject in the index
    expected_subjects: tuple[str, ...] = ()


@dataclass
class KeyframeConsistency:
    """Similarity between clip endpoints and ViMax's generated keyframes."""

    head_similarity: float | None = None
    tail_similarity: float | None = None

    @property
    def score(self) -> float | None:
        values = [
            value for value in (self.head_similarity, self.tail_similarity)
            if value is not None
        ]
        return min(values) if values else None


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def frame_reference_similarity(frame_embeddings: np.ndarray,
                               ref_embeddings: np.ndarray) -> np.ndarray:
    """Best-match cosine per frame: [T,D] x [R,D] -> [T] (max over refs)."""
    sims = _normalize_rows(frame_embeddings) @ _normalize_rows(ref_embeddings).T
    return sims.max(axis=1)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for two individual embedding vectors."""
    a = np.asarray(left, dtype=np.float32).reshape(1, -1)
    b = np.asarray(right, dtype=np.float32).reshape(1, -1)
    return float((_normalize_rows(a) * _normalize_rows(b)).sum())


def score_keyframes(
    frame_embeddings: np.ndarray,
    first_frame_embedding: np.ndarray | None = None,
    last_frame_embedding: np.ndarray | None = None,
) -> KeyframeConsistency | None:
    """Compare actual clip endpoints with planned first/last frame images."""
    if frame_embeddings.size == 0:
        return None
    head = (
        cosine_similarity(frame_embeddings[0], first_frame_embedding)
        if first_frame_embedding is not None else None
    )
    tail = (
        cosine_similarity(frame_embeddings[-1], last_frame_embedding)
        if last_frame_embedding is not None else None
    )
    result = KeyframeConsistency(head_similarity=head, tail_similarity=tail)
    return result if result.score is not None else None


def frame_drift(frame_embeddings: np.ndarray) -> np.ndarray:
    """Within-clip drift: cosine distance between consecutive frames.

    [T,D] -> [T-1]; entry i is the distance between frames i and i+1. Large
    consecutive jumps are morphing/identity-drift candidates. Auxiliary
    signal only: it flags clips for human review, it never auto-REJECTs
    (a hard cut is a legitimate reason for a big jump)."""
    if frame_embeddings.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    n = _normalize_rows(frame_embeddings)
    sims = (n[:-1] * n[1:]).sum(axis=-1)
    return (1.0 - sims).astype(np.float32)


# Floor for the body standard deviation in the edge-outlier test: similarity
# and drift live on a ~[0,1] scale, so deviations below this are sensor noise
# (and a perfectly constant body would otherwise make ANY difference an
# outlier).
_STD_FLOOR = 1e-3


def edge_stability(per_frame_sim: list[float], drift: list[float],
                   times: list[float | None], duration: float | None,
                   window_sec: float = 1.0, sigma: float = 3.0) -> dict | None:
    """Head/tail edge-stability analysis against the clip body.

    Frames are split into head (t < window), tail (t > last_sample - window)
    and body. The tail window is anchored at the last *sampled* timestamp,
    not the container duration: uniform sampling emits frames strictly
    before the duration (an 8 s clip at 1 fps samples t = 0..7), so a
    duration-anchored window (t > duration - window) is empty by
    construction for integer-length clips — the strict '>' lands exactly on
    the last sample and tail analysis silently vanishes (trim_tail could
    never fire on real clips).

    An edge segment is a statistical outlier when its mean per-frame
    reference similarity falls below body_mean - sigma*body_std (similarity:
    lower is worse) OR its mean drift rises above body_mean + sigma*body_std
    (drift: higher is worse). Drift value i (frames i -> i+1) belongs to an
    edge segment when either endpoint does; body baselines use only pairs
    fully inside the body.

    An outlier edge yields a trim suggestion — cut up to the first (head) /
    from the last (tail) body frame — and is meant to route as FIX per §1
    (trims are minor, allowed post-production), never REJECT.

    Returns None when the analysis is impossible: unknown frame times, or no
    body frames to serve as the baseline (clip shorter than ~2 windows).
    """
    if not times or any(t is None for t in times):
        return None
    t = np.asarray(times, dtype=np.float64)
    end = float(duration) if duration else float(t.max())
    last = float(t.max())  # tail anchor: last sampled frame, NOT duration
    head = [i for i in range(len(t)) if t[i] < window_sec]
    tail = [i for i in range(len(t))
            if t[i] > last - window_sec and i not in head]
    body = [i for i in range(len(t)) if i not in head and i not in tail]
    if not body:
        return None

    sim = np.asarray(per_frame_sim, dtype=np.float64)
    dr = np.asarray(drift, dtype=np.float64)

    def pair_vals(idx: list[int], strict: bool) -> np.ndarray:
        s = set(idx)
        if strict:
            keep = [i for i in range(len(dr)) if i in s and i + 1 in s]
        else:
            keep = [i for i in range(len(dr)) if i in s or i + 1 in s]
        return dr[keep]

    body_sim, body_pairs = sim[body], pair_vals(body, strict=True)

    def seg(idx: list[int]) -> dict:
        pv = pair_vals(idx, strict=False)
        return {
            "n_frames": len(idx),
            "sim_mean": round(float(sim[idx].mean()), 4) if idx else None,
            "drift_mean": round(float(pv.mean()), 4) if pv.size else None,
        }

    def outlier(idx: list[int]) -> bool:
        bad = False
        if idx and body_sim.size >= 2:
            sd = max(float(body_sim.std()), _STD_FLOOR)
            bad |= float(sim[idx].mean()) < float(body_sim.mean()) - sigma * sd
        pv = pair_vals(idx, strict=False)
        if pv.size and body_pairs.size >= 2:
            sd = max(float(body_pairs.std()), _STD_FLOOR)
            bad |= float(pv.mean()) > float(body_pairs.mean()) + sigma * sd
        return bool(bad)

    head_out, tail_out = outlier(head), outlier(tail)
    suggestions = []
    if head_out:
        suggestions.append({"action": "trim_head",
                            "suggested_trim_sec": round(float(t[body].min()), 2)})
    if tail_out:
        suggestions.append({"action": "trim_tail",
                            "suggested_trim_sec": round(float(end - t[body].max()), 2)})
    return {
        "window_sec": window_sec,
        "sigma": sigma,
        "head": {**seg(head), "outlier": head_out},
        "tail": {**seg(tail), "outlier": tail_out},
        "body": {
            **seg(body),
            "sim_std": round(float(body_sim.std()), 4) if body_sim.size else None,
            "drift_std": round(float(body_pairs.std()), 4) if body_pairs.size else None,
        },
        "trim_suggestions": suggestions,
    }


def score_clip(
    frame_embeddings: np.ndarray,
    index: ReferenceIndex,
    expected_subjects: list[str] | tuple[str, ...] | None = None,
) -> ClipConsistency | None:
    """Score a clip against reference subjects.

    Generic screening assigns the globally best subject. When ViMax supplies
    ``expected_subjects``, references are restricted to that set and pooled as
    a union per frame. That prevents a wrong character from passing merely
    because it resembles some other portrait in the project, while still
    supporting shots containing multiple expected characters.
    """
    if frame_embeddings.size == 0 or not index:
        return None
    per_subject: dict[str, float] = {}
    per_frame_by_subject: dict[str, np.ndarray] = {}
    for name, refs in index.subjects.items():
        if refs.embeddings.size == 0:
            continue
        pf = frame_reference_similarity(frame_embeddings, refs.embeddings)
        per_frame_by_subject[name] = pf
        per_subject[name] = float(pf.min())     # §5 worst-frame
    if not per_subject:
        return None
    best = max(per_subject, key=per_subject.get)

    if expected_subjects is None:
        selected = (best,)
        per_frame = per_frame_by_subject[best]
        label = best
    else:
        selected = tuple(
            subject for subject in expected_subjects
            if subject in index.subjects
            and index.subjects[subject].embeddings.size > 0
        )
        if not selected:
            return None
        references = np.concatenate(
            [index.subjects[subject].embeddings for subject in selected], axis=0
        )
        per_frame = frame_reference_similarity(frame_embeddings, references)
        label = " + ".join(selected)

    return ClipConsistency(
        subject=label,
        score=float(per_frame.min()),
        per_frame=[float(v) for v in per_frame],
        per_subject=per_subject,
        best_subject=best,
        expected_subjects=selected if expected_subjects is not None else (),
    )


def _safe_cache_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")
    return cleaned or "subject"


def index_reference_paths(
    subject_paths: dict[str, list[str | Path]],
    encoder,
    cache_dir: str | Path,
) -> ReferenceIndex:
    """Embed an explicit subject-to-image map with per-subject caching."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    enc_key = encoder.name.replace(":", "_").replace("/", "_")
    index = ReferenceIndex(encoder_name=encoder.name)
    for subject, supplied_paths in sorted(subject_paths.items()):
        paths = sorted({Path(path).resolve() for path in supplied_paths
                        if Path(path).is_file()
                        and Path(path).suffix.lower() in IMAGE_EXTS})
        if not paths:
            continue
        digest = _content_digest(paths)
        subject_key = _safe_cache_component(subject)
        cache = cache_dir / f"{enc_key}__{subject_key}__{digest}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            emb = encoder.encode_paths([str(p) for p in paths])
            np.save(cache, emb)
        index.subjects[subject] = SubjectReferences(
            subject=subject, paths=[str(p) for p in paths],
            embeddings=emb.astype(np.float32),
        )
    return index


def index_references(reference_dir: str | Path, encoder,
                     cache_dir: str | Path) -> ReferenceIndex:
    """Embed a directory of per-subject reference images, with caching."""
    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference_dir not found: {reference_dir}")
    return index_reference_paths(
        _list_subject_images(reference_dir), encoder, cache_dir
    )


def merge_reference_indexes(*indexes: ReferenceIndex) -> ReferenceIndex:
    """Merge indexes produced by the same encoder, deduplicating image paths."""
    available = [index for index in indexes if index is not None]
    if not available:
        raise ValueError("at least one reference index is required")
    encoder_name = available[0].encoder_name
    if any(index.encoder_name != encoder_name for index in available):
        raise ValueError("cannot merge reference indexes from different encoders")

    merged = ReferenceIndex(encoder_name=encoder_name)
    rows: dict[str, list[tuple[str, np.ndarray]]] = {}
    seen: dict[str, set[str]] = {}
    for index in available:
        for subject, references in index.subjects.items():
            subject_seen = seen.setdefault(subject, set())
            for path, embedding in zip(references.paths, references.embeddings):
                key = os.path.normcase(str(Path(path).resolve()))
                if key not in subject_seen:
                    rows.setdefault(subject, []).append((path, embedding))
                    subject_seen.add(key)

    for subject, items in rows.items():
        merged.subjects[subject] = SubjectReferences(
            subject=subject,
            paths=[path for path, _ in items],
            embeddings=np.stack([embedding for _, embedding in items]).astype(
                np.float32
            ),
        )
    return merged

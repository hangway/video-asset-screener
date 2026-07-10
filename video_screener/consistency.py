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


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def frame_reference_similarity(frame_embeddings: np.ndarray,
                               ref_embeddings: np.ndarray) -> np.ndarray:
    """Best-match cosine per frame: [T,D] x [R,D] -> [T] (max over refs)."""
    sims = _normalize_rows(frame_embeddings) @ _normalize_rows(ref_embeddings).T
    return sims.max(axis=1)


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

    Frames are split into head (t < window), tail (t > end - window) and
    body. An edge segment is a statistical outlier when its mean per-frame
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
    head = [i for i in range(len(t)) if t[i] < window_sec]
    tail = [i for i in range(len(t))
            if t[i] > end - window_sec and i not in head]
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


def score_clip(frame_embeddings: np.ndarray,
               index: ReferenceIndex) -> ClipConsistency | None:
    """Score a clip's frames against every subject; assign the best subject.

    Returns None when there is nothing to compare (no frames or no reference
    images)."""
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
    return ClipConsistency(
        subject=best,
        score=per_subject[best],
        per_frame=[float(v) for v in per_frame_by_subject[best]],
        per_subject=per_subject,
    )


def index_references(reference_dir: str | Path, encoder,
                     cache_dir: str | Path) -> ReferenceIndex:
    """Embed all reference images per subject via ``encoder``, with caching.

    Cache entries live in ``cache_dir`` as
    ``<encoder-name>__<subject>__<content-digest>.npy``; a hit skips the
    encoder entirely. Unreadable images are skipped by the encoder (same
    behaviour as frame encoding); a subject whose images all fail to decode
    is kept with an empty embedding matrix.
    """
    from .vimax import portrait_subjects  # late import: vimax must not import us

    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference_dir not found: {reference_dir}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ViMax portrait registries are auto-detected by structure and mapped to
    # the same subject->images shape; flat / subject-subdir layouts unchanged
    subject_images = portrait_subjects(reference_dir)
    if subject_images is None:
        subject_images = _list_subject_images(reference_dir)

    enc_key = encoder.name.replace(":", "_").replace("/", "_")
    index = ReferenceIndex(encoder_name=encoder.name)
    for subject, paths in subject_images.items():
        digest = _content_digest(paths)
        # ViMax identifiers can carry spaces/odd chars -> sanitize for the
        # cache filename only (the index keeps the original subject name)
        safe_subject = "".join(c if c.isalnum() or c in "-_" else "_"
                               for c in subject)
        cache = cache_dir / f"{enc_key}__{safe_subject}__{digest}.npy"
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

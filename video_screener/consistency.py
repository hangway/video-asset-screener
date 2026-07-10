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
    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference_dir not found: {reference_dir}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    enc_key = encoder.name.replace(":", "_").replace("/", "_")
    index = ReferenceIndex(encoder_name=encoder.name)
    for subject, paths in _list_subject_images(reference_dir).items():
        digest = _content_digest(paths)
        cache = cache_dir / f"{enc_key}__{subject}__{digest}.npy"
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

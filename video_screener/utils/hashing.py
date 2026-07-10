"""Perceptual-hash clip signatures + near-duplicate clustering (§ ingest).

A clip is represented by the perceptual hashes of a few evenly spaced sampled
frames. Two clips are near-duplicates when the mean per-position Hamming
distance between their signatures is <= ``dedup_hamming_threshold``. Exact
byte-duplicates collapse to distance 0; visually distinct clips sit far above
the threshold (empirically >=25 on the synthetic samples).
"""

from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SIG_FRAMES = 4          # frames per clip signature
HASH_SIZE = 8           # phash 8x8 -> 64-bit


def frame_phash(path: str | Path) -> imagehash.ImageHash:
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"), hash_size=HASH_SIZE)


def clip_signature(frame_paths: list[str | Path], k: int = SIG_FRAMES) -> list[str]:
    """Return a k-frame perceptual signature (hex strings) for a clip.

    Frames are chosen at evenly spaced fractional positions so signatures are
    comparable across clips of different lengths."""
    if not frame_paths:
        return []
    n = len(frame_paths)
    if n <= k:
        chosen = list(range(n))
    else:
        chosen = sorted({int(round(f)) for f in np.linspace(0, n - 1, k)})
    return [str(frame_phash(frame_paths[i])) for i in chosen]


def _hex_to_hash(h: str) -> imagehash.ImageHash:
    return imagehash.hex_to_hash(h)


def signature_distance(sig_a: list[str], sig_b: list[str]) -> float:
    """Mean per-position Hamming distance between two signatures.

    Signatures are aligned by index up to the shorter length. Empty vs
    non-empty (e.g. a decode-failed clip) returns infinity (never a dup)."""
    if not sig_a or not sig_b:
        return float("inf")
    m = min(len(sig_a), len(sig_b))
    ha = [_hex_to_hash(x) for x in sig_a[:m]]
    hb = [_hex_to_hash(x) for x in sig_b[:m]]
    dists = [ha[i] - hb[i] for i in range(m)]
    return float(sum(dists)) / m


def cluster_near_duplicates(
    signatures: list[list[str]], threshold: int
) -> list[int]:
    """Union-find clustering. Returns a cluster id per input signature.

    Two clips join a cluster when their signature distance is <= threshold.
    Clips with empty signatures each get their own singleton cluster."""
    n = len(signatures)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        if not signatures[i]:
            continue
        for j in range(i + 1, n):
            if not signatures[j]:
                continue
            if signature_distance(signatures[i], signatures[j]) <= threshold:
                union(i, j)

    # Normalize cluster ids to small contiguous integers.
    roots = {}
    out: list[int] = []
    next_id = 0
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        out.append(roots[r])
    return out

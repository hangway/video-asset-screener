"""Leakage-free train/val/test splitting.

Assets are grouped so that near-duplicates (same ingest perceptual-hash cluster)
and clips sharing an explicit source id never straddle a split boundary. Whole
groups are then distributed across splits to approach the target fractions while
balancing size. Grouping by *content identity* (not by folder) is what prevents
train/test leakage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Asset:
    asset_id: str
    cluster_id: int              # ingest phash near-dup cluster
    source_id: str = ""          # optional explicit source/scene id
    verdict: str = "PASS"


@dataclass
class SplitResult:
    assignment: dict[str, str] = field(default_factory=dict)   # asset_id -> split
    groups: dict[str, list[str]] = field(default_factory=dict)  # group_key -> asset_ids
    group_split: dict[str, str] = field(default_factory=dict)   # group_key -> split


def _union_find_groups(assets: list[Asset]) -> dict[str, list[str]]:
    """Group assets by (cluster_id) and (source_id) via union-find."""
    ids = [a.asset_id for a in assets]
    pos = {aid: i for i, aid in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # union by shared cluster_id
    by_cluster: dict[int, list[int]] = {}
    by_source: dict[str, list[int]] = {}
    for a in assets:
        by_cluster.setdefault(a.cluster_id, []).append(pos[a.asset_id])
        if a.source_id:
            by_source.setdefault(a.source_id, []).append(pos[a.asset_id])
    for members in list(by_cluster.values()) + list(by_source.values()):
        for m in members[1:]:
            union(members[0], m)

    groups: dict[int, list[str]] = {}
    for a in assets:
        r = find(pos[a.asset_id])
        groups.setdefault(r, []).append(a.asset_id)
    # stable string keys
    return {f"g{gi}": sorted(members) for gi, members in enumerate(sorted(groups.values()))}


def make_splits(
    assets: list[Asset],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 1234,
) -> SplitResult:
    groups = _union_find_groups(assets)
    verdict_of = {a.asset_id: a.verdict for a in assets}
    n_total = len(assets)

    # Order groups: largest first (place constrained big groups early), with a
    # seeded shuffle among equal sizes for reproducible variety.
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    targets = {"train": train_frac, "val": val_frac, "test": test_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    group_split: dict[str, str] = {}

    def deficit(split: str) -> float:
        # how far below its proportional target this split is (higher => needier)
        target_assets = targets[split] * n_total
        return target_assets - counts[split]

    for gkey, members in group_items:
        # assign to the neediest split (respecting zero-target splits)
        candidates = [s for s in ("train", "val", "test") if targets[s] > 0]
        split = max(candidates, key=lambda s: (deficit(s), s == "train"))
        group_split[gkey] = split
        counts[split] += len(members)

    assignment: dict[str, str] = {}
    for gkey, members in groups.items():
        for aid in members:
            assignment[aid] = group_split[gkey]

    return SplitResult(assignment=assignment, groups=groups, group_split=group_split)


def leakage_pairs(assets: list[Asset], assignment: dict[str, str]) -> list[tuple[str, str]]:
    """Return asset pairs that share a cluster/source but landed in different
    splits (should always be empty for a valid split)."""
    bad: list[tuple[str, str]] = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            same_group = (a.cluster_id == b.cluster_id) or (
                a.source_id and a.source_id == b.source_id
            )
            if same_group and assignment.get(a.asset_id) != assignment.get(b.asset_id):
                bad.append((a.asset_id, b.asset_id))
    return bad

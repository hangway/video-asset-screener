"""Torch dataset over exported split records + frozen-feature caching.

Each trainable clip (>=1 decoded frame) yields frozen per-frame features and the
taxonomy labels: verdict index, 6 ordinal dimension levels (-1 = N/A, masked in
loss), and a 9-dim multi-hot flag vector. Variable frame counts are padded in
``collate`` with a validity mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, VERDICTS


class ScreenerDataset(Dataset):
    def __init__(self, records: list[dict], encoder, cache_dir: str | Path,
                 max_frames: int = 64):
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        # keep only trainable clips (with frames)
        self.records = [r for r in records if r.get("frames")]

    def __len__(self) -> int:
        return len(self.records)

    def _features(self, rec: dict) -> np.ndarray:
        aid = rec["asset_id"]
        cache = self.cache_dir / f"{self.encoder.name.replace(':', '_')}__{aid}.npy"
        if cache.exists():
            feats = np.load(cache)
        else:
            feats = self.encoder.encode_paths(rec["frames"])
            np.save(cache, feats)
        if feats.shape[0] > self.max_frames:
            idx = np.linspace(0, feats.shape[0] - 1, self.max_frames).astype(int)
            feats = feats[idx]
        return feats.astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        feats = self._features(rec)
        scores = rec.get("scores") or {}
        dim_levels = np.array(
            [(-1 if scores.get(d) is None else int(scores[d])) for d in DIMENSIONS],
            dtype=np.int64,
        )
        flags = np.zeros(len(HARD_FAIL_FLAGS), dtype=np.float32)
        for f in rec.get("hard_fail_flags", []):
            if f in HARD_FAIL_FLAGS:
                flags[HARD_FAIL_FLAGS.index(f)] = 1.0
        return {
            "asset_id": rec["asset_id"],
            "features": torch.from_numpy(feats),
            "verdict": VERDICTS.index(rec["verdict"]),
            "dim_levels": torch.from_numpy(dim_levels),
            "flags": torch.from_numpy(flags),
        }


def collate(batch: list[dict]) -> dict:
    """Pad variable-length feature sequences; build validity mask + labels."""
    B = len(batch)
    Tmax = max(item["features"].shape[0] for item in batch)
    D = batch[0]["features"].shape[1]
    features = torch.zeros(B, Tmax, D)
    mask = torch.zeros(B, Tmax)
    for i, item in enumerate(batch):
        t = item["features"].shape[0]
        features[i, :t] = item["features"]
        mask[i, :t] = 1.0
    return {
        "asset_id": [item["asset_id"] for item in batch],
        "features": features,
        "mask": mask,
        "verdict": torch.tensor([item["verdict"] for item in batch], dtype=torch.long),
        "dim_levels": torch.stack([item["dim_levels"] for item in batch]),  # [B,6]
        "flags": torch.stack([item["flags"] for item in batch]),            # [B,9]
    }


def labels_from_batch(batch: dict) -> dict:
    """Map a collated batch to the label dict expected by combined_loss."""
    return {
        "verdict": batch["verdict"],
        "dim_levels": {DIMENSIONS[j]: batch["dim_levels"][:, j] for j in range(len(DIMENSIONS))},
        "flags": batch["flags"],
    }

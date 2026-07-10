"""Stage 5 — train.

Wrap the frozen encoder + temporal transformer + 3 head groups into a
config-driven, resumable multi-task trainer. Verdict (softmax CE), 6 CORAL
ordinal dimensions (N/A masked), and 9 pos-weighted sigmoid flags are optimised
jointly.

Artifacts (under ``<workdir>/train/``):
  - ``model.pt``       checkpoint (weights + optimiser + epoch + metadata)
  - ``train_log.json`` per-epoch loss history + final summary
  - ``features/``      cached frozen per-frame features
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import PipelineConfig
from ..data.dataset import ScreenerDataset, collate, labels_from_batch
from ..models.encoder import build_encoder
from ..models.losses import combined_loss
from ..models.model import MultiTaskScreener
from ..taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, TAXONOMY_VERSION, VERDICTS
from ..utils.io import read_jsonl, write_json


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _run_epoch(model, loader, optimizer, tcfg, train: bool) -> dict[str, float]:
    model.train(train)
    agg = {"loss": 0.0, "verdict": 0.0, "dims": 0.0, "flags": 0.0}
    n = 0
    for batch in loader:
        outputs = model(batch["features"], batch["mask"])
        labels = labels_from_batch(batch)
        loss, parts = combined_loss(
            outputs, labels, tcfg.w_verdict, tcfg.w_dims, tcfg.w_flags,
            tcfg.flag_pos_weight,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for k in agg:
            agg[k] += parts[k]
        n += 1
    return {k: (v / max(n, 1)) for k, v in agg.items()}


def run(cfg: PipelineConfig, resume: bool = True) -> dict[str, Any]:
    _seed_everything(cfg.train.seed)
    ds_dir = cfg.stage_dir("dataset")
    train_recs = read_jsonl(ds_dir / "train.jsonl")
    val_recs = read_jsonl(ds_dir / "val.jsonl")
    if not train_recs:
        raise RuntimeError("no training records; run the dataset stage first")

    out_dir = cfg.stage_dir("train")
    feat_dir = out_dir / "features"
    encoder = build_encoder(cfg)
    feature_dim = encoder.feature_dim

    train_ds = ScreenerDataset(train_recs, encoder, feat_dir, cfg.model.max_frames)
    val_ds = ScreenerDataset(val_recs, encoder, feat_dir, cfg.model.max_frames)
    if len(train_ds) == 0:
        raise RuntimeError("no trainable clips (all decode failures?)")

    bs = min(cfg.train.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = (
        DataLoader(val_ds, batch_size=max(1, len(val_ds)), shuffle=False, collate_fn=collate)
        if len(val_ds) > 0 else None
    )

    model = MultiTaskScreener(
        feature_dim=feature_dim, d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    ckpt_path = out_dir / "model.pt"
    start_epoch = 0
    best_val = float("inf")
    history: list[dict] = []
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("feature_dim") == feature_dim:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", float("inf"))
            history = ckpt.get("history", [])

    patience = cfg.train.early_stop_patience
    since_best = 0
    best_state = model.state_dict()
    for epoch in range(start_epoch, cfg.train.epochs):
        tr = _run_epoch(model, train_loader, optimizer, cfg.train, train=True)
        va = (_run_epoch(model, val_loader, optimizer, cfg.train, train=False)
              if val_loader else None)
        rec = {"epoch": epoch, "train": tr, "val": va}
        history.append(rec)
        # Early-stop on val loss only when val is large enough to be a reliable
        # signal; with a tiny val set (<3 clips) it is too noisy, so monitor
        # train loss and let the model converge/overfit.
        monitor = va["loss"] if (va and len(val_ds) >= 3) else tr["loss"]
        if monitor < best_val - 1e-5:
            best_val = monitor
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
        # persist every epoch (resumable)
        torch.save({
            "model": model.state_dict(),
            "best_model": best_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "history": history,
            "feature_dim": feature_dim,
            "encoder": encoder.name,
            "taxonomy_version": TAXONOMY_VERSION,
            "config": cfg.model_dump(),
            "verdicts": list(VERDICTS),
            "dimensions": list(DIMENSIONS),
            "flags": list(HARD_FAIL_FLAGS),
            "model_args": {
                "feature_dim": feature_dim, "d_model": cfg.model.d_model,
                "n_layers": cfg.model.n_layers, "n_heads": cfg.model.n_heads,
                "dropout": cfg.model.dropout,
            },
        }, ckpt_path)
        if since_best >= patience:
            break

    # keep last-epoch weights in "model" (for clean resume) and best weights in
    # "best_model" (used by load_model for inference); already persisted above.
    log = {
        "taxonomy_version": TAXONOMY_VERSION,
        "encoder": encoder.name,
        "feature_dim": feature_dim,
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "final_train_loss": history[-1]["train"]["loss"] if history else None,
        "history": history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    write_json(out_dir / "train_log.json", log)
    return {
        "stage": "train",
        "encoder": encoder.name,
        "feature_dim": feature_dim,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs_run": len(history),
        "best_val_loss": round(best_val, 4),
        "final_train_loss": round(history[-1]["train"]["loss"], 4) if history else None,
        "model_path": str(ckpt_path),
    }


def load_model(ckpt_path: str | Path):
    """Rebuild a ``MultiTaskScreener`` from a checkpoint (best weights)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiTaskScreener(**ckpt["model_args"])
    model.load_state_dict(ckpt.get("best_model", ckpt["model"]))
    model.eval()
    return model, ckpt

"""Frozen per-frame feature encoder.

The multi-task screener consumes *frozen* per-frame features (design: frozen
CLIP/SigLIP features -> temporal transformer -> heads). Two backends:

- ``DeterministicEncoder`` (default, offline): interpretable technical scalars
  (brightness, sharpness, clipping, corner energy, colour stats) concatenated
  with a fixed seeded random projection of a downsampled patch grid. Fully
  deterministic and network-free, so the whole pipeline runs and tests are
  reproducible without downloading any weights.
- ``ClipEncoder`` (``encoder: "clip:<name>"``): a real CLIP/SigLIP image tower
  via open_clip, used only when weights are locally available. Falls back to
  the deterministic encoder if the model cannot be loaded (e.g. offline).

Both are FROZEN: no gradients flow into them; features are cached to disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# number of interpretable scalar features placed in the leading slots
_N_INTERP = 32
_PATCH = 16  # downsampled patch grid side for the texture projection


class DeterministicEncoder:
    """Deterministic, offline per-frame feature extractor."""

    def __init__(self, feature_dim: int = 512, seed: int = 20240501):
        self.feature_dim = feature_dim
        self.seed = seed
        self._proj: np.ndarray | None = None
        self.name = "deterministic"

    def _projection(self, raw_dim: int) -> np.ndarray:
        if self._proj is None or self._proj.shape[0] != raw_dim:
            rng = np.random.RandomState(self.seed)
            out_dim = max(1, self.feature_dim - _N_INTERP)
            self._proj = rng.randn(raw_dim, out_dim).astype(np.float32) / np.sqrt(raw_dim)
        return self._proj

    def _interpretable(self, bgr: np.ndarray) -> np.ndarray:
        gu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)   # uint8 (for cv2 filters)
        g = gu.astype(np.float32)                    # float (for mean/std)
        h, w = g.shape
        feats: list[float] = []
        feats.append(g.mean() / 255.0)                                   # brightness
        feats.append(min(cv2.Laplacian(gu, cv2.CV_64F).var() / 2000.0, 2.0))  # sharpness
        feats.append(float((g < 8).mean() + (g > 247).mean()))           # clip fraction
        feats.append(g.std() / 128.0)                                    # contrast
        b, gr, r = bgr[..., 0].mean(), bgr[..., 1].mean(), bgr[..., 2].mean()
        feats.extend([r / 255.0, gr / 255.0, b / 255.0])                 # colour means
        # corner brightnesses (watermark/logo tend to sit in corners)
        ch, cw = h // 4, w // 4
        feats.append(g[:ch, :cw].mean() / 255.0)     # TL
        feats.append(g[:ch, -cw:].mean() / 255.0)    # TR
        feats.append(g[-ch:, :cw].mean() / 255.0)    # BL
        feats.append(g[-ch:, -cw:].mean() / 255.0)   # BR
        feats.append(g[ch:-ch, cw:-cw].mean() / 255.0 if h > 2 * ch and w > 2 * cw else 0.0)
        # edge density
        sx = cv2.Sobel(gu, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gu, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(float(np.sqrt(sx * sx + sy * sy).mean()) / 255.0)
        arr = np.array(feats, dtype=np.float32)
        if arr.shape[0] < _N_INTERP:
            arr = np.pad(arr, (0, _N_INTERP - arr.shape[0]))
        return arr[:_N_INTERP]

    def _texture(self, bgr: np.ndarray) -> np.ndarray:
        patch = cv2.resize(bgr, (_PATCH, _PATCH), interpolation=cv2.INTER_AREA)
        return (patch.astype(np.float32) / 255.0).reshape(-1)

    def encode_frame(self, bgr: np.ndarray) -> np.ndarray:
        interp = self._interpretable(bgr)
        tex = self._texture(bgr)
        proj = self._texture_projection = self._projection(tex.shape[0])
        projected = tex @ proj
        vec = np.concatenate([interp, projected]).astype(np.float32)
        # L2 normalize for stable transformer inputs
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def encode_paths(self, frame_paths: list[str]) -> np.ndarray:
        vecs = []
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            vecs.append(self.encode_frame(img))
        if not vecs:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack(vecs).astype(np.float32)


class ClipEncoder:
    """Real CLIP/SigLIP image tower via open_clip (used only if available)."""

    def __init__(self, model_name: str, feature_dim: int = 512):
        import open_clip  # raises if not installed
        import torch

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.feature_dim = int(self.model.encode_image(dummy).shape[-1])
        self.name = f"clip:{model_name}"

    def encode_paths(self, frame_paths: list[str]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        ims = []
        for fp in frame_paths:
            if not Path(fp).exists():
                continue
            ims.append(self.preprocess(Image.open(fp).convert("RGB")))
        if not ims:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        batch = torch.stack(ims)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def build_encoder(cfg) -> DeterministicEncoder | ClipEncoder:
    """Resolve the encoder from ``cfg.model.encoder``.

    - ``"deterministic"`` -> DeterministicEncoder
    - ``"clip:<name>"``   -> ClipEncoder (fallback to deterministic on failure)
    - ``"auto"``          -> try a small local CLIP, else deterministic
    """
    spec = cfg.model.encoder
    fdim = cfg.model.feature_dim
    if spec == "deterministic":
        return DeterministicEncoder(feature_dim=fdim)
    if spec.startswith("clip:"):
        try:
            return ClipEncoder(spec.split(":", 1)[1], feature_dim=fdim)
        except Exception:
            return DeterministicEncoder(feature_dim=fdim)
    # auto
    try:
        return ClipEncoder("ViT-B-32", feature_dim=fdim)
    except Exception:
        return DeterministicEncoder(feature_dim=fdim)

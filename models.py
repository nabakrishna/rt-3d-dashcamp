"""
models.py — Neural-network model wrappers.

Contains:
  • build_turbo_lut / depth_to_color — fast numpy Turbo colormap
  • DepthEstimator                   — Depth-Anything-V2 (HuggingFace pipeline)
  • load_yolo                        — YOLOv8n object detector
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from config import DEPTH_MODEL, DEPTH_ALPHA

logger = logging.getLogger(__name__)


# ─── Turbo colormap LUT ───────────────────────────────────────────────────────

def build_turbo_lut() -> np.ndarray:
    """Pre-compute a 256-entry BGR Turbo-colormap lookup table (one-time cost)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.50:
            r, g, b = 0, 255, int((1 - (t - 0.25) * 4) * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.50) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int((1 - (t - 0.75) * 4) * 255), 0
        lut[i] = (b, g, r)      # OpenCV uses BGR
    return lut


# Module-level singleton — built once, reused everywhere.
TURBO_LUT: np.ndarray = build_turbo_lut()


def depth_to_color(depth_norm: np.ndarray) -> np.ndarray:
    """Map any-shape normalised depth array [0, 1] → BGR Turbo colours.

    Works for both 1-D (point-cloud scatter) and 2-D (full heat map) inputs
    via numpy fancy indexing.
    """
    idx = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
    return TURBO_LUT[idx]


# ─── Depth Estimator ──────────────────────────────────────────────────────────

class DepthEstimator:
    """Monocular depth via Depth-Anything-V2-Small (HuggingFace pipeline).

    The raw model output is normalised to [0, 1] and then blended with the
    previous frame using exponential smoothing to reduce flickering.

    Args:
        device: ``"cuda"`` or ``"cpu"``.
        alpha:  Smoothing weight for the *previous* frame (higher = smoother
                but lags faster-moving objects).
    """

    def __init__(self, device: str, alpha: float = DEPTH_ALPHA) -> None:
        self.device = device
        self.alpha  = alpha
        self._prev: Optional[np.ndarray] = None
        self._pipe  = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Download (once) and initialise the depth pipeline."""
        logger.info("Loading Depth-Anything-V2-Small …")
        try:
            from transformers import pipeline as hf_pipeline   # lazy import

            hf_device = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=DEPTH_MODEL,
                device=hf_device,
            )
            logger.info("Depth model ready ✓")
        except Exception as exc:
            logger.error("Failed to load depth model: %s", exc)
            raise

    # ------------------------------------------------------------------
    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        """Return a temporally-smoothed, normalised depth map for *bgr*.

        The output has the **same spatial size** as the input frame and
        values in [0, 1] where 1 means *far* in the model's relative scale.

        Args:
            bgr: Input frame in OpenCV BGR format (uint8).

        Returns:
            float32 ndarray with shape ``(H, W)`` and values in [0, 1].
        """
        if self._pipe is None:
            raise RuntimeError("Call DepthEstimator.load() before estimate().")

        from PIL import Image as PILImage   # lazy import

        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pipe(PILImage.fromarray(rgb))
        depth  = np.array(result["depth"], dtype=np.float32)

        # Match the spatial resolution of the input frame.
        depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1].
        mn, mx = depth.min(), depth.max()
        if mx - mn > 1e-5:
            depth = (depth - mn) / (mx - mn)

        # Temporal smoothing.
        if self._prev is not None and self._prev.shape == depth.shape:
            depth = self.alpha * self._prev + (1.0 - self.alpha) * depth

        self._prev = depth.copy()
        return depth


# ─── YOLO loader ─────────────────────────────────────────────────────────────

def load_yolo(device: str):
    """Download (if needed) YOLOv8n weights and move the model to *device*.

    Returns:
        A ``ultralytics.YOLO`` model instance ready for inference.
    """
    from ultralytics import YOLO   # lazy import keeps startup fast if unused

    logger.info("Loading YOLOv8n …")
    model = YOLO("yolov8n.pt")
    model.to(device)
    logger.info("YOLO ready ✓")
    return model
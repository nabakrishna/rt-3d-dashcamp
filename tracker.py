"""
tracker.py — Object tracking and collision risk assessment.

Contains:
  • estimate_distance — fused depth + projection distance estimate
  • get_risk          — distance + lateral offset → risk label
  • ObjectTracker     — ByteTrack wrapper with trajectory and velocity history
"""
from __future__ import annotations

import collections
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import supervision as sv

from config import (
    FOCAL_LEN,
    REAL_CAR_W,
    DANGER_DIST,
    WARNING_DIST,
    CAUTION_DIST,
    LATERAL_THR,
)

logger = logging.getLogger(__name__)


# ─── Distance estimation ─────────────────────────────────────────────────────

def estimate_distance(
    depth_map: np.ndarray,
    box: np.ndarray,
    fw: int,
    fh: int,
) -> float:
    """Return a fused distance estimate (metres) for a detected bounding box.

    Two complementary cues are blended:
      1. **Depth cue** — median of the depth patch under the box, scaled from
         the monocular model's relative [0, 1] range to a rough metric range
         of 1 – 80 m.
      2. **Projection cue** — pinhole-camera estimate using the apparent pixel
         width of the box and an assumed real-world vehicle width.

    Args:
        depth_map: Normalised depth array (H × W, float32, values in [0, 1]).
        box:       Bounding box in ``[x1, y1, x2, y2]`` format.
        fw:        Frame width  (pixels).
        fh:        Frame height (pixels).

    Returns:
        Estimated distance clipped to [1, 100] metres, rounded to 1 dp.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(fw - 1, x2); y2 = min(fh - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 50.0   # degenerate box — return a safe default

    patch   = depth_map[y1:y2, x1:x2]
    med     = float(np.median(patch))
    d_depth = 1.0 + med * 79.0                       # maps [0, 1] → [1, 80] m
    px_w    = max(x2 - x1, 1)
    d_proj  = (FOCAL_LEN * REAL_CAR_W) / px_w        # f · W_real / W_px

    fused = 0.5 * d_depth + 0.5 * d_proj
    return round(float(np.clip(fused, 1.0, 100.0)), 1)


# ─── Risk classification ─────────────────────────────────────────────────────

def get_risk(dist: float, box: np.ndarray, fw: int) -> str:
    """Classify collision risk from distance and lateral offset.

    DANGER is only raised when the vehicle is both close *and* roughly
    centred in the ego-lane (lateral offset < ``LATERAL_THR``).

    Returns:
        One of ``"DANGER"``, ``"WARNING"``, ``"CAUTION"``, ``"SAFE"``.
    """
    cx      = (box[0] + box[2]) / 2.0
    lat_off = abs(cx - fw / 2.0) / (fw / 2.0)   # 0 = centred, 1 = edge

    if dist < DANGER_DIST and lat_off < LATERAL_THR:
        return "DANGER"
    if dist < WARNING_DIST:
        return "WARNING"
    if dist < CAUTION_DIST:
        return "CAUTION"
    return "SAFE"


# ─── Object Tracker ──────────────────────────────────────────────────────────

class ObjectTracker:
    """Wraps ``supervision.ByteTrack`` and enriches tracked detections with
    per-ID trajectory history and frame-to-frame velocity estimates.

    Args:
        max_history: Maximum number of past centre-points kept per track ID.
    """

    def __init__(self, max_history: int = 30) -> None:
        self.tracker     = sv.ByteTrack()
        self.max_history = max_history

        # Keyed by tracker ID (int).
        self.trajectories: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_history)
        )
        self.velocities: Dict[int, Tuple[float, float]] = collections.defaultdict(
            lambda: (0.0, 0.0)
        )
        self._prev_centers: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Pass raw detections through ByteTrack and update internal state.

        Args:
            detections: ``sv.Detections`` from the current frame (untracked).

        Returns:
            ``sv.Detections`` with ``tracker_id`` populated for matched tracks.
        """
        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked

        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            box = tracked.xyxy[i]
            cx  = int((box[0] + box[2]) / 2)
            cy  = int((box[1] + box[3]) / 2)

            self.trajectories[tid].append((cx, cy))

            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                self.velocities[tid] = (float(cx - px), float(cy - py))

            self._prev_centers[tid] = (cx, cy)

        return tracked
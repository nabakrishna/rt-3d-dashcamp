"""
visualizer.py — Frame composition and all drawing routines.

Public API
----------
draw_left_panel  — annotated dashcam view with boxes, trajectories, arrows
draw_right_panel — pseudo-3D Bird's-Eye View with depth point-cloud
compose          — side-by-side merge of both panels
add_hud          — bottom status bar (FPS, frame counter, device)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from config import (
    BEV_Z_FAR,
    PANEL_H,
    PANEL_W,
    RISK_COLORS,
    VEHICLE_CLASSES,
)
from models import depth_to_color
from tracker import ObjectTracker, estimate_distance, get_risk


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _draw_lane(frame: np.ndarray) -> None:
    """Overlay a translucent green ego-lane polygon on *frame* (in-place)."""
    h, w = frame.shape[:2]
    pts = np.array(
        [
            [int(w * 0.35), int(h * 0.55)],
            [int(w * 0.65), int(h * 0.55)],
            [int(w * 0.85), h - 1],
            [int(w * 0.15), h - 1],
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 80, 0))
    # cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame) # new line --------
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 60),
                  thickness=1, lineType=cv2.LINE_AA)


# def _glow_box(
#     frame: np.ndarray,
#     x1: int, y1: int, x2: int, y2: int,
#     color: Tuple[int, int, int],
#     thickness: int = 2,
# ) -> None:
#     """Draw a bounding box with a multi-layer glow effect (in-place)."""
#     for t, alpha in ((thickness + 4, 0.15), (thickness + 2, 0.30), (thickness, 1.0)):
#         c = tuple(int(v * alpha) for v in color)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
#     # Final opaque outline on top.
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

#new code for glow box ---------------------------------------
def _glow_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a bounding box with a multi-layer glow effect (in-place)."""
    for t, alpha in ((thickness + 1, 0.15), (thickness, 1.0)):
        c = tuple(int(v * alpha) for v in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
#-------------------------------------------------------------


# ─── Left panel — dashcam ADAS view ──────────────────────────────────────────

def draw_left_panel(
    frame: np.ndarray,
    tracked: Optional[sv.Detections],
    depth_map: np.ndarray,
    tracker: ObjectTracker,
) -> np.ndarray:
    """Render the annotated dashcam view.

    Draws (on a copy of *frame*):
      • Translucent ego-lane overlay
      • Glow bounding boxes coloured by risk level
      • Label: class, distance, risk
      • Trajectory tail for each track
      • Velocity arrow from the box centre

    Returns:
        A new BGR frame with all annotations applied.
    """
    out      = frame.copy()
    _draw_lane(out)
    h, w     = out.shape[:2]
    font     = cv2.FONT_HERSHEY_DUPLEX

    if tracked is not None and len(tracked) > 0:
        for i in range(len(tracked.xyxy)):
            box   = tracked.xyxy[i].astype(int)
            cls   = int(tracked.class_id[i])   if tracked.class_id   is not None else 2
            tid   = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            label = VEHICLE_CLASSES.get(cls, "vehicle")
            dist  = estimate_distance(depth_map, box, w, h)
            risk  = get_risk(dist, box, w)
            color = RISK_COLORS[risk]

            # Bounding box
            _glow_box(out, box[0], box[1], box[2], box[3], color)

            # Label badge
            txt  = f"{label}  {dist:.0f}m  {risk}"
            fs   = 0.48
            (tw, th), _ = cv2.getTextSize(txt, font, fs, 1)
            lx   = box[0]
            ly   = max(box[1] - 6, th + 4)
            cv2.rectangle(out,
                          (lx - 2, ly - th - 4),
                          (lx + tw + 4, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(out, txt, (lx, ly), font, fs, color, 1, cv2.LINE_AA)

            # Trajectory tail
            if tid >= 0:
                traj = list(tracker.trajectories[tid])
                for k in range(1, len(traj)):
                    cv2.line(out, traj[k - 1], traj[k], color, 1, cv2.LINE_AA)

                # Velocity arrow
                vx, vy = tracker.velocities.get(tid, (0.0, 0.0))
                cx_b   = (box[0] + box[2]) // 2
                cy_b   = (box[1] + box[3]) // 2
                cv2.arrowedLine(
                    out,
                    (cx_b, cy_b),
                    (cx_b + int(vx * 8), cy_b + int(vy * 8)),
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.4,
                )

    cv2.putText(out, "DASHCAM - ADAS PERCEPTION", (10, 22),
                font, 0.55, (0, 255, 80), 1, cv2.LINE_AA)
    return out


# ─── Right panel — 3D Bird's-Eye View ────────────────────────────────────────

def _draw_grid(canvas: np.ndarray, pw: int, ph: int) -> None:
    """Draw a perspective-correct grid on the BEV canvas (in-place)."""
    # Horizontal depth lines
    for frac in np.linspace(0.3, 1.0, 10):
        y      = int(frac * ph)
        xl     = int(pw * 0.5 - pw * 0.4 * frac)
        xr     = int(pw * 0.5 + pw * 0.4 * frac)
        alpha  = int(80 + 120 * frac)
        cv2.line(canvas, (xl, y), (xr, y), (alpha, 0, alpha // 2), 1, cv2.LINE_AA)

    # Vanishing-point lane lines
    vp_x = pw // 2
    vp_y = int(ph * 0.28)
    for k in np.linspace(-0.42, 0.42, 16):
        bx    = int(pw * 0.5 + k * pw)
        alpha = int(60 + 80 * (1 - abs(k) / 0.45))
        cv2.line(canvas, (vp_x, vp_y), (bx, ph),
                 (0, alpha, alpha // 2), 1, cv2.LINE_AA)


def _project_pointcloud(
    depth_norm: np.ndarray,
    canvas: np.ndarray,
    step: int = 3,
) -> None:
    """Scatter depth pixels into BEV canvas with Turbo colouring (in-place).

    Only the lower half of the depth map (road surface) is projected to avoid
    cluttering the BEV with sky / building points.
    """
    h,  w  = depth_norm.shape
    ch, cw = canvas.shape[:2]

    ys = np.arange(h // 2, h, step)
    xs = np.arange(0,      w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    d      = depth_norm[yy, xx].astype(np.float32)
    u      = xx.astype(np.float32) / w
    bev_y  = (ch * 0.28 + (1.0 - d) * ch * 0.70).astype(np.int32)
    spread = 0.38 + 0.52 * d
    bev_x  = (cw * 0.5 + (u - 0.5) * cw * spread).astype(np.int32)

    mask = (bev_x >= 0) & (bev_x < cw) & (bev_y >= 0) & (bev_y < ch)
    canvas[bev_y[mask].ravel(), bev_x[mask].ravel()] = depth_to_color(
        d[mask].ravel()
    )


def _bev_box(
    canvas: np.ndarray,
    cx_norm: float,
    dist_m: float,
    label: str,
    color: Tuple[int, int, int],
    pw: int,
    ph: int,
) -> None:
    """Draw a perspective-scaled 3-D box for one detected vehicle on the BEV."""
    d_norm = float(np.clip(1.0 - dist_m / BEV_Z_FAR, 0.0, 1.0))
    bev_y  = int(ph * 0.28 + (1.0 - d_norm) * ph * 0.70)
    spread = 0.38 + 0.52 * d_norm
    bev_x  = int(pw * 0.5 + (cx_norm - 0.5) * pw * spread)
    scale  = 0.5 + 1.5 * (1.0 - d_norm)

    bw = int(55 * scale);  bh = int(35 * scale);  bd = int(20 * scale)
    fl, fr = bev_x - bw // 2, bev_x + bw // 2
    ft, fb = bev_y - bh // 2, bev_y + bh // 2
    bl, br = fl + bd, fr + bd
    bt, bb = ft - bd // 2, fb - bd // 2

    def _glow_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        for t, a in ((3, 0.20), (2, 0.50), (1, 1.00)):
            c = tuple(int(v * a) for v in color)
            cv2.line(canvas, p1, p2, c, t, cv2.LINE_AA)

    # Front face
    _glow_line((fl, ft), (fr, ft)); _glow_line((fr, ft), (fr, fb))
    _glow_line((fr, fb), (fl, fb)); _glow_line((fl, fb), (fl, ft))
    # Back face
    _glow_line((bl, bt), (br, bt)); _glow_line((br, bt), (br, bb))
    _glow_line((br, bb), (bl, bb)); _glow_line((bl, bb), (bl, bt))
    # Connecting edges
    for p1, p2 in (((fl, ft), (bl, bt)), ((fr, ft), (br, bt)),
                   ((fr, fb), (br, bb)), ((fl, fb), (bl, bb))):
        _glow_line(p1, p2)

    # Label above the box
    fs = 0.38 * (0.6 + 0.8 * scale)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    tx, ty = bev_x - tw // 2, ft - 6
    if 0 < ty < ph:
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA)


def draw_right_panel(
    depth_norm: np.ndarray,
    tracked: Optional[sv.Detections],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Build and return the 3-D BEV panel (black background).

    Steps:
      1. Project depth point-cloud onto perspective canvas.
      2. Apply horizon fade.
      3. Draw perspective grid.
      4. Draw per-vehicle 3-D boxes in BEV space.
    """
    canvas  = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    d_small = cv2.resize(depth_norm, (PANEL_W, PANEL_H))
    _project_pointcloud(d_small, canvas)

    # Horizon atmospheric fade (brightens top of canvas slightly)
    horizon = int(PANEL_H * 0.3)
    grad    = np.linspace(200, 0, horizon, dtype=np.uint8)
    ch0     = canvas[:horizon, :, 0].astype(np.int16) + (grad[:, None] // 4)
    canvas[:horizon, :, 0] = np.clip(ch0, 0, 255).astype(np.uint8)

    _draw_grid(canvas, PANEL_W, PANEL_H)

    if (tracked is not None
            and len(tracked) > 0
            and tracked.tracker_id is not None):
        d_full = cv2.resize(depth_norm, (frame_w, frame_h))
        for i in range(len(tracked.xyxy)):
            box     = tracked.xyxy[i]
            cls     = int(tracked.class_id[i]) if tracked.class_id is not None else 2
            name    = VEHICLE_CLASSES.get(cls, "vehicle")
            dist    = estimate_distance(d_full, box, frame_w, frame_h)
            risk    = get_risk(dist, box, frame_w)
            color   = RISK_COLORS[risk]
            cx_norm = ((box[0] + box[2]) / 2.0) / frame_w
            _bev_box(
                canvas, cx_norm, dist,
                f"{name}  {dist:.0f}m  {risk}",
                color, PANEL_W, PANEL_H,
            )

    cv2.putText(canvas, "3D BEV - DEPTH PERCEPTION", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


# ─── Composition ─────────────────────────────────────────────────────────────

def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize both panels and join them with a thin green divider."""
    l   = cv2.resize(left,  (PANEL_W, PANEL_H))
    r   = cv2.resize(right, (PANEL_W, PANEL_H))
    div = np.full((PANEL_H, 4, 3), (0, 255, 80), dtype=np.uint8)
    return np.hstack([l, div, r])


def add_hud(
    frame: np.ndarray,
    fps: float,
    frame_no: int,
    total: int,
    device_label: str,
) -> None:
    """Draw the bottom-of-frame status bar (in-place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (5, 5, 5), -1)
    pct = frame_no / max(total, 1) * 100
    txt = (
        f"  FPS: {fps:5.1f}"
        f"  |  Frame: {frame_no}/{total} ({pct:.1f}%)"
        f"  |  Device: {device_label}"
        f"  |  3D-PERC v1.3"
    )
    cv2.putText(frame, txt, (10, h - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 220, 80), 1, cv2.LINE_AA)
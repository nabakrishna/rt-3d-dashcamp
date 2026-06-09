"""
config.py — global constants and application configuration.

edit ONLY this file to change thresholds, dimensions,output, sizes, pixels or
model selection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ─── Logging (configure once at import time) ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ─── COCO vehicle class indices ───────────────────────────────────────────────

VEHICLE_CLASSES: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())   # passed to YOLO


# ─── Risk levels & display colours (OpenCV BGR) ───────────────────────────────

RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "SAFE":    (0, 255,  80),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 140, 255),
    "DANGER":  (0,   0, 255),
}


# ─── Output / panel dimensions ────────────────────────────────────────────────

OUTPUT_W: int =  2560 #1920          # total composed frame width  (left + divider + right)
OUTPUT_H: int =  1240 #720          # total composed frame height
PANEL_W:  int = OUTPUT_W // 2 # each side panel width
PANEL_H:  int = OUTPUT_H      # each side panel height


# ─── Depth model ─────────────────────────────────────────────────────────────

DEPTH_MODEL: str        = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_INPUT_W: int      = 640 #320   # width fed to depth model (smaller = faster)
DEPTH_INPUT_H: int      = 384 #192   # height fed to depth model
DEPTH_ALPHA:   float    = 0.60  # temporal smoothing weight for previous frame


# ─── Camera / projection parameters ─────────────────────────────────────────

FOCAL_LEN:  float = 700.0   # assumed focal length (pixels) for projection distance
REAL_CAR_W: float =   2.0   # assumed vehicle width (metres) for projection distance
BEV_Z_FAR:  float =  60.0   # maximum depth shown in the BEV view (metres)


# ─── Risk thresholds ─────────────────────────────────────────────────────────

DANGER_DIST:  float = 8.0    # metres — collision imminent
WARNING_DIST: float = 15.0   # metres — close following
CAUTION_DIST: float = 25.0   # metres — moderate proximity
LATERAL_THR:  float = 0.40   # fraction of frame half-width; inside this = centred


# ─── Application configuration ───────────────────────────────────────────────

@dataclass
class AppConfig:
    """Parsed CLI arguments bundled into a typed, immutable-ish object."""
    input_path: str
    conf:       float = 0.35
    device:     str   = "cpu"
    show:       bool  = False
    output_dir: str   = "./output"


# ─── Banner ──────────────────────────────────────────────────────────────────

BANNER: str = """
╔════════════════════════════════════════════════════════════════════╗
║         Real-Time 3D Dashcam Perception System  v1.3               ║
║         YOLO v8 · Depth-Anything V2 · BEV · Collision Risk         ║
╚════════════════════════════════════════════════════════════════════╝
"""
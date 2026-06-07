# Real-Time 3D Dashcam Perception System

A self-contained Python pipeline that turns any dashcam footage into an
annotated dual-panel video showing:

- **Left panel** — live ADAS view with colour-coded bounding boxes, distance
  labels, trajectory tails, and velocity arrows.
- **Right panel** — 3-D Bird's-Eye View built from a monocular depth map,
  with perspective-scaled vehicle cubes and a vanishing-point road grid.

### Example output layout

```
┌──────────────────────────┬──────────────────────────┐
│  DASHCAM · ADAS          │  3D BEV · DEPTH          │
│  [annotated video feed]  │  [point-cloud + boxes]   │
│                          │                          │
│  ┌────────────────────┐  │        car 18m CAUTION   │
│  │ car  18m  CAUTION  │  │       ╔═══╗              │
│  └────────────────────┘  │       ║   ║              │
└──────────────────────────┴──────────────────────────┘
  FPS: 12.4  |  Frame: 240/1800 (13.3%)  |  Device: CPU
```

---

## Requirements

| Requirement      | Minimum version |
|------------------|-----------------|
| Python           | 3.10, 3.11, **3.12** ✓ |
| Disk (models)    | ~600 MB for YOLO + depth weights |
| GPU *(optional)* | NVIDIA with CUDA 11.8 or 12.x  |
| ffmpeg binary    | any recent version (for H.264 output) |

> **Python 3.12 note** — `numpy < 1.26` does not support Python 3.12.
> This project pins `numpy >= 1.26.4`; the old `1.24.x` pin found in some
> forks has been removed.

---

## Installation

### 1. Clone / download

```bash
git clone https://github.com/nabakrishna/dashcam-perception.git
cd rt-3d-dashcam
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv (py -3.12.5 or less)
```

### Windows (cmd)
```
.venv\Scripts\activate.bat
```

### 3a. CPU install

```bash
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

### 3b. GPU install (NVIDIA CUDA 12.x)

```bash
pip install --upgrade pip
pip install -r requirements-gpu.txt \
    --index-url https://download.pytorch.org/whl/cu121
```

For **CUDA 11.8** replace `cu121` with `cu118`.

### 4. Install the ffmpeg binary

The Python wrapper is not needed — only the `ffmpeg` binary.

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt-get install ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (manual) | https://ffmpeg.org/download.html |

> If ffmpeg is absent the pipeline still works; the raw `.mp4` is saved
> without H.264 re-encoding (larger file, same content).

---

## Usage

```
python main.py <input_video> [options]
```

| Argument | Default | Description |
|---|---|---|
| `input` | *(required)* | Path to dashcam MP4 / MOV / AVI |
| `--conf THRESH` | `0.35` | YOLO detection confidence (0 – 1) |
| `--device cpu\|cuda` | `cpu` | Inference device |
| `--show` | *(off)* | Open live preview window (press **Q** to quit) |
| `--output-dir DIR` | `./output` | Where to save `final_output.mp4` |

### Examples

```bash
# Basic CPU run
python main.py dashcam.mp4

# GPU, lower confidence, live preview
python main.py dashcam-vid1.mp4 --device cuda --conf 0.30 --show

# Custom output folder
python main.py trip.mp4 --output-dir /videos/processed
```

---

## Project structure

```
dashcam-perception/
│
├── main.py          Entry point, argument parsing, video I/O loop
├── config.py        All constants, thresholds, and AppConfig dataclass
├── models.py        DepthEstimator (Depth-Anything-V2) + load_yolo
├── tracker.py       ObjectTracker (ByteTrack), distance & risk helpers
├── visualizer.py    Panel drawing: dashcam view, 3-D BEV, HUD
│
├── requirements.txt        CPU dependencies (Python 3.10–3.12)
├── requirements-gpu.txt    GPU / CUDA 12.x dependencies
└── README.md
```

---

## Architecture

```
Video frame
    │
    ▼
cv2.resize ──────────► work frame (≤ 640 px wide)
    │                         │
    │              ┌──────────┴──────────┐
    │              │                     │
    ▼              ▼                     ▼
YOLOv8n       depth model          (re-used work frame)
detection    (320 × 192 input)
    │              │
    ▼              ▼
ByteTrack     depth_norm (float32, 0-1)
tracking      temporal EMA smoothing
    │              │
    └──────┬────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
Left panel    Right panel
(dashcam +    (BEV point-
 annotations)  cloud + cubes)
    │              │
    └──────┬────────┘
           ▼
       compose()  →  add_hud()  →  VideoWriter
```

### Key algorithms

**Distance estimation** — two cues are averaged:
1. *Depth cue* — median of the normalised depth patch under the bounding
   box, linearly scaled to 1 – 80 m.
2. *Projection cue* — pinhole model: `dist = focal_len × car_width / pixel_width`.

**Risk classification**

| Label   | Condition |
|---------|-----------|
| DANGER  | dist < 8 m **and** laterally centred (< 40 % offset from centre) |
| WARNING | dist < 15 m |
| CAUTION | dist < 25 m |
| SAFE    | otherwise |

**BEV point-cloud** — lower-half depth pixels are projected onto a
perspective canvas using a simple affine mapping that maps near (high depth
value) to the bottom and far (low depth value) toward the vanishing point.

---

## Tuning tips

- Raise `--conf` (e.g. `0.50`) to suppress false positives on busy roads.
- Lower `DEPTH_INPUT_W / H` in `config.py` to speed up CPU inference at
  the cost of depth resolution.
- Adjust `DANGER_DIST`, `WARNING_DIST`, `CAUTION_DIST` in `config.py` to
  suit your dashcam FOV and driving scenario.
- Increase `DEPTH_ALPHA` (toward 1.0) for smoother depth but more temporal
  lag; decrease toward 0.0 for more responsive but noisier depth.

---

## Troubleshooting

read the error

---

## Licence

MIT — see `LICENSE` for details.  

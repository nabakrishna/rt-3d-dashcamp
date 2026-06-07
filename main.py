"""
main.py — Entry point for the Real-Time 3D Dashcam Perception System.

Usage
-----
    python main.py path/to/dashcam.mp4 [--conf 0.35] [--device cpu|cuda] [--show]

Run ``python main.py --help`` for the full option list.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import torch
from tqdm import tqdm

# Lazy imports for supervision / ultralytics happen inside process_video so
# that --help is instant and import errors are reported cleanly.

from config import (
    BANNER,
    AppConfig,
    OUTPUT_W,
    OUTPUT_H,
    PANEL_W,
    PANEL_H,
    VEHICLE_CLASS_IDS,
)
from models import DepthEstimator, load_yolo
from tracker import ObjectTracker
from visualizer import add_hud, compose, draw_left_panel, draw_right_panel

logger = logging.getLogger(__name__)


# ─── FFmpeg helpers ───────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    """Return True only if the ``ffmpeg`` binary is reachable on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _reencode(tmp: str, final: str) -> bool:
    """Re-encode *tmp* to H.264/MP4 at *final* using ffmpeg.

    Returns:
        True on success, False on any error (caller falls back to raw file).
    """
    cmd = [
        "ffmpeg", "-y", "-i", tmp,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        final,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("FFmpeg re-encode failed: %s",
                       exc.stderr.decode(errors="replace")[:400])
        return False


# ─── Core processing loop ────────────────────────────────────────────────────

def process_video(cfg: AppConfig) -> None:
    """Run the full perception pipeline on the input video.

    Stages per frame
    ----------------
    1. Read frame → resize to working resolution (≤ 640 wide).
    2. YOLOv8n detection → ByteTrack tracking.
    3. Depth-Anything-V2 depth estimation on a smaller input.
    4. Left panel: annotated dashcam view.
    5. Right panel: 3-D BEV with depth point-cloud.
    6. Compose, add HUD, write to VideoWriter.
    7. (Optional) re-encode with FFmpeg for H.264 output.
    """
    import supervision as sv   # import here for clean error messages

    print(BANNER)

    # ── Device ───────────────────────────────────────────────────────────
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    if cfg.device == "cuda" and not use_cuda:
        logger.warning("CUDA requested but not available — falling back to CPU.")
    device    = "cuda" if use_cuda else "cpu"
    dev_label = f"CUDA · {torch.cuda.get_device_name(0)}" if use_cuda else "CPU"
    logger.info("Compute device: %s", dev_label)

    # ── Paths ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    tmp_path   = os.path.join(cfg.output_dir, "_tmp_raw.mp4")
    final_path = os.path.join(cfg.output_dir, "final_output.mp4")

    # ── Open input ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.input_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", cfg.input_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Input : %s  (%d × %d  @  %.1f fps  —  %d frames)",
                cfg.input_path, src_w, src_h, src_fps, total_frames)
    logger.info("Output: %s", final_path)

    # ── Load models ───────────────────────────────────────────────────────
    yolo      = load_yolo(device)
    depth_est = DepthEstimator(device)
    depth_est.load()
    tracker   = ObjectTracker()

    # ── VideoWriter ───────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, src_fps, (OUTPUT_W + 4, OUTPUT_H))
    if not writer.isOpened():
        logger.error("Cannot open VideoWriter at: %s", tmp_path)
        cap.release()
        sys.exit(1)

    # ── Main loop ─────────────────────────────────────────────────────────
    pbar     = tqdm(total=total_frames, desc="Processing", unit="fr",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    t0       = time.perf_counter()
    fps_avg  = 0.0
    frame_no = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frame_no += 1

            # Working resolution — cap at 640 px wide for speed.
            work_w = min(src_w, 640)
            work_h = int(src_h * work_w / src_w)
            work   = cv2.resize(bgr, (work_w, work_h))

            # ── Detection + tracking ──────────────────────────────────────
            results  = yolo(
                work,
                conf=cfg.conf,
                classes=VEHICLE_CLASS_IDS,
                verbose=False,
                device=device,
            )
            sv_det  = sv.Detections.from_ultralytics(results[0])
            tracked = tracker.update(sv_det)

            # ── Depth estimation ──────────────────────────────────────────
            # Run on a smaller crop for speed; upscale result back to work size.
            from config import DEPTH_INPUT_W, DEPTH_INPUT_H
            depth_small = cv2.resize(work, (DEPTH_INPUT_W, DEPTH_INPUT_H))
            depth_norm  = depth_est.estimate(depth_small)
            depth_norm  = cv2.resize(depth_norm, (work_w, work_h))


            #new code -----------------------------------------------------
            # det_w    = min(src_w, 640)
            # det_h    = int(src_h * det_w / src_w)
            # det_frame = cv2.resize(bgr, (det_w, det_h))

            # # Render frame — full resolution for quality output
            # work_w = min(src_w, 1920)
            # work_h = int(src_h * work_w / src_w)
            # work   = cv2.resize(bgr, (work_w, work_h))

            # # YOLO runs on the small frame
            # results = yolo(
            #     det_frame,
            #     conf=cfg.conf,
            #     classes=VEHICLE_CLASS_IDS,
            #     verbose=False,
            #     device=device,
            # )

            # # Scale detections boxes up from det_frame coords → work frame coords
            # sv_det  = sv.Detections.from_ultralytics(results[0])
            # scale_x = work_w / det_w
            # scale_y = work_h / det_h
            # if len(sv_det) > 0:
            #     sv_det.xyxy[:, [0, 2]] *= scale_x
            #     sv_det.xyxy[:, [1, 3]] *= scale_y
            # tracked = tracker.update(sv_det)

            # # Depth on a smaller input (saves ~60% of inference time)
            # from config import DEPTH_INPUT_W, DEPTH_INPUT_H
            # depth_small = cv2.resize(work, (DEPTH_INPUT_W, DEPTH_INPUT_H))
            # depth_norm  = depth_est.estimate(depth_small)
            # depth_norm  = cv2.resize(depth_norm, (work_w, work_h))
            #------------------------------------------------------------

            # ── Visualisation ─────────────────────────────────────────────
            left  = draw_left_panel(work, tracked, depth_norm, tracker)
            right = draw_right_panel(depth_norm, tracked, work_w, work_h)
            frame = compose(left, right)

            elapsed = time.perf_counter() - t0
            fps_now = frame_no / max(elapsed, 1e-6)
            fps_avg = (0.9 * fps_avg + 0.1 * fps_now) if fps_avg > 0.0 else fps_now
            add_hud(frame, fps_avg, frame_no, total_frames, dev_label)

            writer.write(frame)

            # ── Optional live preview ─────────────────────────────────────
            if cfg.show:
                preview_w = 1420
                preview_h = int(PANEL_H * preview_w / OUTPUT_W) + 100
                preview   = cv2.resize(frame, (preview_w, preview_h))
                cv2.imshow("3D Dashcam Perception  [Q = quit]", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User quit preview.")
                    break

            pbar.update(1)

    finally:
        # Always release resources even on exceptions.
        pbar.close()
        cap.release()
        writer.release()
        if cfg.show:
            cv2.destroyAllWindows()

    total_elapsed = time.perf_counter() - t0
    logger.info(
        "Finished: %d frames in %.1f s  (%.1f fps avg)",
        frame_no, total_elapsed, frame_no / max(total_elapsed, 1e-6),
    )

    # ── Re-encode ─────────────────────────────────────────────────────────
    if _ffmpeg_available() and _reencode(tmp_path, final_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        size_mb = os.path.getsize(final_path) / (1024 ** 2)
        print(f"\n✅  OUTPUT  →  {final_path}  ({size_mb:.1f} MB)")
    else:
        shutil.move(tmp_path, final_path)
        size_mb = os.path.getsize(final_path) / (1024 ** 2)
        print(f"\n✅  OUTPUT  →  {final_path}  ({size_mb:.1f} MB)  [uncompressed]")
        print("    Tip: install the ffmpeg binary for smaller H.264 output.")
        print("         https://ffmpeg.org/download.html")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dashcam_perception",
        description="Real-Time 3D Dashcam Perception System — v1.3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python main.py dashcam.mp4 --device cuda --conf 0.4 --show\n"
        ),
    )
    p.add_argument(
        "input",
        type=str,
        help="Path to the input dashcam MP4 file.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.35,
        metavar="THRESH",
        help="YOLO detection confidence threshold (0.0 – 1.0).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device.  Use 'cuda' if you have an NVIDIA GPU.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open a live preview window (press Q to quit).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        metavar="DIR",
        help="Directory where the output MP4 is saved.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    cfg = AppConfig(
        input_path=args.input,
        conf=args.conf,
        device=args.device,
        show=args.show,
        output_dir=args.output_dir,
    )
    process_video(cfg)
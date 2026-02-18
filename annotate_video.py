#!/usr/bin/env python3
"""
Ring Dynamics — Video Tracking Annotator
=========================================
Processes a boxing video using YOLOv8 + ByteTrack to detect and track
fighters (people) frame-by-frame, then renders colored bounding boxes
and movement trails on the output video.

Usage:
    python3 annotate_video.py <input_video> [output_video]

Example:
    python3 annotate_video.py fight.mp4
    python3 annotate_video.py fight.mp4 fight_annotated.mp4

If no output path is given, the output is saved next to the input file
with "_annotated" appended to the filename.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np

# ── Ultralytics (YOLOv8 + ByteTrack) ──────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Run: pip3 install ultralytics")
    sys.exit(1)


# ── Visual constants ───────────────────────────────────────────────────────────
# Distinct colours per fighter ID (BGR)
FIGHTER_COLORS = [
    (0,   200, 255),   # Fighter 0 → amber/gold
    (0,   255, 100),   # Fighter 1 → neon green
    (255,  80,  80),   # Fighter 2 → blue
    (255,  50, 200),   # Fighter 3 → magenta
    (200, 255,   0),   # Fighter 4 → lime
    (0,   150, 255),   # Fighter 5 → orange
]

TRAIL_LENGTH   = 40    # frames of movement trail to keep
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.65
FONT_THICKNESS = 2
CONF_THRESHOLD = 0.35  # minimum detection confidence


def get_color(track_id: int) -> tuple:
    """Return a consistent BGR colour for a given track ID."""
    return FIGHTER_COLORS[track_id % len(FIGHTER_COLORS)]


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=8):
    """Draw a bounding box with rounded corners."""
    # Clamp radius
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

    # Straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, thickness)


def draw_label(img, text, x, y, color, bg_alpha=0.55):
    """Draw a semi-transparent label above a bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 4
    bx1, by1 = x, y - th - 2 * pad
    bx2, by2 = x + tw + 2 * pad, y

    # Clamp to frame
    bx1, by1 = max(bx1, 0), max(by1, 0)
    bx2, by2 = min(bx2, img.shape[1] - 1), min(by2, img.shape[0] - 1)

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    # Text
    cv2.putText(img, text, (bx1 + pad, by2 - pad),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_trail(img, trail: deque, color):
    """Draw a fading movement trail from past centre points."""
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thickness = max(1, int(3 * alpha))
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(img, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)


def draw_hud(img, frame_idx: int, fps: float, n_fighters: int):
    """Draw a small HUD overlay in the top-left corner."""
    elapsed = frame_idx / fps if fps > 0 else 0
    mm, ss = divmod(int(elapsed), 60)
    lines = [
        f"Ring Dynamics  |  Frame {frame_idx}",
        f"Time  {mm:02d}:{ss:02d}",
        f"Fighters tracked: {n_fighters}",
    ]
    y0 = 28
    for i, line in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(img, line, (12, y), FONT, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), FONT, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)


def annotate_video(input_path: str, output_path: str,
                   model_name: str = "yolov8n.pt",
                   conf: float = CONF_THRESHOLD,
                   device: str = "cpu",
                   imgsz: int = 1280,
                   scale: float = 1.0,
                   target_fps: float = 0) -> str:
    """
    Main annotation function.

    Parameters
    ----------
    input_path  : path to source video
    output_path : path for annotated output video
    model_name  : YOLOv8 model variant (auto-downloaded if not cached)
    conf        : detection confidence threshold
    device      : 'cpu' or 'cuda' / 'mps'
    imgsz       : inference image size (shorter side), e.g. 640 or 1280
    scale       : output video scale factor (e.g. 0.5 = half resolution)

    Returns
    -------
    output_path on success
    """
    print(f"\n{'='*60}")
    print(f"  Ring Dynamics — Video Tracking Annotator")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Model  : {model_name}  |  Device: {device}  |  Conf: {conf}")
    print(f"  ImgSz  : {imgsz}  |  Scale: {scale}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[1/4] Loading YOLOv8 model …")
    model = YOLO(model_name)

    # ── Open video ────────────────────────────────────────────────────────────
    print("[2/4] Opening video …")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output dimensions (optionally downscaled)
    out_w = int(width  * scale)
    out_h = int(height * scale)
    # Ensure even dimensions for codec compatibility
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    # Frame-skip: process every Nth frame to hit target_fps
    if target_fps > 0 and target_fps < fps:
        frame_skip = max(1, round(fps / target_fps))
        out_fps = fps / frame_skip
    else:
        frame_skip = 1
        out_fps = fps

    frames_to_process = total // frame_skip
    print(f"    Source     : {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total}")
    print(f"    Output     : {out_w}×{out_h}  |  FPS: {out_fps:.1f}  |  Frames: ~{frames_to_process}")
    print(f"    Inference  : imgsz={imgsz}  |  frame_skip={frame_skip}")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # ── Per-track state ───────────────────────────────────────────────────────
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    # ── Process frames ────────────────────────────────────────────────────────
    print("[3/4] Processing frames …")
    frame_idx = 0
    processed = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to hit target fps
        if frame_skip > 1 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Optionally downscale frame for output
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Run YOLOv8 tracking (ByteTrack built-in)
        results = model.track(
            frame,
            persist=True,          # maintain track IDs across frames
            classes=[0],           # class 0 = person
            conf=conf,
            iou=0.45,
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            verbose=False,
            device=device,
        )

        active_ids = set()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Skip if no track ID yet
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                color = get_color(track_id)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Update trail
                trails[track_id].append((cx, cy))
                active_ids.add(track_id)

                # ── Draw trail ────────────────────────────────────────────
                draw_trail(frame, trails[track_id], color)

                # ── Draw bounding box ─────────────────────────────────────
                draw_rounded_rect(frame, x1, y1, x2, y2, color, BOX_THICKNESS)

                # ── Corner accent marks ───────────────────────────────────
                arm = 18
                for px, py, dx, dy in [
                    (x1, y1,  1,  1),
                    (x2, y1, -1,  1),
                    (x1, y2,  1, -1),
                    (x2, y2, -1, -1),
                ]:
                    cv2.line(frame, (px, py), (px + dx * arm, py), color, 3)
                    cv2.line(frame, (px, py), (px, py + dy * arm), color, 3)

                # ── Centre dot ────────────────────────────────────────────
                cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

                # ── Label ─────────────────────────────────────────────────
                label = f"Fighter {track_id}  {conf_val:.0%}"
                draw_label(frame, label, x1, y1, color)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, frame_idx, out_fps, len(active_ids))

        writer.write(frame)
        frame_idx += 1
        processed += 1

        # Progress
        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t_start
            pct = (processed / frames_to_process * 100) if frames_to_process > 0 else 0
            eta = (elapsed / processed) * (frames_to_process - processed) if processed > 0 else 0
            print(f"    Frame {processed:>5}/{frames_to_process}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n[4/4] Done! Processed {processed} frames in {elapsed:.1f}s")
    print(f"      Output saved → {output_path}\n")
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ring Dynamics — annotate a boxing video with tracking boxes"
    )
    parser.add_argument("input",  help="Path to input video (mp4, mov, avi, …)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Path for annotated output video (optional)")
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                 "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model size (n=fastest, x=most accurate)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Detection confidence threshold (default {CONF_THRESHOLD})")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu | cuda | mps")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference image size (default 1280). Use 640 for faster CPU processing.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Output video scale factor (default 1.0 = original size, 0.5 = half)")
    parser.add_argument("--fps", type=float, default=0,
                        help="Target output FPS (0 = keep original). Use 30 on 60fps video to halve processing time.")

    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        p = Path(args.input)
        output_path = str(p.parent / (p.stem + "_annotated" + p.suffix))

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    annotate_video(
        input_path=args.input,
        output_path=output_path,
        model_name=args.model,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        scale=args.scale,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()

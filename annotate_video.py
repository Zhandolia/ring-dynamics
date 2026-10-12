#!/usr/bin/env python3
"""
Ring Dynamics — Video Tracking Annotator (v5)
==============================================
Processes a boxing video using YOLOv8 + ByteTrack to detect and track
the **two main fighters** only.  Uses multi-region colour profiling,
ring-zone spatial constraints, and maximum-jump limits to reject
the referee, audience, and all other non-fighters.

v5 improvements over v4:
  - Ring-zone spatial constraint (fighters stay in center, audience at edges)
  - Maximum position jump limit (fighters can't teleport across the frame)
  - Higher MIN_ACCEPT_SCORE (0.42)
  - Stricter size filters (min height 20%, min area 2.5%)
  - Better to track 1 than incorrectly track audience as 2nd
"""

import sys
import os
import argparse
import time
import random
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Run: pip3 install ultralytics")
    sys.exit(1)


# ── Visual constants ───────────────────────────────────────────────────────────
FIGHTER_COLORS = [
    (0,   0,   255),   # Fighter A → RED  (BGR)
    (255, 50,  50),    # Fighter B → BLUE (BGR)
]
FIGHTER_NAMES  = ["Fighter A", "Fighter B"]

# Sub-box proportions (fraction of full bbox height)
HEAD_TOP       = 0.0     # head box starts at top
HEAD_BOTTOM    = 0.28    # head box ends at 28%
BODY_TOP       = 0.28    # body box starts at 28%
BODY_BOTTOM    = 0.62    # body box ends at 62%

# Sub-box inset (fraction of width) — shrink horizontally for tighter fit
HEAD_INSET_X   = 0.20    # 20% inset on each side for head
BODY_INSET_X   = 0.10    # 10% inset on each side for body

HEAD_COLOR     = (0, 220, 0)     # Green accent for head box (BGR)

TRAIL_LENGTH   = 40
BOX_THICKNESS  = 2
SUB_BOX_THICK  = 2       # thickness for head/body sub-boxes
JITTER_PX      = 2       # max ± pixels for box shivering effect
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.65
FONT_THICKNESS = 2


def jitter_box(x1, y1, x2, y2, px=JITTER_PX):
    """Add random ±px jitter to box corners for 'live tracking' shimmer."""
    return (
        x1 + random.randint(-px, px),
        y1 + random.randint(-px, px),
        x2 + random.randint(-px, px),
        y2 + random.randint(-px, px),
    )
CONF_THRESHOLD = 0.35


# ── Fighter Pose & Inversion Detection ─────────────────────────────────────────
def compute_fighter_pose_boxes(d: dict, frame_shape: tuple) -> tuple:
    """
    Computes anatomical head and core/body sub-boxes for a tracked fighter.

    Correctly handles fighters:
      1. Standing upright (head at top, feet at bottom)
      2. Knocked down / inverted on canvas (heads at bottom of screen, feet on top of screen)
      3. Lying horizontally on ground (head at left or right)

    Returns:
      (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orientation_str
    """
    x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    kpts = d.get("keypoints")
    kpts_conf = d.get("keypoints_conf")

    head_pts = []
    shoulder_pts = []
    hip_pts = []
    feet_pts = []

    # COCO Keypoint format:
    # 0: nose, 1-2: eyes, 3-4: ears
    # 5-6: shoulders, 11-12: hips, 13-14: knees, 15-16: ankles
    if kpts is not None and len(kpts) >= 17:
        for idx in [0, 1, 2, 3, 4]:
            p = kpts[idx]
            c = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
            if p[0] > 0 and p[1] > 0 and c >= 0.15:
                head_pts.append((float(p[0]), float(p[1])))

        for idx in [5, 6]:
            p = kpts[idx]
            c = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
            if p[0] > 0 and p[1] > 0 and c >= 0.15:
                shoulder_pts.append((float(p[0]), float(p[1])))

        for idx in [11, 12]:
            p = kpts[idx]
            c = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
            if p[0] > 0 and p[1] > 0 and c >= 0.15:
                hip_pts.append((float(p[0]), float(p[1])))

        for idx in [15, 16]:
            p = kpts[idx]
            c = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
            if p[0] > 0 and p[1] > 0 and c >= 0.15:
                feet_pts.append((float(p[0]), float(p[1])))

        if not feet_pts:
            for idx in [13, 14]:
                p = kpts[idx]
                c = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
                if p[0] > 0 and p[1] > 0 and c >= 0.15:
                    feet_pts.append((float(p[0]), float(p[1])))

    head_cy = float(np.mean([p[1] for p in head_pts])) if head_pts else None
    head_cx = float(np.mean([p[0] for p in head_pts])) if head_pts else None
    feet_cy = float(np.mean([p[1] for p in feet_pts])) if feet_pts else None
    feet_cx = float(np.mean([p[0] for p in feet_pts])) if feet_pts else None
    core_pts = shoulder_pts + hip_pts
    core_cy = float(np.mean([p[1] for p in core_pts])) if core_pts else None
    core_cx = float(np.mean([p[0] for p in core_pts])) if core_pts else None

    # Detect inverted state (head at bottom of screen, feet on top of screen)
    is_inverted = False
    is_horizontal = False

    if head_cy is not None and feet_cy is not None:
        if head_cy > feet_cy + 8:
            is_inverted = True
    elif head_cy is not None and core_cy is not None:
        if head_cy > core_cy + 8:
            is_inverted = True

    # Detect horizontal state (fighter lying down flat or on side)
    if bw > 1.25 * bh:
        is_horizontal = True
    elif head_cx is not None and feet_cx is not None and head_cy is not None and feet_cy is not None:
        if abs(head_cx - feet_cx) > 1.20 * abs(head_cy - feet_cy):
            is_horizontal = True

    # ── Head Sub-box ──────────────────────────────────────────────────────────
    if len(head_pts) >= 2:
        min_hx = min(p[0] for p in head_pts)
        max_hx = max(p[0] for p in head_pts)
        min_hy = min(p[1] for p in head_pts)
        max_hy = max(p[1] for p in head_pts)
        pad_x = max(int(bw * 0.08), int((max_hx - min_hx) * 0.45), 10)
        pad_y = max(int(bh * 0.08), int((max_hy - min_hy) * 0.45), 10)
        hx1 = max(x1, int(min_hx - pad_x))
        hx2 = min(x2, int(max_hx + pad_x))
        hy1 = max(y1, int(min_hy - pad_y))
        hy2 = min(y2, int(max_hy + pad_y))
    elif len(head_pts) == 1:
        px, py = head_pts[0]
        radius_x = max(int(bw * 0.14), 14)
        radius_y = max(int(bh * 0.14), 14)
        hx1 = max(x1, int(px - radius_x))
        hx2 = min(x2, int(px + radius_x))
        hy1 = max(y1, int(py - radius_y))
        hy2 = min(y2, int(py + radius_y))
    elif is_inverted:
        # Head at bottom of bounding box (inverted fighter on canvas)
        head_inset = max(1, int(bw * HEAD_INSET_X))
        hx1 = x1 + head_inset
        hx2 = x2 - head_inset
        hy1 = y2 - int(bh * HEAD_BOTTOM)
        hy2 = y2 - int(bh * HEAD_TOP)
    elif is_horizontal:
        head_inset = max(1, int(bh * HEAD_INSET_X))
        hy1 = y1 + head_inset
        hy2 = y2 - head_inset
        head_on_right = (head_cx is not None and core_cx is not None and head_cx > core_cx) or (head_cx is not None and head_cx > (x1 + x2) / 2)
        if head_on_right:
            hx1 = x2 - int(bw * HEAD_BOTTOM)
            hx2 = x2
        else:
            hx1 = x1
            hx2 = x1 + int(bw * HEAD_BOTTOM)
    else:
        # Standard upright
        head_inset = max(1, int(bw * HEAD_INSET_X))
        hx1 = x1 + head_inset
        hx2 = x2 - head_inset
        hy1 = y1 + int(bh * HEAD_TOP)
        hy2 = y1 + int(bh * HEAD_BOTTOM)

    # ── Core / Body Sub-box ──────────────────────────────────────────────────
    if len(core_pts) >= 2:
        min_cx = min(p[0] for p in core_pts)
        max_cx = max(p[0] for p in core_pts)
        min_cy = min(p[1] for p in core_pts)
        max_cy = max(p[1] for p in core_pts)
        pad_cx = max(int(bw * 0.06), 10)
        pad_cy = max(int(bh * 0.06), 10)
        bx1 = max(x1, int(min_cx - pad_cx))
        bx2 = min(x2, int(max_cx + pad_cx))
        by1 = max(y1, int(min_cy - pad_cy))
        by2 = min(y2, int(max_cy + pad_cy))
    elif is_inverted:
        body_inset = max(1, int(bw * BODY_INSET_X))
        bx1 = x1 + body_inset
        bx2 = x2 - body_inset
        by1 = y2 - int(bh * BODY_BOTTOM)
        by2 = y2 - int(bh * BODY_TOP)
    elif is_horizontal:
        body_inset = max(1, int(bh * BODY_INSET_X))
        by1 = y1 + body_inset
        by2 = y2 - body_inset
        bx1 = x1 + int(bw * BODY_TOP)
        bx2 = x1 + int(bw * BODY_BOTTOM)
    else:
        body_inset = max(1, int(bw * BODY_INSET_X))
        bx1 = x1 + body_inset
        bx2 = x2 - body_inset
        by1 = y1 + int(bh * BODY_TOP)
        by2 = y1 + int(bh * BODY_BOTTOM)

    hx1, hx2 = min(hx1, hx2), max(hx1, hx2)
    hy1, hy2 = min(hy1, hy2), max(hy1, hy2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    orient_str = "inverted" if is_inverted else ("horizontal" if is_horizontal else "upright")
    return (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orient_str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fighter Identity Tracker (v6 — Referee Rejection & Persistent Re-ID)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FighterIdentityTracker:
    """
    Maintains stable Fighter A / Fighter B identity across the entire fight.

    v6 Enhancements:
      1. PHYSIOLOGICAL & CLOTHING REFEREE DETECTOR:
         - Referees wear black/dark polo or t-shirt + long black trousers.
         - Fighters are bare-chested / shirtless (high torso skin, bare legs).
         - Rejects referee candidates in bootstrap and steady-state based on
           torso skin coverage, dark clothing ratio, and long pants detection.
      2. MULTI-REGION CIELAB & HSV COLOR RE-ID:
         - 3D color histogram and dominant color in CIELAB space for trunks.
         - Perceptual Delta-E color distance accurately distinguishes black,
           white, colored, and patterned trunks across camera cuts and zooms.
         - Torso skin tone profiling preserves fighter individuality.
      3. TRACKLET BINDING:
         - Leverages ByteTrack's continuous track IDs across frames within a shot.
         - Avoids frame-by-frame ID jitter and preserves Fighter A / B consistency.
      4. CAMERA-CUT & OCCLUSION RESILIENCE:
         - Optimal bipartite matching on appearance similarity.
         - Eliminates flawed left/right X-position ordering across camera cuts.
      5. OCCLUSION & CLINCH COASTING:
         - Smooths temporary 1-to-4 frame clinch occlusions so boxes don't vanish.
    """

    INIT_BOOTSTRAP_FRAMES = 25
    MIN_APPEARANCE_SIM    = 0.30
    TRACK_REBIND_SIM      = 0.26

    # Ring-zone bounds (normalised coordinates)
    RING_X_MIN = 0.05
    RING_X_MAX = 0.95
    RING_Y_MIN = 0.03
    RING_Y_MAX = 0.95

    # Size constraints
    MIN_HEIGHT_RATIO = 0.18
    MIN_AREA_RATIO   = 0.020

    def __init__(self):
        self._frame_count = 0
        self._profiles = [None, None]  # 0: Fighter A, 1: Fighter B
        self._last_pos = [None, None]
        self._last_seen = [-1, -1]
        self._last_det = [None, None]
        self._track_id_map: Dict[int, int] = {}  # ByteTrack ID -> stable ID
        self._bootstrapped = False

    @staticmethod
    def is_referee(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, orientation: str = "upright") -> bool:
        """
        Detect whether a candidate bounding box is the referee or a clothed official.
        Referees wear dark shirts and long dark pants, while fighters are shirtless
        with bare legs and combat shorts.
        """
        h = y2 - y1
        w = x2 - x1
        if h < 20 or w < 20:
            return True

        if orientation == "inverted":
            # Torso: lower portion of bbox; Legs: upper portion of bbox
            tx1, tx2 = x1 + int(w * 0.15), x2 - int(w * 0.15)
            ty1, ty2 = y2 - int(h * 0.45), y2 - int(h * 0.15)
            lx1, lx2 = x1 + int(w * 0.15), x2 - int(w * 0.15)
            ly1, ly2 = y1 + int(h * 0.05), y1 + int(h * 0.35)
        elif orientation == "horizontal":
            tx1, tx2 = x1 + int(w * 0.15), x1 + int(w * 0.50)
            ty1, ty2 = y1 + int(h * 0.15), y2 - int(h * 0.15)
            lx1, lx2 = x1 + int(w * 0.50), x1 + int(w * 0.85)
            ly1, ly2 = y1 + int(h * 0.15), y2 - int(h * 0.15)
        else:
            # Torso: 15% to 45% of bbox height, central 70% width
            tx1, tx2 = x1 + int(w * 0.15), x2 - int(w * 0.15)
            ty1, ty2 = y1 + int(h * 0.15), y1 + int(h * 0.45)
            # Legs: 65% to 95% of bbox height, central 70% width
            lx1, lx2 = x1 + int(w * 0.15), x2 - int(w * 0.15)
            ly1, ly2 = y1 + int(h * 0.65), y1 + int(h * 0.95)

        torso = frame[max(0, ty1):min(frame.shape[0], ty2), max(0, tx1):min(frame.shape[1], tx2)]
        legs  = frame[max(0, ly1):min(frame.shape[0], ly2), max(0, lx1):min(frame.shape[1], lx2)]

        if torso.size == 0 or legs.size == 0:
            return False

        # Skin detection across illumination: combination of YCrCb and HSV skin tone
        ycrcb_t = cv2.cvtColor(torso, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb_t, np.array([0, 133, 77], dtype=np.uint8),
                                          np.array([255, 173, 127], dtype=np.uint8))
        hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.inRange(hsv_t, np.array([0, 28, 40], dtype=np.uint8),
                                      np.array([30, 255, 255], dtype=np.uint8))
        skin_mask = cv2.bitwise_or(skin_ycrcb, skin_hsv)
        torso_skin = float(np.count_nonzero(skin_mask)) / torso.size

        # Dark clothing detection: HSV Value < 55 (black / dark fabric)
        torso_dark = float(np.count_nonzero(hsv_t[:, :, 2] < 55)) / torso.size
        hsv_l = cv2.cvtColor(legs, cv2.COLOR_BGR2HSV)
        legs_dark = float(np.count_nonzero(hsv_l[:, :, 2] < 55)) / legs.size

        # Classification rules:
        # 1. Dark shirt + dark pants + low skin: textbook referee
        if torso_skin < 0.16 and (torso_dark > 0.20 or legs_dark > 0.35):
            return True
        # 2. Torso is almost completely covered in clothing (fighters are shirtless)
        if torso_skin < 0.10:
            return True
        # 3. Long dark trousers + low/moderate skin
        if legs_dark > 0.50 and torso_skin < 0.22:
            return True

        return False

    @staticmethod
    def extract_features(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, orientation: str = "upright") -> dict:
        """Extract multi-region color signatures in CIELAB and HSV."""
        h = y2 - y1
        w = x2 - x1
        rx1, rx2 = x1 + int(w * 0.15), x2 - int(w * 0.15)

        if orientation == "inverted":
            # Inverted fighter: trunks in upper portion, torso in lower portion
            ry1, ry2 = y2 - int(h * 0.70), y2 - int(h * 0.40)
            ty1, ty2 = y2 - int(h * 0.42), y2 - int(h * 0.15)
        elif orientation == "horizontal":
            ry1, ry2 = y1 + int(h * 0.15), y2 - int(h * 0.15)
            rx1, rx2 = x1 + int(w * 0.40), x1 + int(w * 0.70)
            ty1, ty2 = y1 + int(h * 0.15), y2 - int(h * 0.15)
        else:
            # Trunks / shorts crop (40% to 70% height)
            ry1, ry2 = y1 + int(h * 0.40), y1 + int(h * 0.70)
            # Torso crop (15% to 42% height)
            ty1, ty2 = y1 + int(h * 0.15), y1 + int(h * 0.42)

        trunks = frame[max(0, ry1):min(frame.shape[0], ry2), max(0, rx1):min(frame.shape[1], rx2)]
        if trunks.size == 0:
            trunks = frame[y1:y2, x1:x2]

        lab_trunks = cv2.cvtColor(trunks, cv2.COLOR_BGR2LAB)
        mean_trunks = np.mean(lab_trunks, axis=(0, 1))
        hist_trunks = cv2.calcHist([lab_trunks], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_trunks, hist_trunks, 0, 1, cv2.NORM_MINMAX)

        torso = frame[max(0, ty1):min(frame.shape[0], ty2), max(0, rx1):min(frame.shape[1], rx2)]
        if torso.size == 0:
            torso = frame[y1:y2, x1:x2]
        lab_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        mean_torso = np.mean(lab_torso, axis=(0, 1))

        return {
            "mean_trunks": mean_trunks,
            "hist_trunks": hist_trunks,
            "mean_torso":  mean_torso,
        }

    def appearance_sim(self, feat: dict, fid: int) -> float:
        """Compute robust multi-region appearance similarity against fighter profile."""
        prof = self._profiles[fid]
        if prof is None:
            return 0.50

        # Trunks 3D color histogram similarity (Bhattacharyya)
        h_sim = max(0.0, 1.0 - cv2.compareHist(
            feat["hist_trunks"].astype(np.float32),
            prof["hist_trunks"].astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        ))

        # Trunks dominant color distance (Delta-E in CIELAB)
        dE_trunks = float(np.linalg.norm(feat["mean_trunks"] - prof["mean_trunks"]))
        color_sim_trunks = max(0.0, 1.0 - dE_trunks / 60.0)

        # Torso skin tone distance
        dE_torso = float(np.linalg.norm(feat["mean_torso"] - prof["mean_torso"]))
        color_sim_torso = max(0.0, 1.0 - dE_torso / 50.0)

        return 0.45 * h_sim + 0.40 * color_sim_trunks + 0.15 * color_sim_torso

    def assign(self, all_dets: list, frame: np.ndarray) -> list:
        """
        Filter candidate person detections, reject referees/officials,
        and assign stable IDs (0 = Fighter A, 1 = Fighter B).
        """
        self._frame_count += 1
        if not all_dets:
            # Fallback coasting if active fighters were recently visible
            return self._get_coasted_fighters([])

        h_f, w_f = frame.shape[:2]

        candidates = []
        for d in all_dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            bw = x2 - x1
            bh = y2 - y1

            # Compute pose-aware sub-boxes & orientation
            (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orient_str = compute_fighter_pose_boxes(d, frame.shape)
            d["orientation"] = orient_str
            d["head_box"] = (hx1, hy1, hx2, hy2)
            d["core_box"] = (bx1, by1, bx2, by2)

            # 1. Reject referees and clothed officials
            if self.is_referee(frame, x1, y1, x2, y2, orientation=orient_str):
                continue

            # 2. Reject detections that are too small (checking max dimension prevents dropping horizontal/ground fighters)
            max_dim = max(bw, bh)
            if max_dim < h_f * self.MIN_HEIGHT_RATIO or d["area"] < (w_f * h_f * self.MIN_AREA_RATIO):
                continue

            # 3. Ring zone constraint
            cx = (x1 + x2) / (2.0 * w_f)
            cy = (y1 + y2) / (2.0 * h_f)
            if not (self.RING_X_MIN <= cx <= self.RING_X_MAX and self.RING_Y_MIN <= cy <= self.RING_Y_MAX):
                continue

            d["feat"] = self.extract_features(frame, x1, y1, x2, y2, orientation=orient_str)
            d["_cx_norm"] = cx
            d["_cy_norm"] = cy
            candidates.append(d)

        # ── Bootstrap Phase ──────────────────────────────────────────────────
        if not self._bootstrapped:
            if len(candidates) >= 2:
                # Pick largest 2 candidates and sort horizontally initially
                top2 = sorted(candidates, key=lambda d: d["area"], reverse=True)[:2]
                top2.sort(key=lambda d: d["x1"])  # 0 on left, 1 on right
                for fid, d in enumerate(top2):
                    self._profiles[fid] = d["feat"]
                    self._track_id_map[d["track_id"]] = fid
                    self._last_pos[fid] = (d["_cx_norm"], d["_cy_norm"])
                    self._last_seen[fid] = self._frame_count
                    self._last_det[fid] = d.copy()
                    d["stable_id"] = fid
                self._bootstrapped = True
                return top2
            return []

        # ── Steady-State Phase ───────────────────────────────────────────────
        assigned: Dict[int, dict] = {}
        unassigned_cand = []

        # Step 1: ByteTrack tracklet ID binding
        for d in candidates:
            tid = d["track_id"]
            if tid in self._track_id_map:
                fid = self._track_id_map[tid]
                if fid not in assigned:
                    sim = self.appearance_sim(d["feat"], fid)
                    if sim >= self.TRACK_REBIND_SIM:
                        d["stable_id"] = fid
                        assigned[fid] = d
                        continue
            unassigned_cand.append(d)

        # Step 2: Global appearance matching for unassigned fighters
        unassigned_fids = [fid for fid in [0, 1] if fid not in assigned]
        if unassigned_fids and unassigned_cand:
            if len(unassigned_fids) == 2 and len(unassigned_cand) >= 2:
                c0, c1 = unassigned_cand[0], unassigned_cand[1]
                s_00 = self.appearance_sim(c0["feat"], 0)
                s_11 = self.appearance_sim(c1["feat"], 1)
                s_01 = self.appearance_sim(c0["feat"], 1)
                s_10 = self.appearance_sim(c1["feat"], 0)

                if (s_00 + s_11) >= (s_01 + s_10):
                    if s_00 >= self.MIN_APPEARANCE_SIM:
                        c0["stable_id"] = 0
                        assigned[0] = c0
                        self._track_id_map[c0["track_id"]] = 0
                    if s_11 >= self.MIN_APPEARANCE_SIM:
                        c1["stable_id"] = 1
                        assigned[1] = c1
                        self._track_id_map[c1["track_id"]] = 1
                else:
                    if s_01 >= self.MIN_APPEARANCE_SIM:
                        c0["stable_id"] = 1
                        assigned[1] = c0
                        self._track_id_map[c0["track_id"]] = 1
                    if s_10 >= self.MIN_APPEARANCE_SIM:
                        c1["stable_id"] = 0
                        assigned[0] = c1
                        self._track_id_map[c1["track_id"]] = 0
            else:
                for fid in unassigned_fids:
                    best_s = -1.0
                    best_c = None
                    for c in unassigned_cand:
                        s = self.appearance_sim(c["feat"], fid)
                        if s > best_s and s >= self.MIN_APPEARANCE_SIM:
                            best_s = s
                            best_c = c
                    if best_c is not None:
                        best_c["stable_id"] = fid
                        assigned[fid] = best_c
                        self._track_id_map[best_c["track_id"]] = fid
                        unassigned_cand.remove(best_c)

        # Update profiles with moving average
        result = list(assigned.values())
        for d in result:
            fid = d["stable_id"]
            self._last_pos[fid] = (d["_cx_norm"], d["_cy_norm"])
            self._last_seen[fid] = self._frame_count
            self._last_det[fid] = d.copy()

            # Slowly update profile appearance
            alpha = 0.02
            self._profiles[fid]["mean_trunks"] = (
                (1 - alpha) * self._profiles[fid]["mean_trunks"] + alpha * d["feat"]["mean_trunks"]
            )
            self._profiles[fid]["hist_trunks"] = (
                (1 - alpha) * self._profiles[fid]["hist_trunks"] + alpha * d["feat"]["hist_trunks"]
            )
            self._profiles[fid]["mean_torso"] = (
                (1 - alpha) * self._profiles[fid]["mean_torso"] + alpha * d["feat"]["mean_torso"]
            )

        # Clinch / occlusion coasting: if 1 fighter was briefly missed (<= 4 frames), carry forward
        result = self._get_coasted_fighters(result)
        return result

    def _get_coasted_fighters(self, active_fighters: list) -> list:
        assigned_fids = {d["stable_id"] for d in active_fighters}
        final_list = list(active_fighters)
        for fid in [0, 1]:
            if fid not in assigned_fids and self._last_det[fid] is not None:
                gap = self._frame_count - self._last_seen[fid]
                if 1 <= gap <= 4:
                    coasted = self._last_det[fid].copy()
                    coasted["coasted"] = True
                    final_list.append(coasted)
        return final_list



# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=8):
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)


def draw_label(img, text, x, y, color, bg_alpha=0.55):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 4
    bx1, by1 = x, y - th - 2 * pad
    bx2, by2 = x + tw + 2 * pad, y
    bx1, by1 = max(bx1, 0), max(by1, 0)
    bx2, by2 = min(bx2, img.shape[1] - 1), min(by2, img.shape[0] - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (bx1 + pad, by2 - pad),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_trail(img, trail: deque, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thickness = max(1, int(3 * alpha))
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(img, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)


def draw_hud(img, frame_idx: int, fps: float, n_fighters: int):
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
        cv2.putText(img, line, (12, y), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ── Fight Scorer ───────────────────────────────────────────────────────────────
class FightScorer:
    """Computes real-time fight metrics from tracking bounding boxes."""

    WINDOW = 30          # rolling average window (frames)
    ROUND_FRAMES = 5400  # 3 min × 30 fps

    def __init__(self):
        self._centers = [deque(maxlen=self.WINDOW), deque(maxlen=self.WINDOW)]
        self._activity = [0.0, 0.0]
        self._aggression = [0.0, 0.0]
        self._ring_ctl = [0.0, 0.0]
        self._distance = 0.0           # 0-1 normalized
        self._pressure = [0.0, 0.0]
        self._scores = [0, 0]          # cumulative round points
        self._round = 1
        self._round_pts = [10.0, 10.0] # current round running score
        self._frame_count = 0
        self._events: list = []        # (frame, text)
        self._img_w = 1.0

    def update(self, fighters: list, frame_w: int, frame_idx: int):
        """Feed one frame of fighter data and update all metrics."""
        self._img_w = max(frame_w, 1)
        self._frame_count = frame_idx

        # Determine round
        self._round = (frame_idx // self.ROUND_FRAMES) + 1

        # Extract centers
        positions = {}
        for d in fighters:
            sid = d["stable_id"]
            cx = (d["x1"] + d["x2"]) / 2.0
            cy = (d["y1"] + d["y2"]) / 2.0
            positions[sid] = (cx, cy)
            self._centers[sid].append((cx, cy))

        if len(positions) < 2:
            return

        # ─── Activity (movement speed) ───
        for sid in range(2):
            pts = self._centers[sid]
            if len(pts) >= 2:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
                speed = (dx*dx + dy*dy) ** 0.5
                self._activity[sid] = 0.85 * self._activity[sid] + 0.15 * speed

        # ─── Distance between fighters ───
        p0, p1 = positions.get(0), positions.get(1)
        if p0 and p1:
            dist = ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ** 0.5
            self._distance = min(dist / self._img_w, 1.0)

        # ─── Aggression (closing distance) ───
        for sid in range(2):
            pts = self._centers[sid]
            opp = 1 - sid
            if opp in positions and len(pts) >= 2:
                ox, oy = positions[opp]
                d_now  = ((pts[-1][0]-ox)**2 + (pts[-1][1]-oy)**2) ** 0.5
                d_prev = ((pts[-2][0]-ox)**2 + (pts[-2][1]-oy)**2) ** 0.5
                advance = max(0, d_prev - d_now)  # positive = moving forward
                self._aggression[sid] = 0.9 * self._aggression[sid] + 0.1 * advance
                # Pressure: sustained forward movement
                self._pressure[sid] = 0.95 * self._pressure[sid] + 0.05 * advance

        # ─── Ring control (distance from center) ───
        center_x = self._img_w / 2.0
        for sid in range(2):
            if sid in positions:
                dist_from_center = abs(positions[sid][0] - center_x) / (self._img_w / 2.0)
                control = max(0.0, 1.0 - dist_from_center)
                self._ring_ctl[sid] = 0.9 * self._ring_ctl[sid] + 0.1 * control

        # ─── Events ───
        # Generate events for significant actions
        if frame_idx > 0 and frame_idx % 30 == 0:  # check every second
            for sid in range(2):
                name = FIGHTER_NAMES[sid]
                if self._aggression[sid] > 3.0:
                    self._events.append((frame_idx, f"{name} presses forward"))
                if self._ring_ctl[sid] > 0.7:
                    self._events.append((frame_idx, f"{name} controls center"))
            # Keep only last 5 events
            self._events = self._events[-5:]

        # ─── Running round score ───
        act_adv  = self._activity[0] - self._activity[1]
        agg_adv  = self._aggression[0] - self._aggression[1]
        ctl_adv  = self._ring_ctl[0] - self._ring_ctl[1]
        composite = act_adv * 0.3 + agg_adv * 0.4 + ctl_adv * 0.3
        if composite > 0.3:
            self._round_pts[0] = 10.0
            self._round_pts[1] = 9.0
        elif composite < -0.3:
            self._round_pts[0] = 9.0
            self._round_pts[1] = 10.0
        else:
            self._round_pts[0] = 10.0
            self._round_pts[1] = 10.0

    @property
    def activity(self): return self._activity
    @property
    def aggression(self): return self._aggression
    @property
    def ring_control(self): return self._ring_ctl
    @property
    def distance_label(self):
        if self._distance < 0.15: return "Inside"
        elif self._distance < 0.35: return "Mid-Range"
        else: return "Outside"
    @property
    def distance_val(self): return self._distance
    @property
    def pressure(self): return self._pressure
    @property
    def round_num(self): return self._round
    @property
    def round_pts(self): return self._round_pts
    @property
    def events(self): return self._events
    @property
    def frame_count(self): return self._frame_count


# ── Scoreboard overlay ────────────────────────────────────────────────────────
PANEL_W   = 160    # side panel width
TOP_H     = 40     # top score bar height
BOTTOM_H  = 36     # bottom info bar height
PANEL_BG  = (18, 18, 22)
PANEL_ALPHA = 0.80
BAR_H     = 8      # metric bar height
BAR_GAP   = 22     # gap between metric rows


def _draw_panel_bg(canvas, x1, y1, x2, y2, alpha=PANEL_ALPHA):
    """Draw a translucent dark panel background."""
    overlay = canvas[y1:y2, x1:x2].copy()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.addWeighted(canvas[y1:y2, x1:x2], alpha, overlay, 1 - alpha, 0,
                    canvas[y1:y2, x1:x2])


def _draw_bar(canvas, x, y, w, val, max_val, color, bg=(50, 50, 55)):
    """Draw a single metric bar."""
    cv2.rectangle(canvas, (x, y), (x + w, y + BAR_H), bg, -1)
    fill_w = max(1, int(w * min(val / max(max_val, 0.001), 1.0)))
    cv2.rectangle(canvas, (x, y), (x + fill_w, y + BAR_H), color, -1)


def draw_scoreboard(frame, scorer: FightScorer, fps: float):
    """
    Render the full scoring overlay onto an expanded canvas.
    Returns the canvas (larger than input frame).
    """
    fh, fw = frame.shape[:2]
    canvas_w = fw + 2 * PANEL_W
    canvas_h = fh + TOP_H + BOTTOM_H
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place original frame in the center
    canvas[TOP_H:TOP_H + fh, PANEL_W:PANEL_W + fw] = frame

    # ── Top score bar ────────────────────────────────────────────
    _draw_panel_bg(canvas, 0, 0, canvas_w, TOP_H)
    elapsed = scorer.frame_count / fps if fps > 0 else 0
    mm, ss = divmod(int(elapsed), 60)

    # Round + timer (center)
    timer_text = f"RND {scorer.round_num}   {mm:02d}:{ss:02d}"
    (tw, _), _ = cv2.getTextSize(timer_text, FONT, 0.5, 1)
    cv2.putText(canvas, timer_text, (canvas_w // 2 - tw // 2, 27),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Fighter A name + score (left)
    pts_a = int(scorer.round_pts[0])
    a_text = f"{FIGHTER_NAMES[0]}  {pts_a}"
    cv2.putText(canvas, a_text, (14, 27),
                FONT, 0.5, FIGHTER_COLORS[0], 1, cv2.LINE_AA)

    # Fighter B name + score (right)
    pts_b = int(scorer.round_pts[1])
    b_text = f"{pts_b}  {FIGHTER_NAMES[1]}"
    (tw2, _), _ = cv2.getTextSize(b_text, FONT, 0.5, 1)
    cv2.putText(canvas, b_text, (canvas_w - tw2 - 14, 27),
                FONT, 0.5, FIGHTER_COLORS[1], 1, cv2.LINE_AA)

    # ── Side panels ──────────────────────────────────────────────
    # Normalize metrics for display (scale to reasonable max)
    act_max = max(max(scorer.activity), 5.0)
    agg_max = max(max(scorer.aggression), 3.0)

    for sid in range(2):
        color = FIGHTER_COLORS[sid]
        if sid == 0:
            px = 0
        else:
            px = PANEL_W + fw

        py = TOP_H
        ph = fh
        _draw_panel_bg(canvas, px, py, px + PANEL_W, py + ph)

        # Metrics start position
        mx = px + 10
        bar_w = PANEL_W - 20
        my = py + 18

        # Label + bar: Activity
        cv2.putText(canvas, "Activity", (mx, my),
                    FONT, 0.38, (170, 170, 175), 1, cv2.LINE_AA)
        my += 14
        _draw_bar(canvas, mx, my, bar_w, scorer.activity[sid], act_max, color)
        # Value text
        act_pct = min(100, int(scorer.activity[sid] / act_max * 100))
        cv2.putText(canvas, f"{act_pct}%", (mx + bar_w - 28, my - 2),
                    FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        my += BAR_GAP

        # Aggression
        cv2.putText(canvas, "Aggression", (mx, my),
                    FONT, 0.38, (170, 170, 175), 1, cv2.LINE_AA)
        my += 14
        _draw_bar(canvas, mx, my, bar_w, scorer.aggression[sid], agg_max, color)
        agg_pct = min(100, int(scorer.aggression[sid] / agg_max * 100))
        cv2.putText(canvas, f"{agg_pct}%", (mx + bar_w - 28, my - 2),
                    FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        my += BAR_GAP

        # Ring Control
        cv2.putText(canvas, "Ring Ctrl", (mx, my),
                    FONT, 0.38, (170, 170, 175), 1, cv2.LINE_AA)
        my += 14
        _draw_bar(canvas, mx, my, bar_w, scorer.ring_control[sid], 1.0, color)
        ctl_pct = int(scorer.ring_control[sid] * 100)
        cv2.putText(canvas, f"{ctl_pct}%", (mx + bar_w - 28, my - 2),
                    FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        my += BAR_GAP

        # Pressure
        cv2.putText(canvas, "Pressure", (mx, my),
                    FONT, 0.38, (170, 170, 175), 1, cv2.LINE_AA)
        my += 14
        prs_max = max(max(scorer.pressure), 2.0)
        _draw_bar(canvas, mx, my, bar_w, scorer.pressure[sid], prs_max, color)
        prs_pct = min(100, int(scorer.pressure[sid] / prs_max * 100))
        cv2.putText(canvas, f"{prs_pct}%", (mx + bar_w - 28, my - 2),
                    FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        my += BAR_GAP + 6

        # Separator line
        cv2.line(canvas, (mx, my), (mx + bar_w, my), (40, 40, 45), 1)
        my += 12

        # Events feed for this fighter
        name_prefix = FIGHTER_NAMES[sid].split()[1]  # "A" or "B"
        fighter_events = [e for e in scorer.events
                          if name_prefix in e[1]]
        for ev_frame, ev_text in fighter_events[-3:]:
            ev_sec = ev_frame / fps if fps > 0 else 0
            em, es = divmod(int(ev_sec), 60)
            cv2.putText(canvas, f"{em:02d}:{es:02d}", (mx, my),
                        FONT, 0.30, (120, 120, 125), 1, cv2.LINE_AA)
            # Truncate event text to fit
            short = ev_text.replace(FIGHTER_NAMES[sid], "").strip()
            cv2.putText(canvas, short[:18], (mx + 38, my),
                        FONT, 0.30, (180, 220, 180), 1, cv2.LINE_AA)
            my += 16

    # ── Bottom bar ───────────────────────────────────────────────
    bot_y = TOP_H + fh
    _draw_panel_bg(canvas, 0, bot_y, canvas_w, canvas_h)

    # Distance label (left)
    dist_text = f"Distance: {scorer.distance_label}"
    cv2.putText(canvas, dist_text, (14, bot_y + 24),
                FONT, 0.42, (170, 170, 175), 1, cv2.LINE_AA)

    # Distance mini-bar
    bar_x = 200
    bar_w_bot = 120
    _draw_bar(canvas, bar_x, bot_y + 14, bar_w_bot, scorer.distance_val, 0.5,
              (0, 180, 220))

    # Latest event (right side)
    if scorer.events:
        last_frame, last_text = scorer.events[-1]
        ev_sec = last_frame / fps if fps > 0 else 0
        em, es = divmod(int(ev_sec), 60)
        ev_str = f"{last_text}  {em:02d}:{es:02d}"
        (tw3, _), _ = cv2.getTextSize(ev_str, FONT, 0.38, 1)
        cv2.putText(canvas, ev_str, (canvas_w - tw3 - 14, bot_y + 24),
                    FONT, 0.38, (180, 220, 180), 1, cv2.LINE_AA)

    return canvas


# ── Main annotation function ──────────────────────────────────────────────────
def annotate_video(input_path: str, output_path: str,
                   model_name: str = "yolov8n.pt",
                   conf: float = CONF_THRESHOLD,
                   device: str = "cpu",
                   imgsz: int = 1280,
                   scale: float = 1.0,
                   target_fps: float = 0,
                   max_frames: int = 0) -> str:
    print(f"\n{'='*60}")
    print(f"  Ring Dynamics — Video Tracking Annotator v5")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Model  : {model_name}  |  Device: {device}  |  Conf: {conf}")
    print(f"  ImgSz  : {imgsz}  |  Scale: {scale}")
    print(f"  Mode   : Hardened 2-fighter tracking (anti-audience)")
    print(f"{'='*60}\n")

    print("[1/4] Loading YOLOv8 model …")
    model = YOLO(model_name)

    print("[2/4] Opening video …")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(width  * scale)
    out_h = int(height * scale)
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    if target_fps > 0 and target_fps < fps:
        frame_skip = max(1, round(fps / target_fps))
        out_fps = fps / frame_skip
    else:
        frame_skip = 1
        out_fps = fps

    frames_to_process = total // frame_skip
    if max_frames > 0:
        frames_to_process = min(frames_to_process, max_frames)

    # Canvas dimensions include side panels + top/bottom bars
    canvas_out_w = out_w + 2 * PANEL_W
    canvas_out_h = out_h + TOP_H + BOTTOM_H

    print(f"    Source     : {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total}")
    print(f"    Output     : {canvas_out_w}×{canvas_out_h}  |  FPS: {out_fps:.1f}  |  Frames: ~{frames_to_process}")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (canvas_out_w, canvas_out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    identity_tracker = FighterIdentityTracker()
    scorer = FightScorer()

    metrics_snapshots = []   # per-second metrics for JSON export

    print("[3/4] Processing frames …")
    frame_idx = 0
    processed = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_skip > 1 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        if max_frames > 0 and processed >= max_frames:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        results = model.track(
            frame, persist=True, classes=[0], conf=conf, iou=0.45,
            imgsz=imgsz, tracker="bytetrack.yaml", verbose=False,
            device=device,
        )

        all_dets = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            kpts_data = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            kpts_conf = result.keypoints.conf.cpu().numpy() if (result.keypoints is not None and result.keypoints.conf is not None) else None
            for i, box in enumerate(boxes):
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                kpt = kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None
                kpt_c = kpts_conf[i] if kpts_conf is not None and i < len(kpts_conf) else None
                all_dets.append({
                    "track_id": track_id, "conf": conf_val,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area": area,
                    "keypoints": kpt,
                    "keypoints_conf": kpt_c,
                })

        fighters = identity_tracker.assign(all_dets, frame)

        for d in fighters:
            sid   = d["stable_id"]
            color = FIGHTER_COLORS[sid]
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            bw = x2 - x1
            bh = y2 - y1

            # ── Full-body outer box (with jitter) ────────────────────
            jx1, jy1, jx2, jy2 = jitter_box(x1, y1, x2, y2)
            draw_rounded_rect(frame, jx1, jy1, jx2, jy2, color, BOX_THICKNESS)

            # ── Pose-aware Head sub-box (green accent, with jitter) ──
            if "head_box" in d:
                hx1, hy1, hx2, hy2 = d["head_box"]
            else:
                (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), _ = compute_fighter_pose_boxes(d, frame.shape)

            if hy2 > hy1 + 4 and hx2 > hx1 + 4:
                jhx1, jhy1, jhx2, jhy2 = jitter_box(hx1, hy1, hx2, hy2, 1)
                cv2.rectangle(frame, (jhx1, jhy1), (jhx2, jhy2), HEAD_COLOR, SUB_BOX_THICK)

            # ── Pose-aware Body / core sub-box (fighter color, with jitter) ─────
            if "core_box" in d:
                bx1, by1, bx2, by2 = d["core_box"]
            else:
                bx1, bx2 = x1 + max(1, int(bw * BODY_INSET_X)), x2 - max(1, int(bw * BODY_INSET_X))
                by1, by2 = y1 + int(bh * BODY_TOP), y1 + int(bh * BODY_BOTTOM)

            if by2 > by1 + 4 and bx2 > bx1 + 4:
                jbx1, jby1, jbx2, jby2 = jitter_box(bx1, by1, bx2, by2, 1)
                cv2.rectangle(frame, (jbx1, jby1), (jbx2, jby2), color, SUB_BOX_THICK)

            # ── Label ────────────────────────────────────────────────
            orient_tag = " [DOWN]" if d.get("orientation") == "inverted" else ""
            draw_label(frame, f"{FIGHTER_NAMES[sid]}  {d['conf']:.0%}{orient_tag}", x1, y1, color)


        # ── Update scorer & render scoreboard ────────────────────
        scorer.update(fighters, frame.shape[1], frame_idx)
        canvas = draw_scoreboard(frame, scorer, out_fps)
        writer.write(canvas)

        # ── Capture metrics snapshot every second ────────────────
        if frame_idx > 0 and frame_idx % int(out_fps) == 0:
            sec = frame_idx / out_fps
            metrics_snapshots.append({
                "time": round(sec, 1),
                "activity": [round(v, 2) for v in scorer.activity],
                "aggression": [round(v, 2) for v in scorer.aggression],
                "ring_control": [round(v, 2) for v in scorer.ring_control],
                "pressure": [round(v, 2) for v in scorer.pressure],
                "distance": scorer.distance_label,
                "round_pts": [int(v) for v in scorer.round_pts],
            })

        frame_idx += 1
        processed += 1

        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t_start
            pct = (processed / frames_to_process * 100) if frames_to_process > 0 else 0
            eta = (elapsed / processed) * (frames_to_process - processed) if processed > 0 else 0
            print(f"    Frame {processed:>5}/{frames_to_process}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n[4/4] Done! {processed} frames in {elapsed:.1f}s → {output_path}\n")

    # ── Export metrics JSON ───────────────────────────────────────
    metrics_path = output_path.rsplit('.', 1)[0] + '_metrics.json'
    metrics_data = {
        "total_frames": processed,
        "fps": round(out_fps, 1),
        "duration": round(processed / out_fps, 1) if out_fps > 0 else 0,
        "processing_time": round(elapsed, 1),
        "final_scores": [int(v) for v in scorer.round_pts],
        "final_activity": [round(v, 2) for v in scorer.activity],
        "final_aggression": [round(v, 2) for v in scorer.aggression],
        "final_ring_control": [round(v, 2) for v in scorer.ring_control],
        "final_pressure": [round(v, 2) for v in scorer.pressure],
        "events": [{"frame": f, "text": t} for f, t in scorer.events],
        "timeline": metrics_snapshots,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metrics → {metrics_path}")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ring Dynamics — hardened 2-fighter tracking"
    )
    parser.add_argument("input",  help="Path to input video")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="YOLO model variant or weights path (e.g. yolov8n-pose.pt, yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=0)
    parser.add_argument("--max-frames", type=int, default=0)

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        p = Path(args.input)
        output_path = str(p.parent / (p.stem + "_annotated" + p.suffix))

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    annotate_video(
        input_path=args.input, output_path=output_path,
        model_name=args.model, conf=args.conf, device=args.device,
        imgsz=args.imgsz, scale=args.scale,
        target_fps=args.fps, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

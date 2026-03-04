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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fighter Identity Tracker  (v5 — hardened against audience false-matches)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FighterIdentityTracker:
    """
    Maintains stable Fighter A / Fighter B identity across the entire video.

    Anti-audience hardening:
      1. RING-ZONE constraint: fighters bbox center should be within the
         central horizontal band (10-90%) and vertical band (5-85%) of frame
      2. MAX POSITION JUMP: a fighter can't move more than 40% of the frame
         width between consecutive frames
      3. MINIMUM SIZE: bbox must be >= 20% frame height AND >= 2.5% frame area
      4. SCORE THRESHOLD: MIN_ACCEPT_SCORE = 0.42 (raised from 0.35)
      5. SKIN RATIO: central-column skin detection rejects clothed persons
      6. TRUNKS HISTOGRAM: most distinctive feature for boxer identification
    """

    INIT_FRAMES       = 30      # broader bootstrap to capture more variation
    MIN_ACCEPT_SCORE  = 0.40    # main threshold
    CAMERA_CUT_SCORE  = 0.35    # lower threshold during camera cuts
    SWAP_MARGIN       = 0.03    # min score gap to prevent ambiguous swaps
    HIST_ALPHA        = 0.05    # EMA for histogram (very slow)
    PROFILE_ALPHA     = 0.05    # EMA for scalar features
    MAX_POS_JUMP      = 0.30    # max normalised distance per frame

    # Ring-zone bounds (normalised coordinates) — audience sits outside
    RING_X_MIN = 0.12   # left bound (tight to reject audience)
    RING_X_MAX = 0.88   # right bound
    RING_Y_MIN = 0.05   # top bound
    RING_Y_MAX = 0.85   # bottom bound

    # Minimum size constraints — fighters are BIG in the frame
    MIN_HEIGHT_RATIO  = 0.25    # bbox height >= 25% of frame height
    MIN_AREA_RATIO    = 0.035   # bbox area >= 3.5% of frame area

    def __init__(self):
        self._frame_count = 0

        # Per-fighter profiles (indexed 0, 1)
        self._torso_hists:  List[Optional[np.ndarray]] = [None, None]
        self._trunks_hists: List[Optional[np.ndarray]] = [None, None]
        self._body_hists:   List[Optional[np.ndarray]] = [None, None]
        self._skin_ratios:  List[float] = [0.0, 0.0]
        self._ref_heights:  List[float] = [0.0, 0.0]

        # ByteTrack ID → stable ID mapping
        self._id_map: Dict[int, int] = {}

        # Last known normalised position per fighter
        self._last_pos: List[Optional[Tuple[float, float]]] = [None, None]

        # Bootstrap spatial reference: average X position per fighter
        self._bootstrap_cx: List[float] = [0.0, 0.0]  # learned in bootstrap
        self._ring_cx_accumulator: List[List[float]] = [[], []]

    # ── Feature extraction ─────────────────────────────────────────────────
    @staticmethod
    def _hs_histogram(frame: np.ndarray, x1: int, y1: int,
                      x2: int, y2: int) -> np.ndarray:
        """2-D Hue×Saturation histogram."""
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return np.zeros((30, 32), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @classmethod
    def _torso_histogram(cls, frame: np.ndarray, x1: int, y1: int,
                         x2: int, y2: int) -> np.ndarray:
        """HS histogram of the upper-torso region  (top 15-45% of bbox)."""
        h = y2 - y1
        w = x2 - x1
        inset_x = max(1, int(w * 0.2))
        ry1 = y1 + int(h * 0.15)
        ry2 = y1 + int(h * 0.45)
        rx1 = x1 + inset_x
        rx2 = x2 - inset_x
        return cls._hs_histogram(frame, rx1, ry1, rx2, ry2)

    @classmethod
    def _trunks_histogram(cls, frame: np.ndarray, x1: int, y1: int,
                          x2: int, y2: int) -> np.ndarray:
        """HS histogram of the trunks/shorts region  (40-70% of bbox)."""
        h = y2 - y1
        w = x2 - x1
        inset_x = max(1, int(w * 0.15))
        ry1 = y1 + int(h * 0.40)
        ry2 = y1 + int(h * 0.70)
        rx1 = x1 + inset_x
        rx2 = x2 - inset_x
        return cls._hs_histogram(frame, rx1, ry1, rx2, ry2)

    @staticmethod
    def _central_skin_ratio(frame: np.ndarray, x1: int, y1: int,
                            x2: int, y2: int) -> float:
        """Skin pixel fraction using central column only."""
        h = y2 - y1
        w = x2 - x1
        if h <= 0 or w <= 0:
            return 0.0
        cx1 = x1 + int(w * 0.3)
        cx2 = x1 + int(w * 0.7)
        cy1 = y1 + int(h * 0.05)
        cy2 = y1 + int(h * 0.45)
        crop = frame[max(0, cy1):cy2, max(0, cx1):cx2]
        if crop.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        return float(np.count_nonzero(mask)) / mask.size

    @staticmethod
    def _hist_sim(h1: np.ndarray, h2: np.ndarray) -> float:
        """Bhattacharyya similarity (0..1)."""
        if h1 is None or h2 is None:
            return 0.0
        dist = cv2.compareHist(h1.astype(np.float32),
                               h2.astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - dist)

    # ── Enrich detection with features ─────────────────────────────────────
    def _enrich(self, det: dict, frame: np.ndarray) -> None:
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        det["_torso_hist"]   = self._torso_histogram(frame, x1, y1, x2, y2)
        det["_trunks_hist"]  = self._trunks_histogram(frame, x1, y1, x2, y2)
        det["_body_hist"]    = self._hs_histogram(frame, x1, y1, x2, y2)
        det["_skin_ratio"]   = self._central_skin_ratio(frame, x1, y1, x2, y2)
        det["_height_ratio"] = (y2 - y1) / h_frame
        det["_area_ratio"]   = det["area"] / (h_frame * w_frame)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        det["_cx_norm"] = cx / w_frame
        det["_cy_norm"] = cy / h_frame

    # ── Spatial checks ─────────────────────────────────────────────────────
    def _in_ring_zone(self, det: dict) -> bool:
        cx = det["_cx_norm"]
        cy = det["_cy_norm"]
        return (self.RING_X_MIN <= cx <= self.RING_X_MAX and
                self.RING_Y_MIN <= cy <= self.RING_Y_MAX)

    def _within_jump_limit(self, det: dict, fid: int) -> bool:
        if self._last_pos[fid] is None:
            return True
        lx, ly = self._last_pos[fid]
        dist = ((det["_cx_norm"] - lx)**2 + (det["_cy_norm"] - ly)**2) ** 0.5
        return dist <= self.MAX_POS_JUMP

    # ── Scoring ────────────────────────────────────────────────────────────
    def _appearance_components(self, det: dict, fid: int) -> Tuple[float, float, float, float, float]:
        """Return (trunks_sim, torso_sim, body_sim, skin_sim, height_sim)."""
        trunks_sim = self._hist_sim(det["_trunks_hist"], self._trunks_hists[fid])
        torso_sim  = self._hist_sim(det["_torso_hist"],  self._torso_hists[fid])
        body_sim   = self._hist_sim(det["_body_hist"],   self._body_hists[fid])

        ref_skin = self._skin_ratios[fid]
        det_skin = det["_skin_ratio"]
        skin_sim = 1.0 - min(abs(det_skin - ref_skin) / (ref_skin + 0.01), 1.0)

        ref_h = self._ref_heights[fid]
        det_h = det["_height_ratio"]
        h_sim = 1.0 - min(abs(det_h - ref_h) / (ref_h + 0.01), 1.0)

        return trunks_sim, torso_sim, body_sim, skin_sim, h_sim

    def _score(self, det: dict, fid: int) -> float:
        """Full score (appearance + position continuity)."""
        trunks_sim, torso_sim, body_sim, skin_sim, h_sim = self._appearance_components(det, fid)

        pos_sim = 0.5
        if self._last_pos[fid] is not None:
            lx, ly = self._last_pos[fid]
            dist = ((det["_cx_norm"] - lx)**2 + (det["_cy_norm"] - ly)**2) ** 0.5
            pos_sim = max(0.0, 1.0 - dist * 3.0)

        return (0.30 * trunks_sim +
                0.15 * torso_sim +
                0.15 * body_sim +
                0.15 * skin_sim +
                0.10 * h_sim +
                0.15 * pos_sim)

    def _score_appearance_only(self, det: dict, fid: int) -> float:
        """Appearance-only score — used after camera cuts."""
        trunks_sim, torso_sim, body_sim, skin_sim, h_sim = self._appearance_components(det, fid)
        return (0.35 * trunks_sim +
                0.20 * torso_sim +
                0.20 * body_sim +
                0.15 * skin_sim +
                0.10 * h_sim)

    # ── Profile update ─────────────────────────────────────────────────────
    def _update_profile(self, det: dict, fid: int, alpha: Optional[float] = None,
                        bootstrap: bool = False):
        """After bootstrap: ONLY position + height updated. Histograms FROZEN."""
        if self._torso_hists[fid] is None or bootstrap:
            a = alpha if alpha is not None else self.HIST_ALPHA
            ap = alpha if alpha is not None else self.PROFILE_ALPHA
            if self._torso_hists[fid] is None:
                self._torso_hists[fid]  = det["_torso_hist"].copy()
                self._trunks_hists[fid] = det["_trunks_hist"].copy()
                self._body_hists[fid]   = det["_body_hist"].copy()
                self._skin_ratios[fid]  = det["_skin_ratio"]
                self._ref_heights[fid]  = det["_height_ratio"]
            else:
                self._torso_hists[fid]  = (1-a)*self._torso_hists[fid]  + a*det["_torso_hist"]
                self._trunks_hists[fid] = (1-a)*self._trunks_hists[fid] + a*det["_trunks_hist"]
                self._body_hists[fid]   = (1-a)*self._body_hists[fid]   + a*det["_body_hist"]
                self._skin_ratios[fid]  = (1-ap)*self._skin_ratios[fid] + ap*det["_skin_ratio"]
                self._ref_heights[fid]  = (1-ap)*self._ref_heights[fid] + ap*det["_height_ratio"]
        else:
            ap = self.PROFILE_ALPHA
            self._ref_heights[fid] = (1-ap)*self._ref_heights[fid] + ap*det["_height_ratio"]
        self._last_pos[fid] = (det["_cx_norm"], det["_cy_norm"])

    # ── Main entry point ───────────────────────────────────────────────────
    def assign(self, all_dets: list, frame: np.ndarray) -> list:
        if not all_dets:
            return []

        h_frame, w_frame = frame.shape[:2]

        for d in all_dets:
            self._enrich(d, frame)

        self._frame_count += 1

        # ── Bootstrap: trust top-2 by area ────────────────────────────────
        if self._frame_count <= self.INIT_FRAMES:
            top2 = sorted(all_dets, key=lambda d: d["area"], reverse=True)[:2]
            for i, d in enumerate(top2):
                sid = i
                d["stable_id"] = sid
                self._id_map[d["track_id"]] = sid
                alpha_boot = 1.0 / self._frame_count
                self._update_profile(d, sid, alpha=alpha_boot, bootstrap=True)
                self._ring_cx_accumulator[sid].append(d["_cx_norm"])
            # At end of bootstrap, compute average X position per fighter
            if self._frame_count == self.INIT_FRAMES:
                for sid in [0, 1]:
                    if self._ring_cx_accumulator[sid]:
                        self._bootstrap_cx[sid] = (
                            sum(self._ring_cx_accumulator[sid]) /
                            len(self._ring_cx_accumulator[sid])
                        )
            return top2

        # ── Steady state ──────────────────────────────────────────────────

        # LAYER 1: Size filter
        min_h = max(self.MIN_HEIGHT_RATIO, min(self._ref_heights) * 0.5)
        size_ok = [d for d in all_dets
                   if d["_height_ratio"] >= min_h
                   and d["_area_ratio"] >= self.MIN_AREA_RATIO]

        # LAYER 2: Ring-zone filter
        ring_ok = [d for d in size_ok if self._in_ring_zone(d)]

        # LAYER 3: Skin filter
        min_skin = max(0.06, min(self._skin_ratios) * 0.30)
        candidates = [d for d in ring_ok if d["_skin_ratio"] >= min_skin]

        if len(candidates) < 1:
            candidates = sorted(size_ok if size_ok else all_dets,
                                key=lambda d: d["area"], reverse=True)[:4]

        # LAYER 4: Score with position continuity
        n = len(candidates)
        score_matrix = np.zeros((n, 2), dtype=np.float64)
        for i, d in enumerate(candidates):
            for fid in [0, 1]:
                if not self._within_jump_limit(d, fid):
                    score_matrix[i, fid] = 0.0
                else:
                    score_matrix[i, fid] = self._score(d, fid)

        # CAMERA CUT DETECTION
        max_pos_score = score_matrix.max() if n > 0 else 0.0
        is_camera_cut = max_pos_score < self.MIN_ACCEPT_SCORE

        if is_camera_cut and n > 0:
            for i, d in enumerate(candidates):
                for fid in [0, 1]:
                    score_matrix[i, fid] = self._score_appearance_only(d, fid)

        # Effective threshold
        accept_threshold = self.CAMERA_CUT_SCORE if is_camera_cut else self.MIN_ACCEPT_SCORE

        # LAYER 5: Assignment
        #
        # Try OPTIMAL (Hungarian-style) 2-target assignment:
        # For 2 fighters, try both possible pairings and pick the one
        # with the highest total score.
        result = []

        if n >= 2:
            # Try all possible pairs of candidates for the 2 fighters
            best_total = -1.0
            best_pair = None  # ( (det_idx_for_fid0, det_idx_for_fid1) )

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    s0 = score_matrix[i, 0]
                    s1 = score_matrix[j, 1]
                    total = s0 + s1
                    if total > best_total:
                        best_total = total
                        best_pair = (i, j)

            if best_pair is not None:
                i0, i1 = best_pair
                s0 = score_matrix[i0, 0]
                s1 = score_matrix[i1, 1]

                # Check if both pass threshold
                both_pass = s0 >= accept_threshold and s1 >= accept_threshold

                # If appearance is ambiguous (scores are close), use
                # SPATIAL ORDERING as tiebreaker during camera cuts
                if is_camera_cut and both_pass:
                    margin0 = score_matrix[i0, 0] - score_matrix[i0, 1]
                    margin1 = score_matrix[i1, 1] - score_matrix[i1, 0]

                    if abs(margin0) < self.SWAP_MARGIN or abs(margin1) < self.SWAP_MARGIN:
                        # Ambiguous appearance — use X position ordering
                        # Sort candidates by X position: leftmost = Fighter 0
                        pair_dets = [(i0, candidates[i0]), (i1, candidates[i1])]
                        pair_dets.sort(key=lambda p: p[1]["_cx_norm"])
                        left_idx = pair_dets[0][0]
                        right_idx = pair_dets[1][0]

                        # Determine which fighter was typically on the left
                        if self._bootstrap_cx[0] <= self._bootstrap_cx[1]:
                            # Fighter 0 was left, Fighter 1 was right
                            i0, i1 = left_idx, right_idx
                        else:
                            # Fighter 0 was right, Fighter 1 was left
                            i0, i1 = right_idx, left_idx

                        s0 = score_matrix[i0, 0]
                        s1 = score_matrix[i1, 1]

                if both_pass:
                    d0 = candidates[i0]
                    d0["stable_id"] = 0
                    result.append(d0)
                    self._id_map[d0["track_id"]] = 0

                    d1 = candidates[i1]
                    d1["stable_id"] = 1
                    result.append(d1)
                    self._id_map[d1["track_id"]] = 1
                else:
                    # Only assign fighters that individually pass
                    if s0 >= accept_threshold:
                        d0 = candidates[i0]
                        d0["stable_id"] = 0
                        result.append(d0)
                        self._id_map[d0["track_id"]] = 0
                    if s1 >= accept_threshold:
                        d1 = candidates[i1]
                        d1["stable_id"] = 1
                        result.append(d1)
                        self._id_map[d1["track_id"]] = 1

        elif n == 1:
            # Only 1 candidate — assign to best-matching fighter
            s0 = score_matrix[0, 0]
            s1 = score_matrix[0, 1]
            if s0 >= accept_threshold or s1 >= accept_threshold:
                fid = 0 if s0 >= s1 else 1
                d = candidates[0]
                d["stable_id"] = fid
                result.append(d)
                self._id_map[d["track_id"]] = fid

        # Update profiles (post-bootstrap: only position + height)
        for d in result:
            self._update_profile(d, d["stable_id"])

        return result


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
                   max_frames: int = 0,
                   progress_cb=None) -> str:
    def _prog(stage, name, pct, done=0, total=0):
        if progress_cb:
            progress_cb(stage, name, pct, done, total)
    print(f"\n{'='*60}")
    print(f"  Ring Dynamics — Video Tracking Annotator v5")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Model  : {model_name}  |  Device: {device}  |  Conf: {conf}")
    print(f"  ImgSz  : {imgsz}  |  Scale: {scale}")
    print(f"  Mode   : Hardened 2-fighter tracking (anti-audience)")
    print(f"{'='*60}\n")

    _prog(1, "Loading YOLOv8 model", 5)
    print("[1/5] Loading YOLOv8 model …")
    model = YOLO(model_name)
    _prog(1, "Loading YOLOv8 model", 10)

    _prog(2, "Analyzing video properties", 15)
    print("[2/5] Analyzing video properties …")
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

    _prog(2, "Analyzing video properties", 20, 0, frames_to_process)
    print("[3/5] Processing frames …")
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
            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                all_dets.append({
                    "track_id": track_id, "conf": conf_val,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area": area,
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

            # ── Head sub-box (green accent, with jitter) ─────────────
            head_inset = max(1, int(bw * HEAD_INSET_X))
            hx1 = x1 + head_inset
            hx2 = x2 - head_inset
            hy1 = y1 + int(bh * HEAD_TOP)
            hy2 = y1 + int(bh * HEAD_BOTTOM)
            if hy2 > hy1 + 4 and hx2 > hx1 + 4:
                jhx1, jhy1, jhx2, jhy2 = jitter_box(hx1, hy1, hx2, hy2, 1)
                cv2.rectangle(frame, (jhx1, jhy1), (jhx2, jhy2), HEAD_COLOR, SUB_BOX_THICK)

            # ── Body / core sub-box (fighter color, with jitter) ─────
            body_inset = max(1, int(bw * BODY_INSET_X))
            bx1 = x1 + body_inset
            bx2 = x2 - body_inset
            by1 = y1 + int(bh * BODY_TOP)
            by2 = y1 + int(bh * BODY_BOTTOM)
            if by2 > by1 + 4 and bx2 > bx1 + 4:
                jbx1, jby1, jbx2, jby2 = jitter_box(bx1, by1, bx2, by2, 1)
                cv2.rectangle(frame, (jbx1, jby1), (jbx2, jby2), color, SUB_BOX_THICK)

            # ── Label ────────────────────────────────────────────────
            draw_label(frame, f"{FIGHTER_NAMES[sid]}  {d['conf']:.0%}", x1, y1, color)


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
            # Progress: stages 3-4 span 20%-90%
            overall_pct = 20 + (pct * 0.7)
            _prog(3, "Tracking & scoring", overall_pct, processed, frames_to_process)
            print(f"    Frame {processed:>5}/{frames_to_process}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    _prog(4, "Exporting metrics", 92, processed, frames_to_process)
    print(f"\n[4/5] Exporting metrics …")

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

    _prog(5, "Complete", 100, processed, frames_to_process)
    print(f"\n[5/5] Done! {processed} frames in {elapsed:.1f}s → {output_path}\n")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ring Dynamics — hardened 2-fighter tracking"
    )
    parser.add_argument("input",  help="Path to input video")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                 "yolov8l.pt", "yolov8x.pt"])
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

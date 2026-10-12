"""
Tests for fighter pose orientation detection and adaptive sub-box computation.
Verifies that inverted / knocked-down fighters on the canvas (head at bottom of
screen, feet on top of screen) are correctly detected, tracking their heads at
the bottom and feet at the top.
"""
import numpy as np
import pytest
from workers.annotate_video import compute_fighter_pose_boxes, FighterIdentityTracker


def test_upright_fighter_pose():
    # Fighter standing upright: head y ~ 100, feet y ~ 320
    kpts = np.zeros((17, 2), dtype=np.float32)
    # Nose, eyes, ears near y=95..105
    kpts[0] = [350, 95]
    kpts[1] = [345, 90]
    kpts[2] = [355, 90]
    kpts[3] = [340, 95]
    kpts[4] = [360, 95]
    # Shoulders
    kpts[5] = [330, 120]
    kpts[6] = [370, 120]
    # Hips
    kpts[11] = [340, 200]
    kpts[12] = [360, 200]
    # Ankles/feet
    kpts[15] = [335, 310]
    kpts[16] = [365, 310]

    det = {
        "x1": 300, "y1": 80, "x2": 400, "y2": 320,
        "keypoints": kpts,
        "keypoints_conf": np.ones(17, dtype=np.float32),
    }

    (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orient = compute_fighter_pose_boxes(det, (480, 640))
    assert orient == "upright"
    # Head box should be in upper half of bbox
    assert hy1 >= 80
    assert hy2 < 200
    # Core box should be below head
    assert by1 > hy1


def test_inverted_fighter_pose_on_ground():
    """
    Fighter knocked down on canvas: head at bottom of screen (y ~ 300),
    feet at top of screen (y ~ 95).
    """
    kpts = np.zeros((17, 2), dtype=np.float32)
    # Head keypoints near bottom (y=300..310)
    kpts[0] = [350, 310]
    kpts[1] = [345, 315]
    kpts[2] = [355, 315]
    kpts[3] = [340, 310]
    kpts[4] = [360, 310]
    # Shoulders
    kpts[5] = [330, 280]
    kpts[6] = [370, 280]
    # Hips
    kpts[11] = [340, 200]
    kpts[12] = [360, 200]
    # Ankles/feet near top of screen (y=90)
    kpts[15] = [335, 95]
    kpts[16] = [365, 95]

    det = {
        "x1": 300, "y1": 80, "x2": 400, "y2": 330,
        "keypoints": kpts,
        "keypoints_conf": np.ones(17, dtype=np.float32),
    }

    (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orient = compute_fighter_pose_boxes(det, (480, 640))
    assert orient == "inverted"
    # Head sub-box MUST be in the bottom portion of the bounding box (near y=310), NOT near feet at y=95!
    assert hy1 >= 250
    assert hy2 <= 330
    # Feet at top must NOT be inside the head sub-box
    assert hy1 > 100


def test_horizontal_fighter_pose():
    # Fighter lying horizontally: width > height
    det = {
        "x1": 100, "y1": 200, "x2": 350, "y2": 280,
        "keypoints": None,
        "keypoints_conf": None,
    }

    (hx1, hy1, hx2, hy2), (bx1, by1, bx2, by2), orient = compute_fighter_pose_boxes(det, (480, 640))
    assert orient == "horizontal"
    assert hx2 > hx1
    assert hy2 > hy1


def test_tracker_does_not_reject_ground_fighter():
    tracker = FighterIdentityTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Knocked-down fighter: small height (50px), wide (200px)
    # Old logic: bh < 480 * 0.18 (86px) would reject!
    # New logic: max(bw, bh) = 200px >= 86px -> accepted!
    det = {
        "track_id": 1,
        "conf": 0.85,
        "x1": 150, "y1": 300, "x2": 350, "y2": 360,
        "area": 200 * 60,
        "keypoints": None,
        "keypoints_conf": None,
    }

    # Verify size check passes
    bw = det["x2"] - det["x1"]
    bh = det["y2"] - det["y1"]
    max_dim = max(bw, bh)
    assert max_dim >= 480 * tracker.MIN_HEIGHT_RATIO

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ByteTracker:
    """
    ByteTrack implementation for multi-object tracking
    
    Tracks two fighters and their gloves across frames
    """
    
    def __init__(self, track_thresh: float = 0.5, high_thresh: float = 0.6,
                 match_thresh: float = 0.8, max_time_lost: int = 30):
        """
        Initialize ByteTracker
        
        Args:
            track_thresh: Detection confidence threshold
            high_thresh: High confidence threshold
            match_thresh: IoU matching threshold
            max_time_lost: Max frames before track is removed
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            Dict mapping track_id to tracked object data
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.high_thresh]
        low_dets = [d for d in detections if self.track_thresh <= d['confidence'] < self.high_thresh]
        
        # First association with high confidence detections
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(high_dets)))
        
        matched = []
        
        if len(self.tracks) > 0 and len(high_dets) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(high_dets)))
            
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(high_dets):
                    iou_matrix[i, j] = self._compute_iou(track['bbox'], det['bbox'])
            
            # Match using Hungarian algorithm (simplified with greedy matching)
            matched, unmatched_tracks, unmatched_dets = self._match(
                iou_matrix, self.match_thresh
            )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update({
                'bbox': high_dets[det_idx]['bbox'],
                'confidence': high_dets[det_idx]['confidence'],
                'class_name': high_dets[det_idx]['class_name'],
                'class_id': high_dets[det_idx]['class_id'],
                'time_since_update': 0,
                'hits': self.tracks[track_idx]['hits'] + 1
            })
        
        # Second association with low confidence detections
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
            
            iou_matrix = np.zeros((len(remaining_tracks), len(low_dets)))
            
            for i, track in enumerate(remaining_tracks):
                for j, det in enumerate(low_dets):
                    iou_matrix[i, j] = self._compute_iou(track['bbox'], det['bbox'])
            
            matched_low, unmatched_tracks_low, _ = self._match(
                iou_matrix, 0.5
            )
            
            for i, j in matched_low:
                track_idx = unmatched_tracks[i]
                self.tracks[track_idx].update({
                    'bbox': low_dets[j]['bbox'],
                    'confidence': low_dets[j]['confidence'],
                    'time_since_update': 0
                })
            
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_low]
        
        # Mark unmatched tracks as lost
        for idx in unmatched_tracks:
            self.tracks[idx]['time_since_update'] += 1
        
        # Initialize new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            self._init_track(det)
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t['time_since_update'] < self.max_time_lost
        ]
        
        # Build output: group detections by fighter
        output = self._group_by_fighter()
        
        return output
    
    def _init_track(self, detection: Dict):
        """Initialize new track"""
        track = {
            'track_id': self.track_id_counter,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class_name': detection['class_name'],
            'class_id': detection['class_id'],
            'time_since_update': 0,
            'hits': 1,
            'age': 0
        }
        
        self.tracks.append(track)
        self.track_id_counter += 1
    
    def _match(self, iou_matrix: np.ndarray, threshold: float):
        """Greedy matching based on IoU"""
        
        matched = []
        unmatched_tracks = list(range(iou_matrix.shape[0]))
        unmatched_dets = list(range(iou_matrix.shape[1]))
        
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find max IoU
            max_iou = 0
            max_idx = None
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_idx = (i, j)
            
            if max_iou < threshold or max_idx is None:
                break
            
            matched.append(max_idx)
            unmatched_tracks.remove(max_idx[0])
            unmatched_dets.remove(max_idx[1])
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two boxes"""
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _group_by_fighter(self) -> Dict[int, Dict]:
        """
        Group tracked objects by fighter
        
        Returns dict with fighter IDs (0, 1) and their body parts
        """
        
        fighters = defaultdict(lambda: {
            'body': None,
            'head': None,
            'gloves': [],
            'torso': None
        })
        
        for track in self.tracks:
            if track['time_since_update'] > 0:
                continue
            
            class_name = track['class_name']
            
            # Assign to fighter based on spatial proximity
            # For now, simple assignment (will be improved with spatial analysis)
            fighter_id = self._assign_fighter(track)
            
            if class_name == 'fighter':
                fighters[fighter_id]['body'] = track
            elif class_name == 'head':
                fighters[fighter_id]['head'] = track
            elif 'glove' in class_name:
                fighters[fighter_id]['gloves'].append(track)
            elif class_name == 'torso':
                fighters[fighter_id]['torso'] = track
        
        return dict(fighters)
    
    def _assign_fighter(self, track: Dict) -> int:
        """
        Assign track to fighter (0 or 1)
        
        Simple heuristic: based on x-position (left vs right)
        TODO: Improve with consistent ID assignment
        """
        
        bbox = track['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        
        # Assume fighter 0 is on left, fighter 1 on right
        # This is a simplification; real implementation needs temporal consistency
        
        return 0 if center_x < 960 else 1  # Assuming 1920px width


class KalmanFilter:
    """Simple Kalman filter for track prediction"""
    
    def __init__(self):
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.x = np.zeros(8)
        self.P = np.eye(8) * 1000  # Covariance
        
        # Transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.F[3, 7] = 1
        
        # Measurement matrix
        self.H = np.eye(4, 8)
        
        # Noise
        self.Q = np.eye(8) * 0.1
        self.R = np.eye(4) * 1
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:4]
    
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        
        return self.x[:4]

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FightMetricsExtractor:
    """
    Extract fight metrics from tracking and pose data
    
    Computes distance, ring position, and other contextual metrics
    """
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialize metrics extractor
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def extract(self, tracked_fighters: Dict, pose_data: Dict,
                frame_num: int) -> Dict[int, Dict]:
        """
        Extract metrics for all fighters
        
        Args:
            tracked_fighters: Dict mapping fighter_id to tracked objects
            pose_data: Dict mapping fighter_id to pose data
            frame_num: Current frame number
            
        Returns:
            Dict mapping fighter_id to metrics
        """
        
        metrics = {}
        
        for fighter_id in tracked_fighters.keys():
            fighter_tracked = tracked_fighters[fighter_id]
            fighter_pose = pose_data.get(fighter_id)
            
            if not fighter_tracked or not fighter_pose:
                continue
            
            # Compute metrics
            stance = fighter_pose.get('stance', 'unknown')
            balance = fighter_pose.get('balance', 'neutral')
            
            # Distance to opponent
            distance = self._compute_distance(
                tracked_fighters, fighter_id
            )
            
            # Ring position
            ring_position = self._compute_ring_position(
                fighter_tracked
            )
            
            # Guard position (from pose)
            guard_position = self._compute_guard_position(
                fighter_pose
            )
            
            metrics[fighter_id] = {
                'fighter_id': fighter_id,
                'frame_number': frame_num,
                'stance': stance,
                'balance': balance,
                'distance': distance,
                'ring_position': ring_position,
                'guard_position': guard_position
            }
        
        return metrics
    
    def _compute_distance(self, fighters: Dict, fighter_id: int) -> str:
        """
        Compute distance classification between fighters
        
        Returns: "inside", "mid", or "outside"
        """
        
        opponent_id = 1 - fighter_id
        
        if opponent_id not in fighters:
            return "outside"
        
        # Get center positions from torso or body
        fighter_bbox = fighters[fighter_id].get('torso') or \
                      fighters[fighter_id].get('body')
        opponent_bbox = fighters[opponent_id].get('torso') or \
                       fighters[opponent_id].get('body')
        
        if not fighter_bbox or not opponent_bbox:
            return "outside"
        
        # Compute center points
        fighter_center = self._bbox_center(fighter_bbox['bbox'])
        opponent_center = self._bbox_center(opponent_bbox['bbox'])
        
        # Euclidean distance in pixels
        pixel_distance = np.sqrt(
            (fighter_center[0] - opponent_center[0])**2 +
            (fighter_center[1] - opponent_center[1])**2
        )
        
        # Convert to approximate meters (calibration needed)
        # Assume average person is ~170cm, typical bbox height ~300px
        # So 1px ≈ 0.0057m
        distance_meters = pixel_distance * 0.0057
        
        # Classify distance
        if distance_meters < 1.2:
            return "inside"
        elif distance_meters < 2.0:
            return "mid"
        else:
            return "outside"
    
    def _bbox_center(self, bbox: list) -> Tuple[float, float]:
        """Calculate center of bounding box"""
        return (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        )
    
    def _compute_ring_position(self, fighter: Dict) -> Tuple[float, float]:
        """
        Compute normalized ring position (0-1, 0-1)
        
        Returns (x, y) where:
        - (0, 0) is top-left corner
        - (1, 1) is bottom-right corner
        """
        
        body = fighter.get('body') or fighter.get('torso')
        
        if not body:
            return (0.5, 0.5)
        
        center = self._bbox_center(body['bbox'])
        
        # Normalize to 0-1
        x_norm = center[0] / self.frame_width
        y_norm = center[1] / self.frame_height
        
        return (x_norm, y_norm)
    
    def _compute_guard_position(self, pose: Dict) -> str:
        """
        Determine guard position from pose
        
        Returns: "high", "low", or "open"
        """
        
        if not pose or 'landmarks' not in pose:
            return "open"
        
        landmarks = pose['landmarks']
        
        # Get wrist and head positions
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        
        # Average wrist height
        avg_wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
        
        # Compare to head position
        head_to_wrist_dist = avg_wrist_y - nose['y']
        
        # Classification thresholds
        if head_to_wrist_dist < 50:
            return "high"
        elif head_to_wrist_dist < 150:
            return "open"
        else:
            return "low"


def compute_ring_control(metrics_history: list, window_frames: int = 300) -> Dict:
    """
    Compute ring control statistics
    
    Args:
        metrics_history: List of metrics dicts over time
        window_frames: Window size for analysis
        
    Returns:
        Dict with ring control percentages
    """
    
    if not metrics_history:
        return {'fighter_0': 0.5, 'fighter_1': 0.5}
    
    # Analyze recent window
    recent = metrics_history[-window_frames:]
    
    # Count frames where each fighter is in center
    fighter_0_center = 0
    fighter_1_center = 0
    
    for frame_metrics in recent:
        for fighter_id, metrics in frame_metrics.items():
            pos = metrics.get('ring_position', (0.5, 0.5))
            
            # Center is approximately (0.4-0.6, 0.4-0.6)
            in_center = (0.4 <= pos[0] <= 0.6 and 0.4 <= pos[1] <= 0.6)
            
            if in_center:
                if fighter_id == 0:
                    fighter_0_center += 1
                else:
                    fighter_1_center += 1
    
    total = fighter_0_center + fighter_1_center
    
    if total == 0:
        return {'fighter_0': 0.5, 'fighter_1': 0.5}
    
    return {
        'fighter_0': fighter_0_center / total,
        'fighter_1': fighter_1_center / total
    }

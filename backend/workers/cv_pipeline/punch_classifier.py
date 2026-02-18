import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PunchClassifier:
    """
    Punch detection and classification
    
    Detects punches from glove trajectories and classifies type/outcome
    """
    
    def __init__(self, velocity_threshold: float = 2.0, window_size: int = 10):
        """
        Initialize punch classifier
        
        Args:
            velocity_threshold: Minimum velocity (m/s) to consider as punch
            window_size: Frame window for trajectory analysis
        """
        self.velocity_threshold = velocity_threshold
        self.window_size = window_size
        
        # Track glove trajectories
        self.trajectories = {
            0: {'left': deque(maxlen=window_size), 'right': deque(maxlen=window_size)},
            1: {'left': deque(maxlen=window_size), 'right': deque(maxlen=window_size)}
        }
        
        # Punch state tracking
        self.active_punches = []
    
    def detect_and_classify(self, tracked_fighters: Dict, pose_data: Dict,
                           frame_num: int, fps: int = 30) -> List[Dict]:
        """
        Detect and classify punches in current frame
        
        Args:
            tracked_fighters: Dict mapping fighter_id to tracked objects
            pose_data: Dict mapping fighter_id to pose estimation results
            frame_num: Current frame number
            fps: Frames per second
            
        Returns:
            List of detected punch events
        """
        
        punches = []
        
        for fighter_id, fighter_data in tracked_fighters.items():
            if 'gloves' not in fighter_data or len(fighter_data['gloves']) < 2:
                continue
            
            gloves = fighter_data['gloves']
            
            # Update trajectories
            for glove in gloves:
                hand = self._determine_hand(glove, pose_data.get(fighter_id))
                if hand:
                    position = (glove['bbox'][0] + glove['bbox'][2]) / 2, \
                              (glove['bbox'][1'] + glove['bbox'][3]) / 2
                    
                    self.trajectories[fighter_id][hand].append({
                        'frame': frame_num,
                        'position': position,
                        'bbox': glove['bbox']
                    })
            
            # Analyze trajectories for punches
            for hand in ['left', 'right']:
                trajectory = self.trajectories[fighter_id][hand]
                
                if len(trajectory) < 3:
                    continue
                
                # Detect punch (sudden velocity increase)
                punch_detected = self._detect_punch_initiation(trajectory, fps)
                
                if punch_detected:
                    # Classify punch type
                    punch_type = self._classify_punch_type(trajectory, pose_data.get(fighter_id))
                    
                    # Determine outcome
                    outcome, target = self._determine_outcome(
                        trajectory, tracked_fighters, fighter_id
                    )
                    
                    # Compute metrics
                    speed = self._compute_speed(trajectory, fps)
                    impact_score = self._compute_impact_score(outcome, speed, punch_type)
                    
                    punch = {
                        'timestamp': frame_num / fps,
                        'frame_number': frame_num,
                        'fighter_id': fighter_id,
                        'punch_type': punch_type,
                        'hand': hand,
                        'outcome': outcome,
                        'target': target,
                        'speed': speed,
                        'impact_score': impact_score
                    }
                    
                    punches.append(punch)
        
        return punches
    
    def _determine_hand(self, glove: Dict, pose: Optional[Dict]) -> Optional[str]:
        """Determine if glove is left or right"""
        
        if not pose:
            # Fallback: use class name if available
            if 'glove_left' in glove.get('class_name', ''):
                return 'left'
            elif 'glove_right' in glove.get('class_name', ''):
                return 'right'
            return None
        
        # Use pose landmarks to determine hand
        landmarks = pose['landmarks']
        
        glove_center = (
            (glove['bbox'][0] + glove['bbox'][2]) / 2,
            (glove['bbox'][1] + glove['bbox'][3]) / 2
        )
        
        # Find closest wrist
        left_wrist = landmarks[15]  # MediaPipe index
        right_wrist = landmarks[16]
        
        dist_left = np.sqrt(
            (glove_center[0] - left_wrist['x'])**2 +
            (glove_center[1] - left_wrist['y'])**2
        )
        
        dist_right = np.sqrt(
            (glove_center[0] - right_wrist['x'])**2 +
            (glove_center[1] - right_wrist['y'])**2
        )
        
        return 'left' if dist_left < dist_right else 'right'
    
    def _detect_punch_initiation(self, trajectory: deque, fps: int) -> bool:
        """
        Detect punch initiation from velocity spike
        
        Returns True if punch detected in recent frames
        """
        
        if len(trajectory) < 3:
            return False
        
        # Compute velocity for last few frames
        velocities = []
        
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]['position']
            p2 = trajectory[i + 1]['position']
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Convert pixels/frame to m/s (approximate)
            # Assume 1 pixel = 0.01m (will be calibrated)
            distance = np.sqrt(dx**2 + dy**2) * 0.01
            time = 1.0 / fps
            
            velocity = distance / time
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return False
        
        # Check for velocity spike
        recent_velocity = velocities[-1]
        avg_velocity = np.mean(velocities[:-1])
        
        # Punch detected if recent velocity is significantly higher
        return recent_velocity > self.velocity_threshold and \
               recent_velocity > avg_velocity * 2
    
    def _classify_punch_type(self, trajectory: deque, pose: Optional[Dict]) -> str:
        """
        Classify punch type based on trajectory pattern
        
        Types: jab, cross, hook, uppercut
        """
        
        if len(trajectory) < 3:
            return "jab"  # Default
        
        # Analyze trajectory pattern
        positions = [t['position'] for t in trajectory]
        
        # Compute trajectory characteristics
        dx_total = positions[-1][0] - positions[0][0]
        dy_total = positions[-1][1] - positions[0][1]
        
        # Check trajectory curvature
        curvature = self._compute_curvature(positions)
        
        # Vertical movement (positive = downward)
        vertical_ratio = abs(dy_total) / (abs(dx_total) + 1e-6)
        
        # Classification heuristics
        if dy_total < -50 and vertical_ratio > 1.5:
            # Significant upward movement
            return "uppercut"
        elif curvature > 0.3:
            # Curved trajectory
            return "hook"
        elif abs(dx_total) < 100:
            # Straight, short punch
            return "jab"
        else:
            # Straight, long punch
            return "cross"
    
    def _compute_curvature(self, positions: List[Tuple[float, float]]) -> float:
        """Compute trajectory curvature (0 = straight, 1 = very curved)"""
        
        if len(positions) < 3:
            return 0.0
        
        # Compare direct distance vs. path length
        direct_dist = np.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )
        
        path_length = 0
        for i in range(len(positions) - 1):
            path_length += np.sqrt(
                (positions[i+1][0] - positions[i][0])**2 +
                (positions[i+1][1] - positions[i][1])**2
            )
        
        if direct_dist < 1:
            return 0.0
        
        curvature = (path_length - direct_dist) / direct_dist
        
        return min(curvature, 1.0)
    
    def _determine_outcome(self, trajectory: deque, all_fighters: Dict,
                          attacker_id: int) -> Tuple[str, str]:
        """
        Determine punch outcome and target
        
        Returns:
            (outcome, target) where outcome is landed/missed/blocked
            and target is head/body
        """
        
        # Get opponent ID
        opponent_id = 1 - attacker_id
        
        if opponent_id not in all_fighters:
            return "missed", "head"
        
        opponent = all_fighters[opponent_id]
        
        # Get punch endpoint
        punch_end = trajectory[-1]['position']
        
        # Check proximity to opponent's head/body
        head_hit = False
        body_hit = False
        blocked = False
        
        if opponent.get('head'):
            head_bbox = opponent['head']['bbox']
            if self._point_in_bbox(punch_end, head_bbox, margin=20):
                head_hit = True
        
        if opponent.get('torso') or opponent.get('body'):
            body_bbox = opponent.get('torso', opponent.get('body'))['bbox']
            if self._point_in_bbox(punch_end, body_bbox, margin=20):
                body_hit = True
        
        # Check if blocked by opponent's gloves
        if opponent.get('gloves'):
            for glove in opponent['gloves']:
                if self._point_in_bbox(punch_end, glove['bbox'], margin=15):
                    blocked = True
                    break
        
        # Determine outcome
        if blocked:
            outcome = "blocked"
            target = "head" if head_hit else "body"
        elif head_hit:
            outcome = "landed"
            target = "head"
        elif body_hit:
            outcome = "landed"
            target = "body"
        else:
            outcome = "missed"
            target = "head"  # Default assumption
        
        return outcome, target
    
    def _point_in_bbox(self, point: Tuple[float, float], bbox: List[int],
                      margin: int = 0) -> bool:
        """Check if point is within bounding box (with margin)"""
        
        x, y = point
        x1, y1, x2, y2 = bbox
        
        return (x1 - margin <= x <= x2 + margin and
                y1 - margin <= y <= y2 + margin)
    
    def _compute_speed(self, trajectory: deque, fps: int) -> float:
        """Compute punch speed in m/s"""
        
        if len(trajectory) < 2:
            return 0.0
        
        # Use last few frames for peak speed
        p1 = trajectory[-2]['position']
        p2 = trajectory[-1]['position']
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Convert to meters (approximate calibration)
        distance = np.sqrt(dx**2 + dy**2) * 0.01
        time = 1.0 / fps
        
        return distance / time
    
    def _compute_impact_score(self, outcome: str, speed: float,
                              punch_type: str) -> float:
        """
        Compute impact score (0-1)
        
        Based on outcome, speed, and punch type
        """
        
        if outcome == "missed":
            return 0.0
        
        # Base score from outcome
        outcome_scores = {
            "landed": 1.0,
            "blocked": 0.3,
            "missed": 0.0
        }
        
        base_score = outcome_scores.get(outcome, 0.0)
        
        # Speed factor (normalize to typical range 5-15 m/s)
        speed_factor = min(speed / 15.0, 1.0)
        
        # Punch type factor
        power_multipliers = {
            "jab": 0.7,
            "cross": 1.0,
            "hook": 1.0,
            "uppercut": 1.1
        }
        
        power_mult = power_multipliers.get(punch_type, 1.0)
        
        impact = base_score * speed_factor * power_mult
        
        return min(impact, 1.0)

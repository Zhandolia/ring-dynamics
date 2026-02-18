import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Pose estimation using MediaPipe
    
    Extracts joint kinematics for fight analysis
    """
    
    def __init__(self):
        """Initialize MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
    
    def estimate(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict]:
        """
        Estimate pose for a person in bounding box
        
        Args:
            frame: Input frame (H, W, 3) BGR
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict with landmarks and computed metrics
        """
        
        # Crop to bbox
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = frame[y1:y2, x1:x2]
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Run pose estimation
        results = self.pose.process(crop_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            # Convert to original frame coordinates
            x = lm.x * (x2 - x1) + x1
            y = lm.y * (y2 - y1) + y1
            z = lm.z
            visibility = lm.visibility
            
            landmarks.append({
                'x': x,
                'y': y,
                'z': z,
                'visibility': visibility
            })
        
        # Compute metrics
        stance = self._compute_stance(landmarks)
        balance = self._compute_balance(landmarks)
        joint_angles = self._compute_joint_angles(landmarks)
        
        return {
            'landmarks': landmarks,
            'stance': stance,
            'balance': balance,
            'joint_angles': joint_angles
        }
    
    def _compute_stance(self, landmarks: List[Dict]) -> str:
        """
        Determine fighting stance
        
        Orthodox: Left foot forward, right foot back
        Southpaw: Right foot forward, left foot back
        Squared: Feet parallel
        """
        
        left_ankle = landmarks[self.LEFT_ANKLE]
        right_ankle = landmarks[self.RIGHT_ANKLE]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        
        # Check which foot is forward (smaller y-value)
        left_forward = left_ankle['y'] < right_ankle['y']
        right_forward = right_ankle['y'] < left_ankle['y']
        
        y_diff = abs(left_ankle['y'] - right_ankle['y'])
        
        # If feet are approximately level (within threshold)
        if y_diff < 20:
            return "squared"
        elif left_forward:
            return "orthodox"
        else:
            return "southpaw"
    
    def _compute_balance(self, landmarks: List[Dict]) -> str:
        """
        Determine weight distribution
        
        Based on center of mass relative to feet
        """
        
        # Approximate center of mass from hips and shoulders
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_ankle = landmarks[self.LEFT_ANKLE]
        right_ankle = landmarks[self.RIGHT_ANKLE]
        
        # Center of mass (simplified)
        com_x = (left_shoulder['x'] + right_shoulder['x'] + 
                 left_hip['x'] + right_hip['x']) / 4
        com_y = (left_shoulder['y'] + right_shoulder['y'] + 
                 left_hip['y'] + right_hip['y']) / 4
        
        # Midpoint between feet
        feet_mid_x = (left_ankle['x'] + right_ankle['x']) / 2
        feet_mid_y = (left_ankle['y'] + right_ankle['y']) / 2
        
        # Check if CoM is forward or backward
        # Assuming camera view from side
        x_offset = com_x - feet_mid_x
        
        if abs(x_offset) < 10:
            return "neutral"
        elif x_offset > 10:
            return "front_foot"
        else:
            return "back_foot"
    
    def _compute_joint_angles(self, landmarks: List[Dict]) -> Dict[str, float]:
        """
        Compute joint angles
        
        Returns angles for shoulders, elbows, hips, knees
        """
        
        angles = {}
        
        # Left elbow angle
        angles['left_elbow'] = self._angle_between_points(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_WRIST]
        )
        
        # Right elbow angle
        angles['right_elbow'] = self._angle_between_points(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_WRIST]
        )
        
        # Left knee angle
        angles['left_knee'] = self._angle_between_points(
            landmarks[self.LEFT_HIP],
            landmarks[self.LEFT_KNEE],
            landmarks[self.LEFT_ANKLE]
        )
        
        # Right knee angle
        angles['right_knee'] = self._angle_between_points(
            landmarks[self.RIGHT_HIP],
            landmarks[self.RIGHT_KNEE],
            landmarks[self.RIGHT_ANKLE]
        )
        
        return angles
    
    def _angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """
        Calculate angle formed by three points
        
        Args:
            p1: First point (start)
            p2: Middle point (vertex)
            p3: End point
            
        Returns:
            Angle in degrees
        """
        
        # Vectors
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        
        # Dot product
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Angle in degrees
        angle = np.arccos(dot_product) * 180 / np.pi
        
        return float(angle)
    
    def compute_center_of_mass(self, landmarks: List[Dict]) -> Tuple[float, float]:
        """
        Compute approximate center of mass
        
        Returns:
            (x, y) coordinates
        """
        
        # Use key body points
        key_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP
        ]
        
        x_sum = sum(landmarks[i]['x'] for i in key_points)
        y_sum = sum(landmarks[i]['y'] for i in key_points)
        
        com_x = x_sum / len(key_points)
        com_y = y_sum / len(key_points)
        
        return com_x, com_y

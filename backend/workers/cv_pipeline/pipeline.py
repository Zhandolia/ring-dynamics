"""
Integrated CV Pipeline for Ring Dynamics

This module orchestrates the entire computer vision pipeline:
- YOLO detection
- ByteTrack tracking
- Pose estimation
- Punch classification
- Fight metrics extraction
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from .detection import YOLODetector
from .tracking import ByteTracker
from .pose import PoseEstimator
from .punch_classifier import PunchClassifier
from .metrics import FightMetricsExtractor

logger = logging.getLogger(__name__)


class CVPipeline:
    """Unified computer vision pipeline for fight analysis"""
    
    def __init__(self, device: str = 'cpu', model_path: Optional[str] = None):
        """
        Initialize CV pipeline
        
        Args:
            device: Device to run models on ('cpu' or 'cuda:0')
            model_path: Path to YOLO model weights
        """
        self.device = device
        
        logger.info(f"Initializing CV Pipeline on device: {device}")
        
        # Initialize components
        self.detector = YOLODetector(device=device, model_path=model_path)
        self.tracker = ByteTracker()
        self.pose_estimator = PoseEstimator()
        self.punch_classifier = PunchClassifier()
        self.metrics_extractor = FightMetricsExtractor()
        
        # State
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Process a single video frame through the full pipeline
        
        Args:
            frame: Video frame (BGR format)
            frame_number: Frame index
            
        Returns:
            Dictionary containing all detected events and metrics
        """
        results = {
            'frame_number': frame_number,
            'detections': [],
            'tracks': [],
            'poses': {},
            'punches': [],
            'metrics': {}
        }
        
        try:
            # 1. Object Detection
            detections = self.detector.detect(frame)
            results['detections'] = detections
            
            # 2. Multi-object Tracking
            tracks = self.tracker.update(detections, frame)
            results['tracks'] = tracks
            
            # 3. Pose Estimation (for each fighter)
            fighter_tracks = [t for t in tracks if t.get('class_name') == 'fighter']
            for track in fighter_tracks[:2]:  # Max 2 fighters
                fighter_id = track['track_id']
                bbox = track['bbox']
                
                # Extract fighter region
                x1, y1, x2, y2 = map(int, bbox)
                fighter_roi = frame[y1:y2, x1:x2]
                
                if fighter_roi.size > 0:
                    pose_data = self.pose_estimator.estimate_pose(fighter_roi)
                    results['poses'][fighter_id] = pose_data
            
            # 4. Punch Detection & Classification
            glove_tracks = [t for t in tracks if 'glove' in t.get('class_name', '')]
            punches = self.punch_classifier.detect_and_classify(glove_tracks, fighter_tracks)
            results['punches'] = punches
            
            # 5. Fight Metrics Extraction
            metrics = self.metrics_extractor.extract_metrics(
                fighter_tracks=fighter_tracks,
                poses=results['poses'],
                frame_shape=frame.shape
            )
            results['metrics'] = metrics
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            
        return results
    
    def process_video(self, video_path: str, fps: int = 30, 
                     max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process entire video file
        
        Args:
            video_path: Path to video file
            fps: Frames per second to process (skip frames if needed)
            max_frames: Maximum number of frames to process (for testing)
            
        Returns:
            List of frame results
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / fps))
        
        all_results = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Process frame
            results = self.process_frame(frame, frame_idx)
            all_results.append(results)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} frames...")
            
            # Check max frames limit
            if max_frames and processed_count >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Video processing complete. Processed {processed_count} frames.")
        
        return all_results
    
    def reset(self):
        """Reset pipeline state for new video"""
        self.tracker = ByteTracker()
        self.punch_classifier = PunchClassifier()
        self.metrics_extractor = FightMetricsExtractor()
        self.frame_count = 0
        logger.info("Pipeline reset")

from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLO-based multi-object detector for boxing
    
    Detects: fighters (bodies), heads, gloves, torso
    """
    
    def __init__(self, model_path: str = "yolov8x.pt", device: str = "cuda:0"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO weights
            device: Device to run inference on
        """
        self.device = device
        logger.info(f"Loading YOLO model from {model_path} on {device}")
        
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Class mapping (will be updated after fine-tuning)
        self.class_names = {
            0: "fighter",
            1: "head",
            2: "glove_left",
            3: "glove_right",
            4: "torso"
        }
        
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            
        Returns:
            List of detections with bbox, class, confidence
        """
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract bbox coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch detection for efficiency
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection lists (one per frame)
        """
        
        results_list = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        
        all_detections = []
        for results in results_list:
            frame_detections = []
            boxes = results.boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                }
                
                frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections


def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
    """
    Apply NMS to detections
    
    Args:
        detections: List of detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Filtered detections
    """
    
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while len(detections) > 0:
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        detections = [
            det for det in detections
            if compute_iou(current['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two bounding boxes"""
    
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

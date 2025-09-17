import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

class Validator:
    def __init__(self, model, iou_threshold=0.5):
        self.model = model
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def evaluate_dataset(self, images_dir: str, labels_dir: str) -> Dict:
        """Evaluate model on validation dataset"""
        all_metrics = []
        
        for image_path in Path(images_dir).glob("*.*"):
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Get predictions
            pred_boxes = []
            detections = self.model.detect(image)
            for det in detections:
                x_min, y_min, x_max, y_max, conf = det
                pred_boxes.append((x_min, y_min, x_max, y_max, conf))
            
            # Load ground truth
            label_path = Path(labels_dir) / f"{image_path.stem}.txt"
            true_boxes = self._load_yolo_labels(label_path, image.shape)
            
            # Calculate metrics
            metrics = self._calculate_image_metrics(true_boxes, pred_boxes)
            all_metrics.append(metrics)
        
        # Aggregate results
        return self._aggregate_metrics(all_metrics)
    
    def _load_yolo_labels(self, label_path: Path, image_shape: tuple) -> List:
        """Load YOLO format labels"""
        true_boxes = []
        
        if label_path.exists():
            h, w = image_shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # Convert to absolute coordinates
                        x_center_abs = x_center * w
                        y_center_abs = y_center * h
                        width_abs = width * w
                        height_abs = height * h
                        
                        x_min = x_center_abs - width_abs / 2
                        y_min = y_center_abs - height_abs / 2
                        x_max = x_center_abs + width_abs / 2
                        y_max = y_center_abs + height_abs / 2
                        
                        true_boxes.append((x_min, y_min, x_max, y_max))
        
        return true_boxes

# Usage example
if __name__ == "__main__":
    from app.detection.detector import LogoDetector
    
    model = LogoDetector()
    validator = Validator(model)
    
    results = validator.evaluate_dataset(
        "validation_dataset/images",
        "validation_dataset/labels"
    )
    
    print("Validation Results:")
    print(json.dumps(results, indent=2))
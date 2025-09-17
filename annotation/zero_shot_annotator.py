import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict

class ZeroShotAnnotator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_grounding_dino()
        self.clip_model = self._load_clip_model()
        
        # Промпты для детекции
        self.prompts = {
            "t_bank": [
                "yellow T logo on shield", "T bank logo", 
                "yellow shield with T letter", "corporate T logo",
                "geometric T symbol", "bank logo with T"
            ],
            "tinkoff": [
                "Tinkoff logo", "orange T logo", "Tinkoff bank logo",
                "orange and white logo", "T with circle logo"
            ]
        }
    
    def _load_grounding_dino(self):
        """Загрузка Grounding DINO модели"""
        model = load_model(
            "GroundingDINO_SwinB.cfg.py",
            "weights/groundingdino_swinb_cogcoor.pth"
        )
        model.to(self.device)
        return model
    
    def _load_clip_model(self):
        """Загрузка CLIP для верификации"""
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return {"model": model, "processor": processor}
    
    def annotate_image(self, image_path, output_dir):
        """Полуавтоматическая разметка изображения"""
        image_source, image = load_image(image_path)
        
        # Детекция всех возможных логотипов
        all_boxes = self._detect_all_logos(image)
        
        # Фильтрация: только Т-банк, не Тинькофф
        filtered_boxes = self._filter_t_bank_only(all_boxes, image_source)
        
        # Сохранение результатов
        self._save_annotations(image_path, output_dir, filtered_boxes, image_source)
        
        return filtered_boxes
    
    def _detect_all_logos(self, image):
        """Детекция всех возможных логотипов"""
        all_detections = []
        
        for logo_type, prompts in self.prompts.items():
            for prompt in prompts:
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.25,
                    text_threshold=0.2
                )
                for box, logit in zip(boxes, logits):
                    all_detections.append({
                        "box": box,
                        "confidence": logit,
                        "type": logo_type,
                        "prompt": prompt
                    })
        
        return all_detections
    
    def _filter_t_bank_only(self, detections, image_source):
        """Фильтрация только Т-банк логотипов"""
        t_bank_boxes = []
        tinkoff_boxes = []
        
        # Разделяем детекции по типам
        for detection in detections:
            if detection["type"] == "t_bank" and detection["confidence"] > 0.3:
                t_bank_boxes.append(detection)
            elif detection["type"] == "tinkoff" and detection["confidence"] > 0.3:
                tinkoff_boxes.append(detection)
        
        # Удаляем Т-банк детекции, которые пересекаются с Тинькофф
        final_boxes = []
        for t_detection in t_bank_boxes:
            is_tinkoff = False
            for tinkoff_detection in tinkoff_boxes:
                if self._calculate_iou(t_detection["box"], tinkoff_detection["box"]) > 0.1:
                    is_tinkoff = True
                    break
            
            if not is_tinkoff:
                final_boxes.append(t_detection)
        
        return final_boxes
    
    def _calculate_iou(self, box1, box2):
        """Вычисление IoU"""
        # ... implementation from previous code ...
        pass
    
    def _save_annotations(self, image_path, output_dir, detections, image_source):
        """Сохранение аннотаций в YOLO формате"""
        # ... implementation from previous code ...
        pass

def process_dataset(input_dir, output_dir):
    """Обработка всего датасета"""
    annotator = ZeroShotAnnotator()
    
    for img_path in Path(input_dir).glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            annotator.annotate_image(str(img_path), output_dir)
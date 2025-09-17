import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from app.detection.detector import LogoDetector

class Validator:
    def __init__(self, validation_dir: str = "data/validation"):
        """Инициализация валидатора"""
        self.validation_dir = validation_dir  # Директория с валидационными данными
        self.detector = LogoDetector()        # Детектор для валидации
        
    def run_validation(self):
        """Запуск валидации на валидационном dataset"""
        annotations_file = os.path.join(self.validation_dir, "annotations.json")
        
        if not os.path.exists(annotations_file):
            print("Validation annotations not found!")
            return
        
        # Загрузка аннотаций
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        all_true = []  # Все ground truth метки
        all_pred = []  # Все предсказанные метки
        
        for img_info in annotations:
            img_path = os.path.join(self.validation_dir, "images", img_info['file_name'])
            if not os.path.exists(img_path):
                continue  # Пропускаем если изображение не найдено
                
            # Загрузка изображения
            image = cv2.imread(img_path)
            if image is None:
                continue  # Пропускаем если не удалось загрузить
            
            # Получение ground truth bounding boxes
            gt_boxes = []
            for ann in img_info.get('annotations', []):
                gt_boxes.append([ann['x_min'], ann['y_min'], ann['x_max'], ann['y_max']])
            
            # Детекция логотипов
            pred_boxes = self.detector.detect(image)
            
            # Расчет метрик для изображения
            img_true, img_pred = self._calculate_image_metrics(gt_boxes, pred_boxes, image.shape)
            all_true.extend(img_true)
            all_pred.extend(img_pred)
        
        # Расчет общих метрик
        precision = precision_score(all_true, all_pred, zero_division=0)  # Precision
        recall = recall_score(all_true, all_pred, zero_division=0)        # Recall
        f1 = f1_score(all_true, all_pred, zero_division=0)                # F1 Score
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_image_metrics(self, gt_boxes, pred_boxes, image_shape):
        """Расчет метрик для одного изображения"""
        # Создание грида для evaluation
        grid_size = 32  # Размер ячейки грида
        h, w = image_shape[:2]
        grid_h, grid_w = h // grid_size, w // grid_size
        
        true_grid = np.zeros((grid_h, grid_w), dtype=int)  # Грид для ground truth
        pred_grid = np.zeros((grid_h, grid_w), dtype=int)  # Грид для predictions
        
        # Разметка ground truth областей
        for x1, y1, x2, y2 in gt_boxes:
            gx1, gy1 = int(x1 / grid_size), int(y1 / grid_size)
            gx2, gy2 = min(int(x2 / grid_size), grid_w-1), min(int(y2 / grid_size), grid_h-1)
            true_grid[gy1:gy2+1, gx1:gx2+1] = 1  # Помечаем область как положительную
        
        # Разметка predicted областей
        for x1, y1, x2, y2 in pred_boxes:
            gx1, gy1 = int(x1 / grid_size), int(y1 / grid_size)
            gx2, gy2 = min(int(x2 / grid_size), grid_w-1), min(int(y2 / grid_size), grid_h-1)
            pred_grid[gy1:gy2+1, gx1:gx2+1] = 1  # Помечаем область как предсказанную
        
        return true_grid.flatten(), pred_grid.flatten()  # Возвращаем flattened arrays для метрик

if __name__ == "__main__":
    validator = Validator()
    results = validator.run_validation()
    
    # Сохранение результатов
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
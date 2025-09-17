import os
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from ultralytics import YOLO

class DatasetPreparer:
    def __init__(self, data_dir: str = "data/data_sirius", output_dir: str = "data/processed"):
        """Инициализация подготовщика dataset"""
        self.data_dir = data_dir      # Директория с исходными данными
        self.output_dir = output_dir  # Директория для обработанных данных
        # Большая модель для лучшей zero-shot детекции
        self.detector = YOLO('yolov8x.pt')
        
    def prepare_training_data(self):
        """Подготовка тренировочных данных с использованием zero-shot подхода"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Создание директорий для изображений и аннотаций
        images_dir = os.path.join(self.output_dir, "images")
        labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Получение всех изображений
        image_files = [f for f in os.listdir(self.data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        # Обработка всех изображений
        for img_file in tqdm(image_files):
            img_path = os.path.join(self.data_dir, img_file)
            self._process_image(img_path, images_dir, labels_dir)
    
    def _process_image(self, img_path: str, images_dir: str, labels_dir: str):
        """Обработка одного изображения и генерация аннотаций"""
        image = cv2.imread(img_path)
        if image is None:
            return  # Пропускаем если изображение не загрузилось
        
        # Запуск zero-shot детекции
        # Классы: TV (62), laptop (63), cell phone (67) - часто содержат логотипы
        results = self.detector(image, classes=[62, 63, 67])
        
        # Фильтрация потенциальных обнаружений логотипов
        logo_boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()  # Уверенность детекции
                    
                    # Дополнительная фильтрация по соотношению сторон и размеру
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = w / h  # Соотношение сторон
                    
                    if (conf > 0.3 and 0.5 < aspect_ratio < 2.0 and 
                        min(w, h) > 20 and self._is_potential_logo(image, (x1, y1, x2, y2))):
                        logo_boxes.append((x1, y1, x2, y2, conf))
        
        # Сохранение изображения и аннотаций если обнаружены логотипы
        if logo_boxes:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_path = os.path.join(images_dir, f"{base_name}.jpg")
            output_label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            # Сохранение изображения
            cv2.imwrite(output_img_path, image)
            
            # Сохранение аннотаций в YOLO формате
            with open(output_label_path, 'w') as f:
                for x1, y1, x2, y2, conf in logo_boxes:
                    # Конвертация в YOLO формат (нормализованные center x, center y, width, height)
                    img_h, img_w = image.shape[:2]
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Запись в формате: class_id center_x center_y width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _is_potential_logo(self, image: np.ndarray, bbox: Tuple) -> bool:
        """Эвристика для идентификации потенциальных регионов с логотипами"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False  # Пустой регион
        
        # Проверка на высокий контраст и специфические цветовые паттерны
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)  # Контрастность региона
        
        # Проверка на желтый цвет (характерен для логотипа Т-банка)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
        
        return contrast > 30 or yellow_ratio > 0.1  # Высокий контраст или желтый цвет
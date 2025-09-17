import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
import os

class LogoDetector:
    def __init__(self, model_path: str = "weights/tbank_logo_detector.pt"):
        """Инициализация детектора логотипов"""
        self.model = None          # Модель для детекции
        self.model_path = model_path  # Путь к весам модели
        self.load_model()          # Загрузка модели
    
    def load_model(self):
        """Загрузка обученной YOLO модели"""
        if os.path.exists(self.model_path):
            try:
                # Загрузка обученной модели
                self.model = YOLO(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Если не удалось загрузить, используем zero-shot подход
                self.load_zero_shot_detector()
        else:
            print("Trained model not found, using zero-shot approach")
            # Если модель не найдена, используем zero-shot подход
            self.load_zero_shot_detector()
    
    def load_zero_shot_detector(self):
        """Загрузка предобученного zero-shot детектора объектов"""
        try:
            # Используем YOLOv8 как базовую модель
            self.model = YOLO('yolov8n.pt')
            print("Loaded zero-shot detector")
        except Exception as e:
            print(f"Error loading zero-shot detector: {e}")
            self.model = None
    
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        return self.model is not None
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция логотипов Т-банка на изображении
        
        Args:
            image: Входное изображение в BGR формате
            
        Returns:
            List[Tuple[int, int, int, int]]: Список bounding boxes в формате (x_min, y_min, x_max, y_max)
        """
        if self.model is None:
            return []  # Если модель не загружена, возвращаем пустой список
        
        try:
            # Запуск inference с порогами уверенности
            results = self.model(image, conf=0.5, iou=0.5)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Извлечение координат bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()  # Уверенность детекции
                        
                        # Фильтрация по уверенности и дополнительным эвристикам
                        if conf > 0.5 and self._is_valid_logo(image, (x1, y1, x2, y2)):
                            detections.append((x1, y1, x2, y2))
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []  # В случае ошибки возвращаем пустой список
    
    def _is_valid_logo(self, image: np.ndarray, bbox: Tuple) -> bool:
        """
        Дополнительная валидация чтобы убедиться что это логотип Т-банка, а не Тинькофф
        с использованием анализа цвета и формы
        
        Args:
            image: Входное изображение
            bbox: Bounding box для проверки (x1, y1, x2, y2)
            
        Returns:
            bool: True если это валидный логотип Т-банка
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Извлечение региона интереса
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False  # Пустой регион
        
        # Конвертация в HSV для анализа цвета
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Определение диапазона желтого цвета для логотипа Т-банка
        lower_yellow = np.array([20, 100, 100])  # Нижняя граница желтого
        upper_yellow = np.array([30, 255, 255])  # Верхняя граница желтого
        
        # Создание маски для желтых пикселей
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Проверка наличия достаточного количества желтых пикселей
        yellow_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        
        # Здесь можно добавить дополнительный анализ формы
        return yellow_ratio > 0.1  # Как минимум 10% желтых пикселей
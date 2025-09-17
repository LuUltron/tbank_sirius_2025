import os
from ultralytics import YOLO
import yaml

def train_model():
    """Обучение YOLO модели на подготовленном dataset"""
    
    # Создание YAML файла конфигурации dataset
    dataset_config = {
        'path': 'data/processed',  # Путь к данным
        'train': 'images',         # Директория с тренировочными изображениями
        'val': 'images',           # Для простоты используем те же данные для валидации
        'names': {0: 'tbank_logo'} # Имена классов
    }
    
    # Сохранение конфигурации
    with open('data/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Загрузка предобученной YOLO модели
    model = YOLO('yolov8n.pt')  # Используем nano версию для скорости
    
    # Обучение модели
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,                # Количество эпох
        imgsz=640,                 # Размер изображения
        batch=16,                  # Размер батча
        patience=20,               # Допуск для early stopping
        project='tbank_detection',
        name='logo_detector',
        optimizer='AdamW',
        lr0=0.001,                # Начальная скорость обучения
        augment=True
    )
    
    # Экспорт лучшей модели
    best_model = YOLO('tbank_detection/logo_detector/weights/best.pt')
    best_model.export(format='onnx')  # Экспорт в ONNX формат
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model()
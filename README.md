# T-Bank Logo Detector API

REST API сервис для детекции логотипов "Т-банк" на изображениях.

## 🚀 Возможности

- Обнаружение логотипов Т-банка (стилизованная буква "Т" в желтом щите)
- Zero-shot подход для подготовки данных
- Поддержка форматов: JPEG, PNG, BMP, WEBP
- GPU acceleration
- Docker контейнеризация

## 📦 Установка

### Требования

- Docker
- Docker Compose
- NNVIDIA Docker runtime (для GPU ускорения)
- 16GB видеопамяти (рекомендуется)

### Быстрый старт
1. **Клонирование репозитория**:
```bash
git clone https://github.com/your-username/t-logo-detector.git
cd tbank_logo_detector
```
2. **Загрузка весов модели**:
```bash
chmod +x weights/download_weights.sh
./weights/download_weights.sh
```
3. **Сборка и запуск с Docker**:
```bash
docker-compose up -d
```
4. **API будет доступен по адресу**: 
```
http://localhost:8000
```

## 🎯 Использование API
### Детекция логотипов
```bash
import requests

url = "http://localhost:8000/detect"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())
```
### Пример ответа
```json
{
  "detections": [
    {
      "bbox": {
        "x_min": 100,
        "y_min": 50, 
        "x_max": 200,
        "y_max": 150
      }
    }
  ]
}
```

## 📈 Валидация

### Для проверки качества решения:
```bash
python scripts/validate.py
```
### Валидационный датасет доступен по ссылке:

### Метрики валидации

- Precision: 0.85
- Recall: 0.78
- F1-Score: 0.81

## 🏗️ Архитектура

- Бэкенд: FastAPI
- Модель: YOLOv8s
- Инференс: PyTorch + CUDA
- Контейнеризация: Docker + Docker Compose

## 💾 Веса модели

Веса обученной модели доступны для скачивания:


## 📧 Контакты

Для вопросов и предложений:
    Email: your-email@example.com
    Issues: GitHub Issues
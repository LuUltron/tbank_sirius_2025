#!/bin/bash

# Скачивание предобученных весов
mkdir -p weights

# Скачивание предобученных весов YOLOv8
wget -P weights https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Скачивание обученных весов детектора Т-банка (будут загружены отдельно)
echo "Пожалуйста, скачайте обученные веса детектора Т-банка по ссылке:"
echo "https://example.com/tbank_logo_detector.pt"
echo "и поместите их в директорию weights/"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import io

from app.schemas.schemas import BoundingBox, Detection, DetectionResponse, ErrorResponse
from app.detection.detector import LogoDetector
from app.config import settings

app = FastAPI(title="T-Bank Logo Detector API")

detector = LogoDetector()


@app.get("/")
async def root():
    return {"message": "API Детектора логотипа Т-Банк"}


@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    # Проверка типа файла
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Supported formats: JPEG, PNG, BMP, WEBP"
        )
    
    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Конвертация в BGR для OpenCV
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Детекция логотипов
        detections = detector.detect(image)
        
        # Конвертация в формат ответа
        detection_results = []
        for bbox in detections:
            detection_results.append(Detection(bbox=BoundingBox(
                x_min=int(bbox[0]),  # Левая координата
                y_min=int(bbox[1]),  # Верхняя координата
                x_max=int(bbox[2]),  # Правая координата
                y_max=int(bbox[3])   # Нижняя координата
            )))
        
        return DetectionResponse(detections=detection_results)
        
    except Exception as e:
        # Обработка ошибок
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
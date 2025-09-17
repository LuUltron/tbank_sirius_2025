import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "T-Bank Logo Detector API"
    model_path: str = "weights/t_logo_yolov8s.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    timeout_seconds: int = 8  # Less than 10s requirement
    
    class Config:
        env_file = ".env"

settings = Settings()
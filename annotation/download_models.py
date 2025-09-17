import os
import torch
from huggingface_hub import hf_hub_download

def download_grounding_dino_weights():
    """Скачивание весов Grounding DINO"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Конфигурационные файлы и веса
    model_configs = {
        "groundingdino_swinb_cogcoor.pth": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
        "groundingdino_swint_ogc.pth": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
    }
    
    for model_name, url in model_configs.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Скачиваю {model_name}...")
            torch.hub.download_url_to_file(url, model_path)
            print(f"Готово: {model_path}")
        else:
            print(f"Модель уже существует: {model_path}")

if __name__ == "__main__":
    download_grounding_dino_weights()
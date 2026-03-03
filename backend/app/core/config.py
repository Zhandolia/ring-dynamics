from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Ring Dynamics"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://zhandolia.github.io",
        "https://ring-dynamics-api.onrender.com",
    ]
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ring_dynamics"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Video Storage
    VIDEO_STORAGE_PATH: str = "./storage/videos"
    ANNOTATED_VIDEO_PATH: str = "./storage/annotated"
    MODELS_PATH: str = "./models/weights"
    YOLO_MODEL: str = "./models/yolov8n.pt"
    
    # Processing
    MAX_VIDEO_SIZE_MB: int = 500
    FRAMES_PER_SECOND: int = 30
    ANNOTATION_SCALE: float = 0.5
    ANNOTATION_IMGSZ: int = 640
    ANNOTATION_CONF: float = 0.30
    
    # GPU
    DEVICE: str = "mps"  # 'mps' for Mac, 'cuda:0' for NVIDIA, 'cpu' for fallback
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

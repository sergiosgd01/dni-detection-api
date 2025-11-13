"""
Configuración centralizada de la aplicación
"""
import os
from typing import Set


class Settings:
    """Configuración de la aplicación DNI Detection API"""
    
    # Modelo
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/model_trained_victor_yolo11n.pt")
    
    # CORS - Orígenes permitidos
    ALLOWED_ORIGINS: list[str] = ["*"]  

    # ALLOWED_ORIGINS: list[str] = [
    #     "http://localhost:5173",
    #     "http://localhost:3000",
    #     "http://localhost:5174",
    #     "https://proyectodni.netlify.app",
    #     "*"  # ⚠️ En producción, elimina esto y deja solo dominios específicos
    # ]
    
    # YOLO - Parámetros de detección
    CONFIDENCE_THRESHOLD: float = 0.4
    MIN_CONFIDENCE_OUTPUT: float = 0.80  # Confianza mínima para devolver resultado
    MASK_THRESHOLD: float = 0.5
    IMG_SIZE: int = 640
    
    # Archivos
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.webp'}
    JPEG_QUALITY: int = 95
    
    # Servidor
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = "0.0.0.0"
    
    # Entorno
    @property
    def is_production(self) -> bool:
        """Detecta si estamos en producción (Render)"""
        return os.getenv('RENDER') is not None
    
    @property
    def environment(self) -> str:
        """Retorna el nombre del entorno"""
        return "production" if self.is_production else "development"


# Instancia única de configuración
settings = Settings()

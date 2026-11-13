"""
Configuración centralizada de la aplicación
"""
import os
from typing import Set


class Settings:
    """Configuración de la aplicación DNI Detection API"""
    
    # Modelo
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/model_trained_yolov8-seg_sergio.pt")
    
    # CORS. Solo las webs propias pueden llamar a la API desde el navegador
    DEV_ORIGINS: list[str] = [
        "http://localhost:5173",
        "https://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
    ]

    PRODUCTION_ORIGINS: list[str] = [
        "https://proyectodni.netlify.app",
        "https://defiendetuidentidad.es",
        "https://www.defiendetuidentidad.es",
        "https://defiendetuidentidad.com",
        "https://www.defiendetuidentidad.com",
        "https://protegetuidentidad.es",
        "https://www.protegetuidentidad.es",
    ]
    
    # YOLO - Parámetros de detección
    CONFIDENCE_THRESHOLD: float = 0.4
    MIN_CONFIDENCE_OUTPUT: float = 0.80  # Confianza mínima para devolver resultado
    MASK_THRESHOLD: float = 0.5
    IMG_SIZE: int = 640
    # Segundo intento para escaneos y fotos recortadas al borde del DNI. El modelo no encuentra la tarjeta si no ve
    # fondo alrededor, así que, si falla, se repite con este margen (fracción del lado mayor) de color de mesa de madera.
    # Con 0 no se reintenta.
    MARGEN_REINTENTO: float = float(os.getenv("MARGEN_REINTENTO", 0.25))
    COLOR_MARGEN_REINTENTO: tuple[int, int, int] = (52, 88, 128)  # BGR
    # El segundo intento solo vale si el DNI ocupa al menos esta fracción de la foto. Si ocupa menos, ya había fondo
    # alrededor y el primer intento falló por otra causa, así que se deja el error de antes.
    OCUPACION_MINIMA_REINTENTO: float = 0.5
    
    # Archivos
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.webp'}
    JPEG_QUALITY: int = 90
    # Lado mayor del DNI recortado. 2000 px son unos 590 ppp en un DNI de 85,6 mm, de sobra para leerlo e imprimirlo.
    # Las fotos con menos resolución se devuelven tal cual, sin ampliar.
    MAX_LADO_SALIDA: int = int(os.getenv("MAX_LADO_SALIDA", 2000))
    
    # Servidor
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = "0.0.0.0"
    
    # Entorno. Se lee de ENVIRONMENT (docker-compose ya pone "production").
    # Si no está definida se asume producción, así un despliegue sin variables
    # nunca deja la documentación abierta.
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production").strip().lower()

    @property
    def is_production(self) -> bool:
        """True salvo que ENVIRONMENT sea "development" """
        return self.ENVIRONMENT != "development"

    @property
    def environment(self) -> str:
        """Retorna el nombre del entorno"""
        return "production" if self.is_production else "development"

    @property
    def cors_origins(self) -> list[str]:
        """Orígenes permitidos. Los de localhost solo en desarrollo"""
        if self.is_production:
            return self.PRODUCTION_ORIGINS
        return self.PRODUCTION_ORIGINS + self.DEV_ORIGINS


# Instancia única de configuración
settings = Settings()

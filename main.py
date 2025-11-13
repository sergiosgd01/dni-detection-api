from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from config.settings import settings
from detection.model_utils import cargar_modelo
from detection.yolo_processor import set_model
from api.routes import router

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación
    - Carga el modelo YOLO al iniciar
    - Limpia recursos al cerrar
    """
    logger.info("🚀 Iniciando aplicación DNI Detection API (Versión Modular)...")
    logger.info(f"📍 Entorno: {settings.environment.upper()}")
    
    # Validar que existe el archivo del modelo
    model_path = Path(settings.MODEL_PATH)
    logger.info(f"🔍 Buscando modelo en: {model_path.absolute()}")
    
    if not model_path.exists():
        logger.error(f"❌ Modelo no encontrado en: {model_path.absolute()}")
        logger.error(f"📂 Archivos en directorio actual: {list(Path('.').iterdir())}")
        raise FileNotFoundError(f"Modelo no encontrado: {settings.MODEL_PATH}")
    
    # Cargar modelo
    try:
        logger.info(f"📦 Cargando modelo YOLO desde: {settings.MODEL_PATH}")
        model = cargar_modelo(str(model_path))
        set_model(model)  # Establecer modelo global en yolo_processor
        logger.info("✅ Modelo YOLO cargado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Cerrando aplicación...")
    set_model(None)


# Crear aplicación FastAPI
app = FastAPI(
    title="DNI Segmentation API",
    description="API para detección y recorte de DNI usando YOLO11n - Versión Modular",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-confidence", "x-processing-time"]
)

# Incluir rutas desde el módulo api
app.include_router(router)

# Información adicional
logger.info("📦 Arquitectura modular activada:")
logger.info("   • config/settings.py - Configuración")
logger.info("   • detection/ - Procesamiento de imágenes y YOLO")
logger.info("   • api/routes.py - Endpoints REST")


# Entry point para desarrollo local
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🌐 Iniciando servidor en {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )

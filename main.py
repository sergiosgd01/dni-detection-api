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


# La documentación (/docs, /redoc, /openapi.json) solo existe en desarrollo
docs_enabled = not settings.is_production

# Crear aplicación FastAPI
app = FastAPI(
    title="DNI Segmentation API",
    description="API para detección y recorte de DNI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None
)

# CORS. Solo las webs propias, sin credenciales
# El frontend lee la cabecera x-confidence, por eso se expone
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["x-confidence", "x-process-status"]
)

# Cabeceras de seguridad básicas en todas las respuestas
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response

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
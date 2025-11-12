from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import io
import cv2
import numpy as np
from typing import Optional
import logging
import os

from detection.model_utils import cargar_modelo
from detection.geometry_utils import corregir_perspectiva

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
class Config:
    # 🔧 Ruta relativa para Render (funciona tanto local como en producción)
    MODEL_PATH = os.getenv("MODEL_PATH", "model/model_trained_victor_yolo11n.pt")
    
    # 🌐 CORS - Añade aquí tu dominio de React cuando lo despliegues
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
        "https://your-react-app.vercel.app",  # ⚠️ Cambia esto por tu URL de React
        "https://your-react-app.netlify.app", # ⚠️ Cambia esto por tu URL de React
        "*"  # ⚠️ En producción, elimina esto y deja solo tus dominios específicos
    ]
    
    CONFIDENCE_THRESHOLD = 0.4
    MASK_THRESHOLD = 0.5
    IMG_SIZE = 640
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    JPEG_QUALITY = 95
    
    # 🚀 Puerto configurable para Render
    PORT = int(os.getenv("PORT", 8000))

# Variable global para el modelo
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación"""
    global model
    logger.info("🚀 Iniciando aplicación DNI Detection API...")
    logger.info(f"📍 Entorno: {'PRODUCCIÓN (Render)' if os.getenv('RENDER') else 'DESARROLLO (Local)'}")
    
    # Validar que existe el archivo del modelo
    model_path = Path(Config.MODEL_PATH)
    logger.info(f"🔍 Buscando modelo en: {model_path.absolute()}")
    
    if not model_path.exists():
        logger.error(f"❌ Modelo no encontrado en: {model_path.absolute()}")
        logger.error(f"📂 Archivos en directorio actual: {list(Path('.').iterdir())}")
        raise FileNotFoundError(f"Modelo no encontrado: {Config.MODEL_PATH}")
    
    # Cargar modelo
    try:
        logger.info(f"📦 Cargando modelo YOLO desde: {Config.MODEL_PATH}")
        model = cargar_modelo(str(model_path))
        logger.info("✅ Modelo YOLO cargado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}", exc_info=True)
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Cerrando aplicación...")
    model = None

# Crear aplicación FastAPI
app = FastAPI(
    title="DNI Segmentation API",
    description="API para detección y recorte de DNI usando YOLO11n",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# 🌐 Configurar CORS (crítico para React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-confidence", "x-processing-time"]
)

# Utilidades
def validate_image_file(file: UploadFile) -> None:
    """Valida el archivo de imagen subido"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo no tiene nombre"
        )
    
    # Validar extensión
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no permitido. Use: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    
    # Validar tipo MIME
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser una imagen válida"
        )

def decode_image(contents: bytes) -> Optional[np.ndarray]:
    """Decodifica bytes a imagen OpenCV"""
    try:
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        return frame
    except Exception as e:
        logger.error(f"Error decodificando imagen: {e}")
        return None

def process_yolo_detection(frame: np.ndarray):
    """Ejecuta la detección YOLO y procesa la máscara"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    # Inferencia YOLO
    results = model.predict(
        source=frame,
        conf=Config.CONFIDENCE_THRESHOLD,
        imgsz=Config.IMG_SIZE,
        verbose=False
    )
    
    # ✅ CASO 1: No se detectó nada
    if not results or len(results) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_detection",
                "message": "No se detectó ningún DNI en la imagen",
                "suggestion": "Asegúrate de que:\n• El DNI esté dentro del recuadro\n• El DNI sea visible y esté bien iluminado\n• No haya objetos tapando el DNI",
                "action": "retry"
            }
        )
    
    result = results[0]
    
    # ✅ CASO 2: Detección sin máscaras
    if result.masks is None or len(result.masks) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_mask",
                "message": "No se pudo identificar claramente el DNI",
                "suggestion": "Intenta:\n• Acercar más el DNI a la cámara\n• Mejorar la iluminación\n• Reducir el movimiento",
                "action": "retry"
            }
        )
    
    # Obtener máscara con mayor confianza
    mask = result.masks.data[0].cpu().numpy()
    
    # Obtener confianza de detección
    confidence = float(result.boxes.conf[0].cpu().numpy()) if result.boxes else None
    
    return mask, confidence

def create_binary_mask(mask: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Crea máscara binaria redimensionada"""
    h, w = frame_shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    mask_bin = (mask_resized > Config.MASK_THRESHOLD).astype(np.uint8) * 255
    return mask_bin

# 📍 Endpoints

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "service": "DNI Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "environment": "production" if os.getenv('RENDER') else "development",
        "model_loaded": model is not None,
        "endpoints": {
            "process": "/process",
            "process_debug": "/process-debug",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio (para Render health checks)"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "environment": "production" if os.getenv('RENDER') else "development"
    }

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    🎯 Endpoint principal para procesar imágenes de DNI
    
    - **file**: Imagen del DNI (JPG, PNG, WEBP, max 10MB)
    
    Returns:
        - **200**: Imagen procesada (image/jpeg) con header x-confidence
        - **422**: Errores controlados (no_detection, no_mask, low_confidence, extraction_failed)
        - **400**: Error en el formato de archivo
        - **413**: Archivo muy grande
        - **500**: Error interno del servidor
    """
    try:
        # Validar archivo
        validate_image_file(file)
        
        # Leer y validar tamaño
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Archivo muy grande. Máximo: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Decodificar imagen
        frame = decode_image(contents)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo decodificar la imagen. Archivo corrupto o formato inválido"
            )
        
        logger.info(f"📸 Procesando imagen: {file.filename} ({frame.shape[1]}x{frame.shape[0]})")
        
        # Detección YOLO
        mask, confidence = process_yolo_detection(frame)
        logger.info(f"🎯 DNI detectado con confianza: {confidence:.2%}")
        
        # ✅ CASO 3: Confianza insuficiente (< 80%)
        MIN_CONFIDENCE = 0.80
        if confidence is not None and confidence < MIN_CONFIDENCE:
            confidence_percent = f"{confidence * 100:.1f}%"
            logger.warning(f"⚠️ Confianza insuficiente: {confidence_percent}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "low_confidence",
                    "message": f"La imagen del DNI no es suficientemente clara (confianza: {confidence_percent})",
                    "confidence": float(confidence),
                    "min_required": MIN_CONFIDENCE,
                    "suggestion": "Para una mejor captura:\n• Mejora la iluminación\n• Mantén la cámara estable\n• Asegúrate de que el DNI esté enfocado\n• Evita reflejos y sombras",
                    "action": "retry"
                }
            )
        
        # Crear máscara binaria
        mask_bin = create_binary_mask(mask, frame.shape)
        
        # Corregir perspectiva y recortar
        dni_crop = corregir_perspectiva(frame, mask_bin)
        
        # ✅ CASO 4: No se pudo extraer el DNI
        if dni_crop is None or dni_crop.size == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "extraction_failed",
                    "message": "No se pudo extraer el DNI correctamente",
                    "suggestion": "Verifica que:\n• El DNI esté completamente visible\n• No esté parcialmente tapado\n• Esté dentro del recuadro guía",
                    "action": "retry"
                }
            )
        
        logger.info(f"✂️  DNI recortado: {dni_crop.shape[1]}x{dni_crop.shape[0]}")
        
        # Codificar imagen de salida
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        success, img_encoded = cv2.imencode('.jpg', dni_crop, encode_params)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al codificar la imagen procesada"
            )
        
        # ✅ ÉXITO: Crear respuesta
        response = StreamingResponse(
            io.BytesIO(img_encoded.tobytes()),
            media_type="image/jpeg"
        )
        
        # Headers personalizados
        if confidence is not None:
            response.headers["x-confidence"] = str(confidence)
        
        logger.info(f"✅ Procesamiento exitoso - Confianza: {confidence:.1%}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.post("/process-debug")
async def process_image_debug(file: UploadFile = File(...)):
    """
    🔍 Versión de debug que devuelve JSON con información detallada
    Útil para testing y desarrollo
    """
    try:
        validate_image_file(file)
        contents = await file.read()
        frame = decode_image(contents)
        
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Imagen inválida"}
            )
        
        mask, confidence = process_yolo_detection(frame)
        mask_bin = create_binary_mask(mask, frame.shape)
        dni_crop = corregir_perspectiva(frame, mask_bin)
        
        return {
            "success": dni_crop is not None,
            "confidence": float(confidence) if confidence else None,
            "original_size": {"width": frame.shape[1], "height": frame.shape[0]},
            "cropped_size": {"width": dni_crop.shape[1], "height": dni_crop.shape[0]} if dni_crop is not None else None,
            "mask_points": int(np.sum(mask_bin > 0)),
            "environment": "production" if os.getenv('RENDER') else "development"
        }
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error en debug: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 🚀 Entry point
if __name__ == "__main__":
    import uvicorn
    
    # Para desarrollo local
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=True,  # Hot reload en desarrollo
        log_level="info"
    )
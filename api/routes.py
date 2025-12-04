"""
Endpoints de la API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import io
import logging

from config.settings import settings
from detection.validators import validate_image_file, validate_file_size
from detection.image_utils import decode_image, encode_image_to_jpeg
# process_yolo_detection ahora devuelve la imagen final procesada
from detection.yolo_processor import process_yolo_detection, validate_confidence, get_model

logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()


@router.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "service": "DNI Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment,
        "model_loaded": get_model() is not None,
        "endpoints": {
            "process": "/process",
            "process_debug": "/process-debug",
            "health": "/health",
            "docs": "/docs"
        }
    }


@router.get("/health")
async def health_check():
    """
    Verifica el estado del servicio
    """
    if get_model() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "environment": settings.environment
    }


@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    🎯 Endpoint principal: Detecta -> Recorta -> Mejora -> Devuelve JPEG
    """
    try:
        # 1. Validaciones iniciales
        validate_image_file(file)
        contents = await file.read()
        validate_file_size(contents)
        
        # 2. Decodificar imagen
        frame = decode_image(contents)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo decodificar la imagen. Archivo corrupto."
            )
        
        logger.info(f"📸 Procesando: {file.filename} ({frame.shape[1]}x{frame.shape[0]})")
        
        # 3. Procesamiento Integral (YOLO + Warp + Enhance)
        # process_yolo_detection ahora retorna la imagen final lista
        processed_image, confidence = process_yolo_detection(frame)
        
        logger.info(f"🎯 DNI procesado con confianza: {confidence:.2%}")
        
        # 4. Validar confianza
        validate_confidence(confidence)
        
        # 5. Codificar imagen de salida
        img_bytes = encode_image_to_jpeg(processed_image)
        if img_bytes is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al codificar la imagen procesada"
            )
        
        # 6. Crear respuesta streaming
        response = StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg"
        )
        
        # Headers informativos
        response.headers["x-confidence"] = str(confidence)
        response.headers["x-process-status"] = "success"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )


@router.post("/process-debug")
async def process_image_debug(file: UploadFile = File(...)):
    """
    🔍 Endpoint de Debug: Devuelve JSON con metadatos del proceso
    """
    try:
        validate_image_file(file)
        contents = await file.read()
        validate_file_size(contents)
        
        frame = decode_image(contents)
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Imagen inválida"}
            )
        
        # Ejecutar proceso
        processed_image, confidence = process_yolo_detection(frame)
        
        return {
            "success": True,
            "confidence": float(confidence) if confidence else 0.0,
            "original_size": {
                "width": frame.shape[1], 
                "height": frame.shape[0]
            },
            "processed_size": {
                "width": processed_image.shape[1], 
                "height": processed_image.shape[0]
            },
            "environment": settings.environment
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
"""
Endpoints de la API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import io
import numpy as np
import logging

from config.settings import settings
from detection.validators import validate_image_file, validate_file_size
from detection.image_utils import decode_image, create_binary_mask, encode_image_to_jpeg
from detection.yolo_processor import process_yolo_detection, validate_confidence, get_model
from detection.geometry_utils import corregir_perspectiva

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
    Verifica el estado del servicio (para Render health checks)
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
        # 1. Validar archivo
        validate_image_file(file)
        
        # 2. Leer y validar tamaño
        contents = await file.read()
        validate_file_size(contents)
        
        # 3. Decodificar imagen
        frame = decode_image(contents)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo decodificar la imagen. Archivo corrupto o formato inválido"
            )
        
        logger.info(f"📸 Procesando imagen: {file.filename} ({frame.shape[1]}x{frame.shape[0]})")
        
        # 4. Detección YOLO
        mask, confidence = process_yolo_detection(frame)
        logger.info(f"🎯 DNI detectado con confianza: {confidence:.2%}")
        
        # 5. Validar confianza
        validate_confidence(confidence)
        
        # 6. Crear máscara binaria
        mask_bin = create_binary_mask(mask, frame.shape)
        
        # 7. Corregir perspectiva y recortar
        dni_crop = corregir_perspectiva(frame, mask_bin)
        
        # 8. Validar que se extrajo correctamente
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
        
        # 9. Codificar imagen de salida
        img_bytes = encode_image_to_jpeg(dni_crop)
        if img_bytes is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al codificar la imagen procesada"
            )
        
        # 10. Crear respuesta
        response = StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg"
        )
        
        # Añadir header con confianza
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


@router.post("/process-debug")
async def process_image_debug(file: UploadFile = File(...)):
    """
    🔍 Versión de debug que devuelve JSON con información detallada
    Útil para testing y desarrollo
    """
    try:
        # Validar y procesar
        validate_image_file(file)
        contents = await file.read()
        validate_file_size(contents)
        
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

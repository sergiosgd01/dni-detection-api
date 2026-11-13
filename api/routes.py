"""
Endpoints de la API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import logging

from detection.validators import validate_image_file, validate_file_size
from detection.image_utils import decode_image, encode_image_to_jpeg
# process_yolo_detection ahora devuelve la imagen final procesada
from detection.yolo_processor import process_yolo_detection, validate_confidence, get_model

logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()


@router.get("/")
async def root():
    """Endpoint raíz. No expone detalles internos"""
    return {
        "service": "DNI Segmentation API",
        "status": "running"
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
        "model_loaded": True
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
        
        # No se registra el nombre del archivo, a menudo incluye nombres o números de DNI
        logger.info(f"📸 Procesando imagen ({frame.shape[1]}x{frame.shape[0]})")
        
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
        
        # 6. Respuesta de una sola pieza. StreamingResponse con BytesIO troceaba el JPEG por cada byte 0x0A,
        # unos 26.000 fragmentos en una respuesta de 9 MB, y solo el envío tardaba más de 2 segundos
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={"x-confidence": str(confidence), "x-process-status": "success"},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # El detalle se queda en el log del servidor, al cliente solo le llega un mensaje genérico
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la imagen. Inténtalo de nuevo."
        )

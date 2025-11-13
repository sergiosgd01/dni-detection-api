"""
Procesamiento de detección con YOLO
"""
import numpy as np
from fastapi import HTTPException, status
import logging
from typing import Tuple, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Variable global para el modelo (se inicializa en main.py)
model = None


def set_model(yolo_model):
    """
    Establece el modelo YOLO global
    
    Args:
        yolo_model: Modelo YOLO cargado
    """
    global model
    model = yolo_model


def get_model():
    """
    Obtiene el modelo YOLO global
    
    Returns:
        Modelo YOLO o None si no está cargado
    """
    return model


def process_yolo_detection(frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Ejecuta la detección YOLO y procesa la máscara
    
    Args:
        frame: Imagen como array de NumPy (BGR)
        
    Returns:
        Tupla (máscara, confianza)
        
    Raises:
        HTTPException: Si hay errores en la detección
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    # Inferencia YOLO
    results = model.predict(
        source=frame,
        conf=settings.CONFIDENCE_THRESHOLD,
        imgsz=settings.IMG_SIZE,
        verbose=False
    )
    
    # CASO 1: No se detectó nada
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
    
    # CASO 2: Detección sin máscaras
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


def validate_confidence(confidence: Optional[float]) -> None:
    """
    Valida que la confianza sea suficiente
    
    Args:
        confidence: Nivel de confianza de la detección
        
    Raises:
        HTTPException: Si la confianza es insuficiente
    """
    if confidence is not None and confidence < settings.MIN_CONFIDENCE_OUTPUT:
        confidence_percent = f"{confidence * 100:.1f}%"
        logger.warning(f"⚠️ Confianza insuficiente: {confidence_percent}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "low_confidence",
                "message": f"La imagen del DNI no es suficientemente clara (confianza: {confidence_percent})",
                "confidence": float(confidence),
                "min_required": settings.MIN_CONFIDENCE_OUTPUT,
                "suggestion": "Para una mejor captura:\n• Mejora la iluminación\n• Mantén la cámara estable\n• Asegúrate de que el DNI esté enfocado\n• Evita reflejos y sombras",
                "action": "retry"
            }
        )

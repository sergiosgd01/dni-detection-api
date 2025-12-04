"""
Procesamiento de detección con YOLO
"""
import numpy as np
import cv2
from fastapi import HTTPException, status
import logging
from typing import Tuple, Optional

from config.settings import settings
from detection.geometry_utils import four_point_transform 

logger = logging.getLogger(__name__)

# Variable global para el modelo
model = None

def set_model(yolo_model):
    global model
    model = yolo_model

def get_model():
    return model

def process_yolo_detection(frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Ejecuta detección YOLO y aplica la tubería de corrección.
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
        raise_detection_error("no_detection", "No se detectó ningún DNI en la imagen")
    
    result = results[0]
    
    # CASO 2: Detección sin máscaras
    if result.masks is None or len(result.masks) == 0:
        raise_detection_error("no_mask", "No se pudo identificar la segmentación del DNI")

    # Obtener la detección con mayor confianza
    if result.boxes:
        confidences = result.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        confidence = float(confidences[best_idx])
    else:
        confidence = 0.0
        best_idx = 0

    # Obtener coordenadas XY del polígono
    mask_data = result.masks.xy[best_idx]
    
    if len(mask_data) < 4:
        raise_detection_error("invalid_mask", "La máscara detectada tiene muy pocos puntos")

    contour = np.array(mask_data, dtype=np.int32)
    
    # Intentar aproximación a 4 puntos
    approx = None
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        temp_approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(temp_approx) == 4:
            approx = temp_approx
            break
    
    # Si no funciona, usar minAreaRect
    if approx is None or len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = np.int32(box)
    
    # Validar y Transformar
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # Aplicar transformación
        warped_dni = four_point_transform(frame, pts, scale=4)
        
        if warped_dni is None:
            raise_detection_error("warp_error", "Error calculando la perspectiva")

        # 👈 Devolver directamente sin enhance (mantiene color original)
        return warped_dni, confidence
    else:
        raise_detection_error("geometry_error", "No se pudieron determinar las 4 esquinas del DNI")


def validate_confidence(confidence: Optional[float]) -> None:
    if confidence is not None and confidence < settings.MIN_CONFIDENCE_OUTPUT:
        confidence_percent = f"{confidence * 100:.1f}%"
        logger.warning(f"⚠️ Confianza insuficiente: {confidence_percent}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "low_confidence",
                "message": f"Detección poco clara ({confidence_percent}). Mejora la iluminación.",
                "action": "retry"
            }
        )


def raise_detection_error(code: str, msg: str):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "error": code,
            "message": msg,
            "action": "retry"
        }
    )
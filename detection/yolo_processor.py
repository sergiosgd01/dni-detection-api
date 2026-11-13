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

class DeteccionFallida(Exception):
    """El modelo no da un contorno del DNI utilizable. Lleva el código y el mensaje que verá el cliente"""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code
        self.msg = msg


def segmentar(imagen: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pasa el modelo por la imagen y devuelve el contorno del DNI con más confianza y esa confianza.
    """
    results = model.predict(
        source=imagen,
        conf=settings.CONFIDENCE_THRESHOLD,
        imgsz=settings.IMG_SIZE,
        verbose=False
    )
    
    # CASO 1: No se detectó nada
    if not results or len(results) == 0:
        raise DeteccionFallida("no_detection", "No se detectó ningún DNI en la imagen")
    
    result = results[0]
    
    # CASO 2: Detección sin máscaras
    if result.masks is None or len(result.masks) == 0:
        raise DeteccionFallida("no_mask", "No se pudo identificar la segmentación del DNI")

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
        raise DeteccionFallida("invalid_mask", "La máscara detectada tiene muy pocos puntos")

    return mask_data, confidence


def segmentar_con_margen(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Repite la detección con un margen alrededor de la foto y devuelve el contorno en coordenadas de la foto original.

    Sirve para escaneos y fotos recortadas al borde del DNI, en las que el modelo no ve fondo alrededor de la
    tarjeta. La foto se reduce antes al tamaño con el que trabaja el modelo, así el margen no ocupa memoria de más.
    El recorte final se hace después sobre la foto original, con su resolución.
    """
    alto, ancho = frame.shape[:2]
    escala = min(1.0, settings.IMG_SIZE / (max(alto, ancho) * (1 + 2 * settings.MARGEN_REINTENTO)))
    imagen = frame
    if escala < 1.0:
        imagen = cv2.resize(frame, (max(1, round(ancho * escala)), max(1, round(alto * escala))),
                            interpolation=cv2.INTER_AREA)
    escala_x, escala_y = imagen.shape[1] / ancho, imagen.shape[0] / alto
    borde = round(max(imagen.shape[:2]) * settings.MARGEN_REINTENTO)
    imagen = cv2.copyMakeBorder(imagen, borde, borde, borde, borde, cv2.BORDER_CONSTANT,
                                value=settings.COLOR_MARGEN_REINTENTO)

    contorno, confianza = segmentar(imagen)

    # De vuelta a la foto original, sin salirse de ella
    contorno = np.array(contorno, dtype=np.float32)
    contorno[:, 0] = np.clip((contorno[:, 0] - borde) / escala_x, 0, ancho)
    contorno[:, 1] = np.clip((contorno[:, 1] - borde) / escala_y, 0, alto)
    return contorno, confianza


def process_yolo_detection(frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Ejecuta detección YOLO y aplica la tubería de corrección.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )

    mask_data, confidence, fallo = None, 0.0, None
    try:
        mask_data, confidence = segmentar(frame)
    except DeteccionFallida as e:
        fallo = e

    # Si no hay un DNI claro se prueba otra vez con margen. Solo se usa si esta vez se reconoce y el DNI llena
    # la foto. Las fotos que ya se reconocían dan el mismo recorte y las demás, el mismo error que antes
    if confidence < settings.MIN_CONFIDENCE_OUTPUT and settings.MARGEN_REINTENTO > 0:
        try:
            mask_reintento, confidence_reintento = segmentar_con_margen(frame)
            ocupacion = cv2.contourArea(mask_reintento) / (frame.shape[0] * frame.shape[1])
            logger.info(f"🔁 Reintento con margen: confianza {confidence:.2%} -> {confidence_reintento:.2%}, "
                        f"el DNI ocupa el {ocupacion:.0%} de la foto")
            if (confidence_reintento >= settings.MIN_CONFIDENCE_OUTPUT
                    and ocupacion >= settings.OCUPACION_MINIMA_REINTENTO):
                mask_data, confidence, fallo = mask_reintento, confidence_reintento, None
        except DeteccionFallida as e:
            logger.info(f"🔁 Reintento con margen sin resultado ({e.code})")

    if fallo is not None:
        raise_detection_error(fallo.code, fallo.msg)

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
        warped_dni = four_point_transform(frame, pts, max_lado=settings.MAX_LADO_SALIDA)
        
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
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def order_points(pts):
    """
    Ordena los puntos: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Suma de coordenadas: esquina superior izquierda tiene menor suma, inferior derecha mayor
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    
    # Diferencia de coordenadas: esquina superior derecha tiene menor diff, inferior izquierda mayor
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    
    return rect

def four_point_transform(image, pts, max_lado=None):
    """
    Endereza el DNI a partir de sus 4 esquinas.

    La salida conserva la resolución que tiene el DNI en la foto, sin ampliarla. Ampliar no añade
    detalle y multiplicaba por 16 los píxeles que había que codificar, enviar y procesar en el navegador.

    Args:
        image: Imagen original
        pts: Array de 4 puntos (ordenados o no)
        max_lado: Lado mayor máximo de la salida en píxeles. None para no limitarlo
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 1. Calcular dimensiones originales
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Validar dimensiones mínimas para evitar errores de warp
    if maxWidth < 50 or maxHeight < 50:
        logger.warning(f"Dimensiones muy pequeñas detectadas ({maxWidth}x{maxHeight})")
        return None

    # 2. Limitar el tamaño de salida (nunca se amplía)
    escala = 1.0
    if max_lado and max(maxWidth, maxHeight) > max_lado:
        escala = max_lado / max(maxWidth, maxHeight)
    if escala < 0.75:
        # Si hay que reducir mucho, se reduce antes la foto con INTER_AREA para que no salgan dientes de sierra
        image = cv2.resize(image, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        rect = rect * escala
        escala = 1.0
        maxWidth = max(int(np.linalg.norm(rect[2] - rect[3])), int(np.linalg.norm(rect[1] - rect[0])))
        maxHeight = max(int(np.linalg.norm(rect[1] - rect[2])), int(np.linalg.norm(rect[0] - rect[3])))
    dstWidth = max(1, round(maxWidth * escala))
    dstHeight = max(1, round(maxHeight * escala))

    # 3. Puntos de destino
    dst = np.array([
        [0, 0],
        [dstWidth - 1, 0],
        [dstWidth - 1, dstHeight - 1],
        [0, dstHeight - 1]], dtype="float32")

    # 4. Transformación de perspectiva con interpolación LANCZOS4
    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dstWidth, dstHeight), 
                                   flags=cv2.INTER_LANCZOS4)
        return warped
    except Exception as e:
        logger.error(f"Error en cv2.warpPerspective: {e}")
        return None
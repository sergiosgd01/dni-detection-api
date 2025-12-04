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

def four_point_transform(image, pts, scale=4):
    """
    Aplica transformación de perspectiva a 4 puntos con escalado.
    
    Args:
        image: Imagen original
        pts: Array de 4 puntos (ordenados o no)
        scale: Factor para aumentar la resolución de salida (Default: 4 según tu script)
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
    
    # 2. Aplicar el escalado
    dstWidth = maxWidth * scale
    dstHeight = maxHeight * scale
    
    # 3. Puntos de destino ajustados al nuevo tamaño escalado
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
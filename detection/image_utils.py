"""
Utilidades para procesamiento de imágenes
"""
import cv2
import numpy as np
from typing import Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def decode_image(contents: bytes) -> Optional[np.ndarray]:
    """
    Decodifica bytes a imagen OpenCV
    
    Args:
        contents: Bytes de la imagen
        
    Returns:
        Imagen como array de NumPy (BGR) o None si falla
    """
    try:
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("No se pudo decodificar la imagen")
            return None
        
        return frame
    except Exception as e:
        logger.error(f"Error decodificando imagen: {e}")
        return None


def create_binary_mask(mask: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """
    Crea máscara binaria redimensionada
    
    Args:
        mask: Máscara de segmentación de YOLO
        frame_shape: Forma de la imagen original (height, width, channels)
        
    Returns:
        Máscara binaria redimensionada (0 o 255)
    """
    h, w = frame_shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    mask_bin = (mask_resized > settings.MASK_THRESHOLD).astype(np.uint8) * 255
    return mask_bin


def encode_image_to_jpeg(image: np.ndarray, quality: int = None) -> Optional[bytes]:
    """
    Codifica imagen a JPEG
    
    Args:
        image: Imagen como array de NumPy
        quality: Calidad JPEG (1-100), por defecto usa settings.JPEG_QUALITY
        
    Returns:
        Bytes de la imagen JPEG o None si falla
    """
    if quality is None:
        quality = settings.JPEG_QUALITY
    
    try:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, img_encoded = cv2.imencode('.jpg', image, encode_params)
        
        if not success:
            logger.error("Error al codificar la imagen")
            return None
        
        return img_encoded.tobytes()
    except Exception as e:
        logger.error(f"Error codificando imagen: {e}")
        return None

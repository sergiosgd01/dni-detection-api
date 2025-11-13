"""
Validadores para la API
"""
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def validate_image_file(file: UploadFile) -> None:
    """
    Valida el archivo de imagen subido
    
    Args:
        file: Archivo subido por el usuario
        
    Raises:
        HTTPException: Si el archivo no es válido
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo no tiene nombre"
        )
    
    # Validar extensión
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no permitido. Use: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Validar tipo MIME
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser una imagen válida"
        )


def validate_file_size(contents: bytes) -> None:
    """
    Valida el tamaño del archivo
    
    Args:
        contents: Contenido del archivo en bytes
        
    Raises:
        HTTPException: Si el archivo es muy grande
    """
    if len(contents) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archivo muy grande. Máximo: {max_mb}MB"
        )

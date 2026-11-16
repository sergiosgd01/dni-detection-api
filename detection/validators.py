"""
Validadores para la API
"""
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
from PIL import Image
import io
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def detalle_error(codigo: str, mensaje: str, sugerencia: str = "") -> dict:
    """Detalle de un error con el formato del resto de la API, que es el que lee la web (processWithYolo.js)"""
    detalle = {"error": codigo, "message": mensaje, "action": "retry"}
    if sugerencia:
        detalle["suggestion"] = sugerencia
    return detalle


def error_archivo_grande() -> HTTPException:
    """413 para una foto que pesa más de lo permitido. Lo usan la comprobación del cuerpo y la del archivo"""
    max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=detalle_error(
            "file_too_large",
            f"La foto pesa más de {max_mb} MB, el máximo que admite el servidor.",
            "Prueba con otra foto o hazla con menos resolución."
        )
    )


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
        raise error_archivo_grande()


def validate_image_dimensions(contents: bytes) -> None:
    """
    Rechaza la foto si tiene demasiados píxeles, antes de decodificarla.

    Pillow lee el ancho y el alto de la cabecera del archivo sin descomprimir la imagen, así que una bomba de
    descompresión (un PNG pequeño de 30000 x 30000) no llega a ocupar memoria. Si Pillow no reconoce la imagen
    se contesta lo mismo que cuando OpenCV no la puede decodificar.

    Raises:
        HTTPException: 413 si tiene más píxeles de los permitidos, 400 si no se puede leer
    """
    try:
        with Image.open(io.BytesIO(contents)) as imagen:
            ancho, alto = imagen.size
    except Image.DecompressionBombError:
        # Pillow corta por su cuenta las imágenes de más de 179 MP, muy por encima del límite de aquí
        ancho, alto = None, None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo decodificar la imagen. Archivo corrupto."
        )

    if ancho is None or ancho * alto > settings.max_pixeles:
        megapixeles = f" ({ancho * alto / 1_000_000:.0f} MP)" if ancho is not None else ""
        logger.warning(f"🧱 Foto rechazada por tener demasiados píxeles{megapixeles}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=detalle_error(
                "image_too_large",
                f"La foto tiene demasiada resolución{megapixeles}. El máximo son {settings.MAX_MEGAPIXELES:g} MP.",
                "Prueba con otra foto o hazla con menos resolución."
            )
        )

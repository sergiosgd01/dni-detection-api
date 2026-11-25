"""
Procesamiento de detección con YOLO
"""
import numpy as np
import cv2
from fastapi import HTTPException, status
import logging
from ultralytics.data.converter import merge_multi_segment
from ultralytics.utils import ops
from typing import Tuple, Optional

from config.settings import settings
from detection.esquinas import (angulo_del_dni, esquinas_por_lados, esquinas_por_poligono, girar_puntos, mismo_dni,
                                segmentar_de_cerca, separacion_esquinas)
from detection.geometry_utils import four_point_transform, order_points

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

    mask_data = contorno_sin_manchas(result, best_idx)

    if len(mask_data) < 4:
        raise DeteccionFallida("invalid_mask", "La máscara detectada tiene muy pocos puntos")

    return np.asarray(mask_data, dtype=np.float32), confidence


def contorno_sin_manchas(result, indice: int) -> np.ndarray:
    """
    Contorno de la máscara en píxeles de la foto, como `result.masks.xy[indice]` pero sin las manchas sueltas.

    Ultralytics junta en un solo contorno todos los trozos de la máscara, y a veces hay una mancha pequeña en el fondo,
    lejos del DNI. El contorno saltaba hasta ella y torcía las esquinas. Ahora se quitan los trozos de menos del 5 %
    del área del mayor. Si el DNI queda partido en trozos grandes (un dedo que lo cruza), se siguen juntando.
    """
    mascara = result.masks.data[indice].byte().cpu().numpy()
    contornos = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contornos:
        return np.zeros((0, 2), dtype=np.float32)
    areas = [cv2.contourArea(c) for c in contornos]
    grandes = [c.reshape(-1, 2) for c, a in zip(contornos, areas) if a >= 0.05 * max(areas)]
    puntos = np.concatenate(merge_multi_segment(grandes)) if len(grandes) > 1 else grandes[0]
    return ops.scale_coords(result.masks.data.shape[1:], puntos.astype(np.float32), result.masks.orig_shape,
                            normalize=False)


def calentar_modelo() -> None:
    """
    Pasa el modelo una vez por una imagen vacía.

    Ultralytics prepara el modelo (fusiona capas y crea el predictor) en la primera predicción. Si eso pasa con dos
    fotos a la vez en dos hilos, las dos lo preparan al mismo tiempo. Después las predicciones ya van de una en una,
    porque el predictor tiene su propio cerrojo.
    """
    try:
        segmentar(np.zeros((settings.IMG_SIZE, settings.IMG_SIZE, 3), dtype=np.uint8))
    except DeteccionFallida:
        pass


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


def process_yolo_detection(frame: np.ndarray) -> Tuple[np.ndarray, Optional[float], np.ndarray]:
    """
    Ejecuta detección YOLO y aplica la tubería de corrección.

    Devuelve el DNI enderezado, la confianza y las cuatro esquinas con las que se ha recortado. Las esquinas
    van en píxeles de la foto recibida y en el orden superior izquierda, superior derecha, inferior derecha
    e inferior izquierda, el mismo con el que se endereza.
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

    # Un DNI lejos se mira de cerca antes de rechazarlo (FLU-4, ver confianza_de_cerca). El recorte sale igual que
    # si la primera pasada hubiera llegado al límite
    if fallo is None and confidence < settings.MIN_CONFIDENCE_OUTPUT:
        confidence = confianza_de_cerca(frame, np.array(mask_data, dtype=np.float32), confidence)

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

    pts = esquinas_del_dni(frame, np.array(mask_data, dtype=np.float32))

    warped_dni = four_point_transform(frame, pts, max_lado=settings.MAX_LADO_SALIDA)
    if warped_dni is None:
        raise_detection_error("warp_error", "Error calculando la perspectiva")

    # Se devuelve sin retocar, con el color original
    return warped_dni, confidence, order_points(pts)


def confianza_de_cerca(frame: np.ndarray, contorno: np.ndarray, confianza: float) -> float:
    """
    La confianza del DNI mirado de cerca, si sale pequeño en la foto (FLU-4), o la de la primera pasada.

    Un DNI que ocupa poco de la foto llega al modelo con pocos píxeles y le da menos confianza aunque se vea bien.
    En las fotos reales del banco, con el DNI lejos la primera pasada daba entre 0,70 y 0,80 y la pasada de cerca
    entre 0,90 y 0,97. Es la misma pasada que usan las esquinas (ver esquinas_nuevas), con el DNI derecho y
    ocupando casi toda la imagen del modelo. Su confianza vale solo si es el mismo DNI (ver mismo_dni).
    """
    if not settings.CONFIANZA_DE_CERCA:
        return confianza
    alto, ancho = frame.shape[:2]
    _, _, w, h = cv2.boundingRect(contorno.reshape(-1, 2))
    if max(w, h) >= settings.DNI_PEQUENO * max(ancho, alto):
        return confianza
    try:
        angulo = angulo_del_dni(contorno)
        girado = settings.GIRO_SEGUNDA_PASADA > 0 and abs(angulo) >= settings.GIRO_SEGUNDA_PASADA
        cerca, confianza_cerca = segmentar_de_cerca(frame, contorno, angulo if girado else 0.0, segmentar,
                                                    settings.IMG_SIZE, settings.MARGEN_SEGUNDA_PASADA)
    except DeteccionFallida as e:
        logger.info(f"🔍 DNI pequeño sin resultado de cerca ({e.code}), se queda en {confianza:.2%}")
        return confianza
    except Exception:
        # Un fallo inesperado no debe tumbar la petición. Se sigue como antes de FLU-4
        logger.exception(f"🔍 Error al mirar de cerca un DNI pequeño, se queda en {confianza:.2%}")
        return confianza
    if not mismo_dni(contorno, cerca):
        logger.info(f"🔍 DNI pequeño, de cerca sale otro contorno ({confianza_cerca:.2%}), se queda en {confianza:.2%}")
        return confianza
    logger.info(f"🔍 DNI pequeño en la foto, confianza {confianza:.2%} y de cerca {confianza_cerca:.2%}")
    return max(confianza, confianza_cerca)


def esquinas_del_dni(frame: np.ndarray, contorno: np.ndarray) -> np.ndarray:
    """
    Las cuatro esquinas del DNI en píxeles de la foto (FLU-3, ver detection/esquinas.py).

    Se calculan de las dos formas, con el polígono de siempre y con las rectas de los lados (con otra pasada del
    modelo de cerca si el DNI está girado o lejos), y se usan las de siempre salvo que las nuevas se separen de ellas
    más de SEPARACION_ESQUINAS de la diagonal. Las nuevas se ajustan más a la tarjeta, pero el modelo de campos de la
    web se entrenó con los recortes de siempre, y en fotos reales con el DNI derecho un recorte un 2 a 4 % distinto le
    hacía perder algún campo. Solo cuando el polígono falla de verdad (la sombra comida con el DNI girado, un dedo en
    una esquina) las esquinas se separan más, y entonces las nuevas son mucho mejores.
    """
    antiguas = esquinas_por_poligono(contorno)
    nuevas = esquinas_nuevas(frame, contorno)
    if nuevas is None:
        return antiguas
    separacion = separacion_esquinas(antiguas, nuevas)
    if separacion < settings.SEPARACION_ESQUINAS:
        return antiguas
    logger.info(f"📐 Esquinas por los lados, a un {separacion:.1%} de la diagonal de las del polígono")
    return nuevas


def esquinas_nuevas(frame: np.ndarray, contorno: np.ndarray) -> Optional[np.ndarray]:
    """Las esquinas por las rectas de los lados, o None si no salen o están desactivadas"""
    if settings.ESQUINAS_POR_LADOS:
        try:
            contorno_lados = contorno
            alto, ancho = frame.shape[:2]
            angulo = angulo_del_dni(contorno)
            _, _, w, h = cv2.boundingRect(contorno.reshape(-1, 2))
            girado = settings.GIRO_SEGUNDA_PASADA > 0 and abs(angulo) >= settings.GIRO_SEGUNDA_PASADA
            pequeno = max(w, h) < settings.DNI_PEQUENO * max(ancho, alto)
            if girado or pequeno:
                giro = angulo if girado else 0.0
                # Una segunda pasada de cerca y, si el contorno nuevo, más fino, dice que el DNI sigue girado,
                # una tercera con el giro corregido
                for _ in range(2):
                    try:
                        cerca, confianza_cerca = segmentar_de_cerca(frame, contorno_lados, giro, segmentar,
                                                                    settings.IMG_SIZE, settings.MARGEN_SEGUNDA_PASADA)
                    except DeteccionFallida as e:
                        logger.info(f"📐 Pasada de cerca sin resultado ({e.code})")
                        break
                    if confianza_cerca < settings.MIN_CONFIDENCE_OUTPUT or not mismo_dni(contorno, cerca):
                        logger.info(f"📐 Pasada de cerca descartada ({confianza_cerca:.2%})")
                        break
                    contorno_lados = cerca
                    resto = angulo_del_dni(girar_puntos(cerca, giro))
                    logger.info(f"📐 Pasada de cerca con el DNI girado {giro:.1f}°, queda {resto:.1f}°")
                    if settings.GIRO_SEGUNDA_PASADA <= 0 or abs(resto) < settings.GIRO_SEGUNDA_PASADA:
                        break
                    giro += resto
            esquinas = esquinas_por_lados(contorno_lados, ancho, alto)
            if esquinas is None:
                logger.info("📐 Las rectas de los lados no dan un DNI, se usa el polígono")
            return esquinas
        except Exception:
            logger.exception("📐 Error al sacar las esquinas por los lados, se usa el polígono")
    return None


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
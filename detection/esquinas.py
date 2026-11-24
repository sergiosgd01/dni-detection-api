"""
Esquinas del DNI a partir del contorno que da el modelo (FLU-3)

Antes las esquinas salían de simplificar el contorno de la máscara hasta dejarlo en cuatro puntos. Con el DNI girado
la máscara se come la sombra de la tarjeta hasta el borde de su caja, que no está girada, y deja bloques rectos en las
esquinas. La simplificación cogía las esquinas de esos bloques y el recorte salía torcido (en B02, girado 15°, las
esquinas se iban un 13 y un 21 % de la diagonal, y un cambio pequeño en la foto cambiaba qué cara salía bien).

Lo principal se arregló quitando las manchas sueltas de la máscara (contorno_sin_manchas en yolo_processor.py), porque
la sombra llegaba en trozos aparte. Aquí están además unas esquinas más finas. Si el DNI está girado o lejos, su zona
se vuelve a pasar por el modelo, derecha y de cerca, y la máscara sale más fina y sin la sombra. Después se ajusta una
recta a la parte central de cada lado y las esquinas son los cruces de esas rectas, así no cuentan las esquinas
redondeadas del DNI ni un dedo que tape una esquina. yolo_processor.esquinas_del_dni solo las usa cuando se separan
mucho de las del polígono de siempre (ver allí por qué).
"""
import logging
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Parte de cada lado que se usa para ajustar su recta. Fuera queda el 15 % de cada punta, donde están las esquinas
# redondeadas (el radio del DNI es un 6 % de su alto) y lo que tape una esquina
TRAMO_CENTRAL = (0.15, 0.85)
# Un punto más lejos que esto del lado que le toca (en fracción de la diagonal) no cuenta para su recta
DISTANCIA_MAXIMA_AL_LADO = 0.08
# Puntos que necesita cada lado para ajustar su recta
PUNTOS_MINIMOS_POR_LADO = 10
# Ángulos que puede tener una esquina. Un DNI muy de lado en la foto se queda en unos 60 a 120 grados
ANGULO_ESQUINA = (55.0, 125.0)
# Proporción entre el área del cuadrilátero y la del contorno
PROPORCION_AREA = (0.8, 1.25)


def densificar(contorno: np.ndarray, paso: float) -> np.ndarray:
    """
    Puntos a lo largo del contorno cerrado, cada `paso` píxeles como mucho. El contorno del modelo solo guarda las
    puntas de los tramos rectos, y para ajustar la recta de un lado hacen falta puntos repartidos por todo él.
    """
    c = contorno.reshape(-1, 2).astype(np.float64)
    puntos = []
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / paso)))
        puntos.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.concatenate(puntos)


def angulo_del_dni(contorno: np.ndarray) -> float:
    """
    Giro del DNI en la foto en grados, entre -45 y 45, con el signo de arctan2 en la imagen (positivo si los lados
    largos bajan hacia la derecha). Sale del ángulo más repetido entre los tramos rectos del contorno, contando cada
    tramo por su largo. Se coge el pico y no la media, porque los bloques rectos que deja la máscara con el DNI girado
    tiran hacia 0 grados.
    """
    c = contorno.reshape(-1, 2).astype(np.float32)
    poligono = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float64)
    tramos = np.roll(poligono, -1, axis=0) - poligono
    largos = np.hypot(tramos[:, 0], tramos[:, 1])
    angulos = (np.degrees(np.arctan2(tramos[:, 1], tramos[:, 0])) + 45) % 90 - 45
    histograma = np.zeros(90)
    np.add.at(histograma, np.floor(angulos + 45).astype(int) % 90, largos)
    suavizado = sum(np.roll(histograma, k) for k in range(-2, 3))
    pico = float(np.argmax(suavizado)) - 45 + 0.5
    # Media de los tramos del pico, medida desde el pico para que no se rompa entre -45 y 45
    diferencia = (angulos - pico + 45) % 90 - 45
    cerca = np.abs(diferencia) <= 2.5
    if not cerca.any():
        return pico
    return float((pico + np.average(diferencia[cerca], weights=largos[cerca]) + 45) % 90 - 45)


def segmentar_de_cerca(frame: np.ndarray, contorno: np.ndarray, angulo: float,
                       segmentar: Callable[[np.ndarray], Tuple[np.ndarray, float]],
                       lado_modelo: int, margen: float) -> Tuple[np.ndarray, float]:
    """
    Vuelve a pasar por el modelo solo la zona del DNI, girada `angulo` grados para dejarlo derecho y con `margen` de
    fondo alrededor (fracción de su lado mayor, el modelo no reconoce el DNI si no ve fondo). Así el DNI ocupa casi
    toda la imagen del modelo y su máscara sale más fina. Devuelve el contorno en píxeles de la foto original y la
    confianza. Lo que cae fuera de la foto se pinta del color de sus bordes.
    """
    alto, ancho = frame.shape[:2]
    puntos = contorno.reshape(-1, 2).astype(np.float64)
    centro = (puntos.min(axis=0) + puntos.max(axis=0)) / 2
    giro = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), angulo, 1.0)
    girados = puntos @ giro[:, :2].T + giro[:, 2]
    minimo, maximo = girados.min(axis=0), girados.max(axis=0)
    borde = margen * float((maximo - minimo).max())
    minimo, maximo = minimo - borde, maximo + borde
    escala = min(1.5, lado_modelo / float((maximo - minimo).max()))
    # De la foto original a la imagen que ve el modelo: girar, llevar la esquina de la zona a (0, 0) y escalar
    matriz = giro.copy()
    matriz[:, 2] -= minimo
    matriz *= escala
    tam = (max(1, int(np.ceil((maximo - minimo)[0] * escala))), max(1, int(np.ceil((maximo - minimo)[1] * escala))))

    # Si hay que reducir, se reduce antes la foto entera con INTER_AREA, que no deja dientes de sierra
    imagen, matriz_imagen = frame, matriz
    if escala < 1.0:
        imagen = cv2.resize(frame, (max(1, round(ancho * escala)), max(1, round(alto * escala))),
                            interpolation=cv2.INTER_AREA)
        matriz_imagen = matriz.copy()
        matriz_imagen[:, 0] *= ancho / imagen.shape[1]
        matriz_imagen[:, 1] *= alto / imagen.shape[0]
    bordes = np.concatenate([imagen[0], imagen[-1], imagen[:, 0], imagen[:, -1]])
    color = tuple(int(v) for v in np.median(bordes, axis=0))
    de_cerca = cv2.warpAffine(imagen, matriz_imagen, tam, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=color)

    contorno_cerca, confianza = segmentar(de_cerca)

    inversa = cv2.invertAffineTransform(matriz)
    vuelta = np.asarray(contorno_cerca, np.float64) @ inversa[:, :2].T + inversa[:, 2]
    vuelta[:, 0] = np.clip(vuelta[:, 0], 0, ancho)
    vuelta[:, 1] = np.clip(vuelta[:, 1], 0, alto)
    return vuelta.astype(np.float32), confianza


def girar_puntos(puntos: np.ndarray, angulo: float) -> np.ndarray:
    """Los puntos girados `angulo` grados como gira cv2.getRotationMatrix2D, para medir el giro que les queda"""
    giro = cv2.getRotationMatrix2D((0.0, 0.0), angulo, 1.0)
    return (puntos.reshape(-1, 2).astype(np.float64) @ giro[:, :2].T).astype(np.float32)


def mismo_dni(contorno_a: np.ndarray, contorno_b: np.ndarray) -> bool:
    """Si dos contornos son del mismo DNI, por cómo se solapan sus cajas (para no saltar a otra cosa de la foto)"""
    xa, ya, wa, ha = cv2.boundingRect(contorno_a.reshape(-1, 2).astype(np.float32))
    xb, yb, wb, hb = cv2.boundingRect(contorno_b.reshape(-1, 2).astype(np.float32))
    ancho = max(0, min(xa + wa, xb + wb) - max(xa, xb))
    alto = max(0, min(ya + ha, yb + hb) - max(ya, yb))
    comun = ancho * alto
    return comun / max(wa * ha + wb * hb - comun, 1) >= 0.7


def _recta(puntos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recta por mínimos cuadrados que deja fuera los puntos lejanos (más de tres desviaciones, medidas con la mediana).
    La máscara del modelo es más basta que la foto y sus lados son escalones de varios píxeles. Mínimos cuadrados
    sigue la inclinación media de los escalones, mientras que un ajuste robusto con umbral fijo en píxeles (Huber) se
    quedaba con el escalón más largo y torcía la recta.
    """
    usados = puntos
    for _ in range(3):
        vx, vy, x0, y0 = cv2.fitLine(usados.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        normal = np.array([-vy, vx], np.float64)
        distancia = np.abs((puntos - np.array([x0, y0], np.float64)) @ normal)
        limite = max(3 * 1.4826 * float(np.median(distancia)), 1e-6)
        nuevos = puntos[distancia <= limite]
        if len(nuevos) < PUNTOS_MINIMOS_POR_LADO or len(nuevos) == len(usados):
            break
        usados = nuevos
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def _cruce(recta_a, recta_b) -> Optional[np.ndarray]:
    (p, v), (q, w) = recta_a, recta_b
    sistema = np.array([v, -w]).T
    if abs(np.linalg.det(sistema)) < 1e-6:
        return None
    s, _ = np.linalg.solve(sistema, q - p)
    return p + s * v


def esquinas_por_lados(contorno: np.ndarray, ancho: int, alto: int) -> Optional[np.ndarray]:
    """
    Cuatro esquinas como cruces de las rectas de los lados, en píxeles de la foto. None si el resultado no parece un
    DNI (lados con pocos puntos, esquinas muy abiertas o cerradas, o un área muy distinta de la del contorno).
    """
    c = contorno.reshape(-1, 2).astype(np.float32)
    if len(c) < 4:
        return None
    puntos = densificar(c, max(1.0, cv2.arcLength(c, True) / 800))
    # Los puntos pegados al borde de la foto son el corte de la foto, no un lado del DNI
    margen = 2.0
    dentro = ((puntos[:, 0] > margen) & (puntos[:, 0] < ancho - 1 - margen)
              & (puntos[:, 1] > margen) & (puntos[:, 1] < alto - 1 - margen))
    cuadrilatero = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
    diagonal = max(np.linalg.norm(cuadrilatero[0] - cuadrilatero[2]), 1.0)

    # Cada punto va al lado más cercano y se ajustan las rectas. Se repite para que, partiendo del rectángulo que
    # envuelve el contorno, los lados se vayan ajustando a un DNI visto en perspectiva
    for _ in range(3):
        inicio = cuadrilatero
        fin = np.roll(cuadrilatero, -1, axis=0)
        largo = np.maximum(np.linalg.norm(fin - inicio, axis=1), 1e-6)
        direccion = (fin - inicio) / largo[:, None]
        normal = np.stack([-direccion[:, 1], direccion[:, 0]], axis=1)
        relativo = puntos[:, None, :] - inicio[None]
        t = (relativo * direccion[None]).sum(-1) / largo[None]
        distancia = np.abs((relativo * normal[None]).sum(-1))
        fuera = (np.clip(-t, 0, None) + np.clip(t - 1, 0, None)) * largo[None]
        lado = np.argmin(np.hypot(distancia, fuera), axis=1)
        rectas = []
        for i in range(4):
            elegidos = ((lado == i) & dentro & (t[:, i] > TRAMO_CENTRAL[0]) & (t[:, i] < TRAMO_CENTRAL[1])
                        & (distancia[:, i] < DISTANCIA_MAXIMA_AL_LADO * diagonal))
            if elegidos.sum() < PUNTOS_MINIMOS_POR_LADO:
                return None
            rectas.append(_recta(puntos[elegidos]))
        nuevo = [_cruce(rectas[i - 1], rectas[i]) for i in range(4)]
        if any(p is None for p in nuevo):
            return None
        cuadrilatero = np.array(nuevo)

    if not cv2.isContourConvex(cuadrilatero.astype(np.float32)):
        return None
    for i in range(4):
        a = cuadrilatero[i - 1] - cuadrilatero[i]
        b = cuadrilatero[(i + 1) % 4] - cuadrilatero[i]
        coseno = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
        if not ANGULO_ESQUINA[0] <= np.degrees(np.arccos(np.clip(coseno, -1, 1))) <= ANGULO_ESQUINA[1]:
            return None
    proporcion = cv2.contourArea(cuadrilatero.astype(np.float32)) / max(cv2.contourArea(c), 1.0)
    if not PROPORCION_AREA[0] <= proporcion <= PROPORCION_AREA[1]:
        return None
    # Una esquina fuera de la foto (DNI cortado) se deja en el borde, como antes
    cuadrilatero[:, 0] = np.clip(cuadrilatero[:, 0], 0, ancho - 1)
    cuadrilatero[:, 1] = np.clip(cuadrilatero[:, 1], 0, alto - 1)
    return cuadrilatero.astype(np.float32)


def separacion_esquinas(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cuánto se separan dos juegos de cuatro esquinas, en fracción de la diagonal del segundo. Es la mayor distancia
    entre esquinas emparejadas, probando los cuatro giros y los dos sentidos, porque cada método las da en su orden.
    """
    a = np.asarray(a, np.float64).reshape(4, 2)
    b = np.asarray(b, np.float64).reshape(4, 2)
    diagonal = max(np.linalg.norm(b[0] - b[2]), np.linalg.norm(b[1] - b[3]), 1.0)
    mejor = min(np.linalg.norm(a - np.roll(orden, k, axis=0), axis=1).max()
                for orden in (b, b[::-1]) for k in range(4))
    return float(mejor / diagonal)


def esquinas_por_poligono(contorno: np.ndarray) -> np.ndarray:
    """Lo de antes de FLU-3. Simplifica el contorno hasta cuatro puntos y, si no lo consigue, usa el rectángulo girado"""
    contour = np.asarray(contorno).reshape(-1, 2).astype(np.int32)
    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        approx = cv2.approxPolyDP(contour, epsilon_factor * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return np.int32(cv2.boxPoints(cv2.minAreaRect(contour))).astype(np.float32)

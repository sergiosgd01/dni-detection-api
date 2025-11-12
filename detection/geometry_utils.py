import cv2
import numpy as np

def ordenar_puntos(pts):
    """Ordena 4 puntos como: TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # TR
    rect[3] = pts[np.argmax(d)]  # BL
    return rect

def corregir_perspectiva(img, mask):
    """Corrige la perspectiva del DNI usando la máscara segmentada."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]

    pts = approx.reshape(4, 2)
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

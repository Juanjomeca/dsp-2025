import cv2
import numpy as np


def split_quadtree_custom(img, condition_fn, min_size=8):
    """
    Divide la imagen en bloques usando quadtree según una condición booleana.

    img: imagen en escala de grises (np.uint8)
    condition_fn: función que recibe un bloque y devuelve True/False
    min_size: tamaño mínimo de bloque para dejar de dividir
    """
    h, w = img.shape
    blocks = []

    def split(x, y, w, h):
        block = img[y:y+h, x:x+w]
        if w <= min_size or h <= min_size or condition_fn(block):
            if condition_fn(block):
                blocks.append((x, y, w, h))
        else:
            hw, hh = w//2, h//2
            split(x, y, hw, hh)
            split(x+hw, y, w-hw, hh)
            split(x, y+hh, hw, h-hh)
            split(x+hw, y+hh, w-hw, h-hh)

    split(0, 0, w, h)
    return blocks


def draw_blocks(img, blocks, color=(0, 0, 255)):
    """Dibuja los bloques en la imagen"""
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in blocks:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 1)
    return overlay


img = cv2.imread('lab_images/frutas.jpg', 0)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen.")

blur = cv2.GaussianBlur(img, (3, 3), 0)


def is_dark(block, dark_threshold=50):
    return np.mean(block) <= dark_threshold

def is_between(block, low=100, high=120):
    mean_intensity = np.mean(block)
    return low <= mean_intensity <= high


blocks = split_quadtree_custom(
    blur, lambda blk: is_between(blk), min_size=16)
result = draw_blocks(blur, blocks)

cv2.imshow("Original", img)
cv2.imshow("Quadtree Custom Regions", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

def resize_image(image, width=500):
    """Resmi bozmadan belirli bir genişliğe boyutlandırır."""
    h, w = image.shape[:2]
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def show_results_side_by_side(ref, test, title="Comparison"):
    """İki resmi yan yana birleştirip gösterir."""
    combined = np.hstack((ref, test))
    cv2.imshow(title, combined)
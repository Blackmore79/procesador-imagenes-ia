from PIL import Image
import numpy as np

try:
    from rembg import remove
except ImportError:
    remove = None

try:
    import cv2
except ImportError:
    cv2 = None

def get_subject_mask(image, logger=None):
    if remove is not None:
        if logger:
            logger.debug("Usando rembg para segmentación.")
        mask = remove(image)
        # rembg devuelve una imagen RGBA, la máscara es el canal A
        return Image.fromarray(np.array(mask)[:,:,3]).convert("L")
    elif cv2 is not None:
        if logger:
            logger.debug("Usando cv2.saliency para saliencia.")
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(opencv_image)
        mask = (saliencyMap * 255).astype("uint8")
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        return Image.fromarray(mask).convert("L")
    else:
        if logger:
            logger.warning("No hay método de segmentación disponible.")
        # fallback: máscara blanca (todo es sujeto)
        return Image.new("L", image.size, 255)
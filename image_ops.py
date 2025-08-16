from PIL import Image, ImageFilter, ImageDraw
import os
import numpy as np
from real_esrgan import run_realesrgan
from subject_detection import get_subject_mask

def should_crop(mask, output_size, logger=None):
    mask_np = np.array(mask)
    coords = np.column_stack(np.where(mask_np > 128))
    if coords.size == 0:
        if logger:
            logger.warning("No se detectó sujeto principal, se usará letterbox.")
        return False
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    subject_bbox = [x1, y1, x2, y2]
    iw, ih = mask_np.shape[1], mask_np.shape[0]
    ow, oh = output_size
    aspect = ow / oh
    crop_w = min(iw, int(ih * aspect))
    crop_h = min(ih, int(iw / aspect))
    crop_x1 = (iw - crop_w) // 2
    crop_y1 = (ih - crop_h) // 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    [sx1, sy1, sx2, sy2] = subject_bbox
    inside = (sx1 >= crop_x1 and sx2 <= crop_x2 and sy1 >= crop_y1 and sy2 <= crop_y2)
    if logger:
        logger.debug(f"Crop: {inside}, bbox sujeto: {subject_bbox}, crop: {(crop_x1, crop_y1, crop_x2, crop_y2)}")
    return inside

def crop_center(image, output_size):
    iw, ih = image.size
    ow, oh = output_size
    aspect = ow / oh
    if iw / ih > aspect:
        new_w = int(ih * aspect)
        left = (iw - new_w) // 2
        box = (left, 0, left+new_w, ih)
    else:
        new_h = int(iw / aspect)
        top = (ih - new_h) // 2
        box = (0, top, iw, top+new_h)
    return image.crop(box).resize(output_size, Image.LANCZOS)

def letterbox_blur(image, output_size):
    ow, oh = output_size
    iw, ih = image.size
    scale = min(ow/iw, oh/ih)
    resized = image.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)

    # Obtener color promedio de los bordes izquierdo y derecho
    left_crop = image.crop((0, 0, 1, ih)).resize((100, 100))
    right_crop = image.crop((iw-1, 0, iw, ih)).resize((100, 100))
    left_color = left_crop.resize((1,1)).getpixel((0,0))
    right_color = right_crop.resize((1,1)).getpixel((0,0))

    # Crear fondo degradado
    bg = Image.new("RGB", output_size)
    draw = ImageDraw.Draw(bg)
    for x in range(ow):
        factor = x / ow
        color = tuple([
            int(left_color[i] * (1 - factor) + right_color[i] * factor)
            for i in range(3)
        ])
        draw.line([(x,0), (x,oh)], fill=color)

    # Pegar la imagen centrada
    bg.paste(resized, ((ow-resized.width)//2, (oh-resized.height)//2))
    return bg

def process_image(input_path, output_path, upscale=True, target_size=(3440,1440), logger=None):
    temp_path = input_path
    if upscale:
        temp_path = input_path + ".up.png"
        run_realesrgan(input_path, temp_path, logger)
    image = Image.open(temp_path).convert("RGB")
    mask = get_subject_mask(image, logger)
    if should_crop(mask, target_size, logger):
        result = crop_center(image, target_size)
        if logger:
            logger.info(f"Imagen recortada (crop) a {output_path}")
    else:
        result = letterbox_blur(image, target_size)
        if logger:
            logger.info(f"Imagen letterbox a {output_path}")
    result.save(output_path)
    if upscale and os.path.exists(temp_path) and temp_path != input_path:
        os.remove(temp_path)

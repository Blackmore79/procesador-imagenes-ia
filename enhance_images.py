import argparse
import os
import subprocess
from tqdm import tqdm
from PIL import Image, ImageFilter
import cv2
import numpy as np

try:
    from rembg import remove
except ImportError:
    remove = None

# --- Configuración por defecto ---
RES_W = 3440
RES_H = 1440
ASPECT = RES_W / RES_H

# --- Función para correr Real-ESRGAN ---
def run_realesrgan(input_path, output_path):
    # Asume que realesrgan-ncnn-vulkan está instalado y en PATH
    command = [
        "realesrgan-ncnn-vulkan", 
        "-i", input_path, 
        "-o", output_path
    ]
    subprocess.run(command, check=True)

# --- Detección del sujeto principal ---
def get_subject_mask(image):
    if remove is not None:
        # Usa rembg si está disponible
        return remove(image)
    # Si no, usa cv2.saliency para obtener un mapa de saliencia aproximado
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(opencv_image)
    mask = (saliencyMap * 255).astype("uint8")
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask).convert("L")

# --- Decide si crop o letterbox ---
def should_crop(mask, output_size):
    # Encuentra el bbox del sujeto principal
    mask_np = np.array(mask)
    coords = cv2.findNonZero((mask_np > 128).astype(np.uint8))
    if coords is None:
        return False
    x, y, w, h = cv2.boundingRect(coords)
    # Calcula el bbox propuesto para crop 21:9
    aspect = output_size[0] / output_size[1]
    ih, iw = mask_np.shape
    crop_w = min(iw, int(ih * aspect))
    crop_h = min(ih, int(iw / aspect))
    # Chequea si el bbox del sujeto principal cabe dentro del crop central
    crop_x1 = (iw - crop_w) // 2
    crop_y1 = (ih - crop_h) // 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    subject_inside = (x >= crop_x1 and x+w <= crop_x2 and y >= crop_y1 and y+h <= crop_y2)
    return subject_inside

# --- Crop central ---
def crop_center(image, output_size):
    iw, ih = image.size
    ow, oh = output_size
    aspect = ow / oh
    if iw / ih > aspect:
        # recortar ancho
        new_w = int(ih * aspect)
        left = (iw - new_w) // 2
        box = (left, 0, left+new_w, ih)
    else:
        # recortar alto
        new_h = int(iw / aspect)
        top = (ih - new_h) // 2
        box = (0, top, iw, top+new_h)
    return image.crop(box).resize(output_size, Image.LANCZOS)

# --- Letterbox con fondo difuminado ---
def letterbox_blur(image, output_size):
    ow, oh = output_size
    # Redimensionar manteniendo aspecto
    iw, ih = image.size
    scale = min(ow/iw, oh/ih)
    resized = image.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
    # Crear fondo difuminado
    bg = image.resize(output_size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(32))
    # Pegar imagen centrada
    bg.paste(resized, ((ow-resized.width)//2, (oh-resized.height)//2))
    return bg

# --- Letterbox con inpainting ---
def letterbox_inpaint(image, output_size):
    ow, oh = output_size
    iw, ih = image.size
    scale = min(ow/iw, oh/ih)
    new_w, new_h = int(iw*scale), int(ih*scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Crear imagen base y máscara para inpainting
    base = Image.new("RGB", output_size, (0,0,0))
    base.paste(resized, ((ow-new_w)//2, (oh-new_h)//2))
    
    # Crear máscara: 0 donde hay imagen, 255 donde hay que rellenar
    mask = Image.new("L", output_size, 255)
    mask.paste(0, ((ow-new_w)//2, (oh-new_h)//2, (ow+new_w)//2, (oh+new_h)//2))
    
    # Convertir a numpy
    base_np = np.array(base)
    mask_np = np.array(mask)
    
    # Inpainting con OpenCV
    inpainted = cv2.inpaint(base_np, mask_np, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)

# --- Proceso principal ---
def process_image(input_path, output_path, upscale=True, target_size=(RES_W, RES_H)):
    # 1. Upscale con Real-ESRGAN si corresponde
    temp_path = input_path
    if upscale:
        temp_path = input_path + ".up.png"
        run_realesrgan(input_path, temp_path)

    # 2. Leer imagen
    image = Image.open(temp_path).convert("RGB")

    # 3. Detectar sujeto principal
    mask = get_subject_mask(image)

    # 4. Decidir crop o letterbox
    if should_crop(mask, target_size):
        result = crop_center(image, target_size)
    else:
        result = letterbox_inpaint(image, target_size)

    # 5. Guardar salida
    result.save(output_path)

    # 6. Limpiar temporal si fue necesario
    if upscale and os.path.exists(temp_path) and temp_path != input_path:
        os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Upscale y recorta imágenes para fondo 21:9 (3440x1440)")
    parser.add_argument("origen", type=str, help="Carpeta de origen")
    parser.add_argument("destino", type=str, help="Carpeta de destino")
    parser.add_argument("--no-upscale", action="store_true", help="No usar Real-ESRGAN")
    parser.add_argument("--ancho", type=int, default=RES_W, help="Ancho de salida")
    parser.add_argument("--alto", type=int, default=RES_H, help="Alto de salida")
    args = parser.parse_args()

    os.makedirs(args.destino, exist_ok=True)
    archivos = [f for f in os.listdir(args.origen) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]

    for archivo in tqdm(archivos, desc="Procesando imágenes"):
        input_path = os.path.join(args.origen, archivo)
        output_path = os.path.join(args.destino, archivo)
        process_image(
            input_path, 
            output_path, 
            upscale=not args.no_upscale, 
            target_size=(args.ancho, args.alto)
        )

if __name__ == "__main__":
    main()
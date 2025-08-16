import argparse
import os
from tqdm import tqdm
from image_ops import process_image
from logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="Upscale y recorta imágenes para fondo 21:9 (3440x1440)")
    parser.add_argument("origen", type=str, help="Carpeta de origen")
    parser.add_argument("destino", type=str, help="Carpeta de destino")
    parser.add_argument("--no-upscale", action="store_true", help="No usar Real-ESRGAN")
    parser.add_argument("--ancho", type=int, default=3440, help="Ancho de salida")
    parser.add_argument("--alto", type=int, default=1440, help="Alto de salida")
    parser.add_argument("--log", type=str, default="info", help="Nivel de logging: debug|info|warning|error")
    args = parser.parse_args()

    logger = get_logger(args.log)
    os.makedirs(args.destino, exist_ok=True)
    archivos = [f for f in os.listdir(args.origen) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]

    logger.info(f"Iniciando procesamiento de {len(archivos)} imágenes.")
    for archivo in tqdm(archivos, desc="Procesando imágenes"):
        input_path = os.path.join(args.origen, archivo)
        output_path = os.path.join(args.destino, archivo)
        try:
            process_image(
                input_path,
                output_path,
                upscale=not args.no_upscale,
                target_size=(args.ancho, args.alto),
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error en {archivo}: {e}")

    logger.info("Procesamiento finalizado.")

if __name__ == "__main__":
    main()
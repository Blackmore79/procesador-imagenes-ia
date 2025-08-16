import argparse
import os
import shutil

def procesar_imagenes(origen, destino):
    if not os.path.exists(destino):
        os.makedirs(destino)
    for archivo in os.listdir(origen):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            origen_path = os.path.join(origen, archivo)
            destino_path = os.path.join(destino, archivo)
            shutil.copy2(origen_path, destino_path)
            print(f"Copiada: {archivo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa imágenes de una carpeta a otra.")
    parser.add_argument("origen", type=str, help="Carpeta de origen")
    parser.add_argument("destino", type=str, help="Carpeta de destino")
    args = parser.parse_args()

    procesar_imagenes(args.origen, args.destino)
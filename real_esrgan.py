import subprocess
import os

def run_realesrgan(input_path, output_path, logger=None):
    command = [
        "realesrgan-ncnn-vulkan",
        "-i", input_path,
        "-o", output_path
    ]
    if logger:
        logger.debug(f"Ejecutando: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        if logger:
            logger.error(f"Real-ESRGAN falló: {result.stderr.decode()}")
        raise RuntimeError("Fallo Real-ESRGAN")
    if logger:
        logger.info(f"Upscaling completado: {output_path}")
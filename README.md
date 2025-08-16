# Image CLI 21:9

## Descripción
Procesa imágenes de una carpeta, aumentando la resolución con Real-ESRGAN y adaptando el encuadre a formato ultrawide (21:9, 3440x1440) con crop inteligente o letterbox difuminado según detección automática del sujeto principal.

## Instalación

1. Instala Python 3.9+.
2. Instala dependencias:

   ```
   pip install -r requirements.txt
   ```

3. Descarga o compila el binario de [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) y asegúrate de tenerlo en tu PATH.

## Uso

```bash
python main.py carpeta_entrada carpeta_salida
```

Opciones:

- `--no-upscale` : No usar Real-ESRGAN.
- `--ancho N`    : Cambia el ancho de salida (default 3440).
- `--alto N`     : Cambia el alto de salida (default 1440).
- `--log nivel`  : Cambia el nivel de log (debug, info, warning, error).

## Lógica

1. **Aumenta resolución** con Real-ESRGAN (si no usas --no-upscale).
2. **Detecta sujeto principal** usando rembg o cv2.saliency.
3. Si el sujeto cabe en crop 21:9, recorta.
4. Si no, aplica letterbox con fondo difuminado.
5. Guarda en la carpeta de destino.

## Logging

El log muestra información sobre cada imagen procesada, avisos y errores. Puedes ajustar el nivel con `--log debug` para ver más detalles.

## Notas

- Si tienes rembg instalado, se usará para segmentación precisa. Si no, se usa cv2.saliency.
- Si falta cualquier dependencia, el script intenta continuar con métodos alternativos y lo avisa en el log.
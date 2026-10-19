from ultralytics import YOLO

def iniciar_entrenamiento():
    # 1. Cargar el modelo base. Usamos la versión "nano" (yolov8n.pt) 
    # porque es la más rápida para procesar en una computadora estándar.
    modelo = YOLO('yolov8n.pt')

    # 2. Definir la ruta de tu archivo de configuración (reemplaza con tu ruta exacta si es diferente)
    ruta_yaml = 'datasets/daños_auto/data.yaml'

    print("Iniciando el entrenamiento...")
    
    # 3. Ejecutar el entrenamiento
    # epochs: Cuántas veces el modelo repasará todas las fotos. 50 es un buen inicio.
    # imgsz: Resolución a la que se redimensionan las imágenes (640 píxeles es el estándar de YOLO).
    # plots: Genera gráficos automáticos para evaluar qué tan bien está aprendiendo.
    resultados = modelo.train(
        data=ruta_yaml,
        epochs=50,
        imgsz=640,
        plots=True,
        device='0' # Fuerza a usar el procesador. Si tienes tarjeta NVIDIA configurada, cambia 'cpu' por 0.
    )

if __name__ == '__main__':
    # Esta condición es obligatoria en Windows para evitar errores de multiprocesamiento al entrenar
    iniciar_entrenamiento()
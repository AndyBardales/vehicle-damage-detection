from ultralytics import YOLO

def continuar_entrenamiento():
    # Cargar el último punto de control guardado (last.pt)
    # No cargues 'yolov8n.pt', debe ser el de la carpeta 'runs'
    model = YOLO('runs/detect/train/weights/last.pt')

    # Reanudar el entrenamiento
    # El parámetro resume=True le indica al modelo que busque 
    # en qué época se quedó y continúe desde ahí.
    model.train(resume=True)

if __name__ == '__main__':   
    continuar_entrenamiento()
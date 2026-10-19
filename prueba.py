import cv2
from ultralytics import YOLO

print("¡Las librerías se importaron correctamente!")
print(f"Versión de OpenCV: {cv2.__version__}")

# Descarga un modelo base muy pequeño para comprobar que YOLO funciona
modelo = YOLO('yolov8n.pt')
print("¡El modelo YOLO base se cargó con éxito!")
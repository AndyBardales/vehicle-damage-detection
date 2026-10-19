import cv2
from ultralytics import YOLO

# 1. Cargar tu modelo
modelo = YOLO('modelo_danos.pt')

# 2. Cargar el video (puedes cambiar la ruta por 0 si quieres usar tu cámara web)
video_path = 'pruebas/video_prueba.mp4'
cap = cv2.VideoCapture(video_path)

print("Iniciando análisis de video... Presiona 'q' en tu teclado para salir.")

while cap.isOpened():
    exito, frame = cap.read()
    
    if exito:
        # Analizar el frame actual
        resultados = modelo(frame, conf=0.15)
        
        # Dibujar los cuadros en el frame
        frame_anotado = resultados[0].plot()
        
        # Mostrar el video con las detecciones
        cv2.imshow("Deteccion de Danos (Video)", frame_anotado)
        
        # Romper el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # El video terminó
        break

# Liberar los recursos de tu computadora
cap.release()
cv2.destroyAllWindows()
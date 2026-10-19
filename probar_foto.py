import cv2
from ultralytics import YOLO

# 1. Cargar tu modelo personalizado
modelo = YOLO('modelo_danos.pt')

# 2. Definir la ruta de tu foto de prueba
ruta_foto = 'pruebas/xd.jpg'

print("Analizando fotografía...")

# 3. Realizar la predicción
# conf=0.5 significa que solo mostrará detecciones de las que esté al menos 50% seguro
resultados = modelo(ruta_foto, conf=0.15)

# 4. Dibujar y mostrar los resultados
for r in resultados:
    # r.plot() toma la foto original y le dibuja los cuadros y porcentajes encima
    imagen_anotada = r.plot()
    
    # Mostrar la imagen en una ventana
    cv2.imshow('Deteccion de Danos (Foto)', imagen_anotada)
    
    # El programa se pausa aqui hasta que presiones cualquier tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()
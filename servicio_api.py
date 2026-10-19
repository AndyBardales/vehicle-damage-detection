import os
import tempfile
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Configuración
MODELO_PATH = 'modelo_danos.pt'       # Ruta a tu modelo entrenado
TAMANO_MAXIMO_MB = 50                 # Límite de tamaño en MB
EXTENSIONES_PERMITIDAS = {'jpg'}   # Formatos de imagen aceptados

# Cargar el modelo (una sola vez al iniciar el servidor)
modelo = YOLO(MODELO_PATH)

def extension_permitida(nombre_archivo):
    """Verifica si el archivo tiene una extensión de imagen permitida."""
    return '.' in nombre_archivo and \
        nombre_archivo.rsplit('.', 1)[1].lower() in EXTENSIONES_PERMITIDAS

@app.route('/detectar', methods=['POST'])
def detectar():
    # 1. Verificar que el campo 'imagen' existe
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se encontró el campo "imagen"'}), 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    # 2. Validar extensión
    if not extension_permitida(archivo.filename):
        return jsonify({'error': f'Formato no permitido. Use: {", ".join(EXTENSIONES_PERMITIDAS)}'}), 400

    # 3. Validar tamaño (máximo 50 MB)
    archivo.seek(0, os.SEEK_END)
    tamanio_bytes = archivo.tell()
    archivo.seek(0)  # volver al inicio para poder leerlo después
    if tamanio_bytes > TAMANO_MAXIMO_MB * 1024 * 1024:
        return jsonify({'error': f'El archivo excede el tamaño máximo de {TAMANO_MAXIMO_MB} MB'}), 400

    # 4. Guardar imagen temporal con la extensión original
    extension = archivo.filename.rsplit('.', 1)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=f'.{extension}', delete=False) as tmp:
        archivo.save(tmp.name)
        tmp_path = tmp.name

    try:
        # 5. Realizar la inferencia
        resultados = modelo(tmp_path, conf=0.15)   # umbral de confianza
    except Exception as e:
        return jsonify({'error': f'Error durante la inferencia: {str(e)}'}), 500
    finally:
        # 6. Eliminar el archivo temporal
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 7. Procesar los resultados y devolver JSON
    detecciones = []
    for r in resultados:
        boxes = r.boxes
        if boxes is not None:
            img_width = r.orig_shape[1]
            img_height = r.orig_shape[0]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()           # coordenadas absolutas
                rel_x1 = (x1 / img_width) * 100
                rel_y1 = (y1 / img_height) * 100
                rel_x2 = (x2 / img_width) * 100
                rel_y2 = (y2 / img_height) * 100
                centro_x = (rel_x1 + rel_x2) / 2
                centro_y = (rel_y1 + rel_y2) / 2

                clase = int(box.cls[0])
                tipo = "RAYADO" if clase == 0 else "ABOLLADO"   # supone 0=RAYADO, 1=ABOLLADO
                confianza = float(box.conf[0])

                detecciones.append({
                    "tipo": tipo,
                    "pos_x": round(centro_x, 2),
                    "pos_y": round(centro_y, 2),
                    "confianza": confianza
                })

    return jsonify(detecciones)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
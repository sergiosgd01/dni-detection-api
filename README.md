# DNI Detection API 🪪

API para detección y recorte automático de DNI usando YOLO11 y FastAPI.

## 🎯 Dos versiones disponibles

| Versión | Archivo | Descripción |
|---------|---------|-------------|
| **Original** | `main.py` | Versión monolítica (todo en un archivo) |
| **Modular** ⭐ | `main_modular.py` | Versión modularizada (recomendada) |

> 💡 **Recomendación**: Usa `main_modular.py` para nuevo desarrollo.

## 🚀 Características

- ✅ Detección automática de DNI en imágenes
- ✅ Corrección de perspectiva
- ✅ Validación de confianza (80% mínimo)
- ✅ API REST con FastAPI
- ✅ Manejo robusto de errores

## 📁 Estructura del Proyecto

```
dni-detection-api/
├── main.py                 # Aplicación FastAPI principal
├── detection/
│   ├── model_utils.py     # Carga del modelo YOLO
│   └── geometry_utils.py  # Corrección de perspectiva
├── model/
│   └── model_trained_victor_yolo11n.pt  # Modelo entrenado
├── requirements.txt        # Dependencias
└── render.yaml            # Configuración de Render
```

## 🛠️ Instalación Local

### 1. Clonar el repositorio
```bash
git clone <tu-repo>
cd dni-detection-api
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

## 📡 Endpoints

### `GET /`
Información general de la API
```json
{
  "service": "DNI Segmentation API",
  "version": "1.0.0",
  "status": "running"
}
```

### `GET /health`
Estado del servicio
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /process`
Procesa una imagen de DNI

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (imagen JPG, PNG, WEBP)

**Response exitosa:**
- Status: `200 OK`
- Content-Type: `image/jpeg`
- Header: `x-confidence: 0.95`
- Header: `x-esquinas: {"ancho":3000,"alto":1835,"puntos":[[12.0,8.5],[2990.0,10.0],[2988.5,1830.0],[10.0,1828.0]]}`. Las cuatro esquinas del recorte en píxeles de la foto enviada, en orden superior izquierda, superior derecha, inferior derecha e inferior izquierda. La web las usa para abrir el recorte manual ya colocado

**Errores controlados:**

1. **No se detectó DNI** (422)
```json
{
  "error": "no_detection",
  "message": "No se detectó ningún DNI en la imagen",
  "suggestion": "Asegúrate de que el DNI esté visible..."
}
```

2. **Confianza baja** (422)
```json
{
  "error": "low_confidence",
  "message": "La imagen del DNI no es suficientemente clara",
  "confidence": 0.65,
  "min_required": 0.80
}
```

### `POST /process-debug`
Versión de debug que devuelve JSON con información detallada

## 🌐 Despliegue en Render

### Opción 1: Desde GitHub (Recomendado)

1. **Sube el código a GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <tu-repo-url>
git push -u origin main
```

2. **En Render:**
   - Ve a [render.com](https://render.com)
   - Click en "New +" → "Web Service"
   - Conecta tu repositorio de GitHub
   - Render detectará automáticamente el `render.yaml`
   - Click en "Apply"

### Opción 2: Blueprint (render.yaml)

El archivo `render.yaml` incluido configura automáticamente:
- Runtime: Python 3.11
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## ⚙️ Configuración

### Variables de entorno (opcional)
Puedes añadir en Render:
- `CONFIDENCE_THRESHOLD`: Umbral de confianza mínimo (default: 0.80)
- `MAX_FILE_SIZE`: Tamaño máximo de archivo en MB (default: 10)

## 📦 Dependencias Principales

- **FastAPI**: Framework web
- **Ultralytics**: YOLO11 para detección
- **OpenCV**: Procesamiento de imágenes
- **PyTorch**: Backend de YOLO

## 🧪 Pruebas

### Con curl:
```bash
curl -X POST http://localhost:8000/process \
  -F "file=@dni.jpg" \
  -o resultado.jpg
```

### Con Python:
```python
import requests

url = "http://localhost:8000/process"
files = {"file": open("dni.jpg", "rb")}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open("dni_procesado.jpg", "wb") as f:
        f.write(response.content)
    print(f"Confianza: {response.headers.get('x-confidence')}")
else:
    print(response.json())
```

## 📝 Notas

- El modelo debe estar en `model/model_trained_victor_yolo11n.pt`
- Formatos soportados: JPG, JPEG, PNG, WEBP
- Tamaño máximo: 10MB por imagen
- La API requiere confianza ≥ 80% para devolver la imagen procesada

## 🔧 Troubleshooting

### Error: "Modelo no encontrado"
Verifica que el archivo del modelo esté en `model/model_trained_victor_yolo11n.pt`

### Error: "No se detectó ningún DNI"
- Mejora la iluminación
- Asegúrate de que el DNI esté completamente visible
- Evita reflejos y sombras

### Error: "Confianza insuficiente"
- Mantén la cámara estable
- Acerca más el DNI
- Mejora el enfoque

## 📄 Licencia

Este proyecto es privado y de uso interno.
# dni-detection-api

from ultralytics import YOLO

def cargar_modelo(model_path: str):
    """Carga el modelo YOLO de segmentación."""
    model = YOLO(model_path)
    print("✅ Modelo YOLO cargado correctamente")
    return model

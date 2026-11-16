"""
Configuración centralizada de la aplicación
"""
import os
from typing import Set


class Settings:
    """Configuración de la aplicación DNI Detection API"""
    
    # Modelo
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/model_trained_yolov8-seg_sergio.pt")
    
    # CORS. Solo las webs propias pueden llamar a la API desde el navegador
    DEV_ORIGINS: list[str] = [
        "http://localhost:5173",
        "https://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
    ]

    PRODUCTION_ORIGINS: list[str] = [
        "https://proyectodni.netlify.app",
        "https://defiendetuidentidad.es",
        "https://www.defiendetuidentidad.es",
        "https://defiendetuidentidad.com",
        "https://www.defiendetuidentidad.com",
        "https://protegetuidentidad.es",
        "https://www.protegetuidentidad.es",
    ]
    
    # YOLO - Parámetros de detección
    CONFIDENCE_THRESHOLD: float = 0.4
    MIN_CONFIDENCE_OUTPUT: float = 0.80  # Confianza mínima para devolver resultado
    MASK_THRESHOLD: float = 0.5
    IMG_SIZE: int = 640
    # Segundo intento para escaneos y fotos recortadas al borde del DNI. El modelo no encuentra la tarjeta si no ve
    # fondo alrededor, así que, si falla, se repite con este margen (fracción del lado mayor) de color de mesa de madera.
    # Con 0 no se reintenta.
    MARGEN_REINTENTO: float = float(os.getenv("MARGEN_REINTENTO", 0.25))
    COLOR_MARGEN_REINTENTO: tuple[int, int, int] = (52, 88, 128)  # BGR
    # El segundo intento solo vale si el DNI ocupa al menos esta fracción de la foto. Si ocupa menos, ya había fondo
    # alrededor y el primer intento falló por otra causa, así que se deja el error de antes.
    OCUPACION_MINIMA_REINTENTO: float = 0.5
    
    # Archivos
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.webp'}
    JPEG_QUALITY: int = 90
    # Lado mayor del DNI recortado. 2000 px son unos 590 ppp en un DNI de 85,6 mm, de sobra para leerlo e imprimirlo.
    # Las fotos con menos resolución se devuelven tal cual, sin ampliar.
    MAX_LADO_SALIDA: int = int(os.getenv("MAX_LADO_SALIDA", 2000))

    # Protección de /process. Los valores se pueden cambiar con variables de entorno sin tocar el código (ver
    # DEPLOY.md). Las cuentas se llevan en la memoria del proceso, y el Dockerfile arranca uvicorn con uno solo.
    # El cuerpo de la petición puede pasar de MAX_FILE_SIZE en lo que ocupa la parte multipart que rodea a la foto
    # (separadores, nombre del campo y del archivo y tipo), que son unos cientos de bytes. Se deja bastante más.
    MARGEN_MULTIPART: int = 64 * 1024
    # Píxeles de la foto, contra las bombas de descompresión. Un PNG de menos de 1 MB con 30000 x 30000 son 900 MP y
    # 2,7 GB al decodificarlo. La galería de la web sube como mucho 4096 px de lado (16,8 MP), pero la cámara manda la
    # foto de ImageCapture sin reducir, y hay móviles de 48 y 50 MP. Una foto de verdad con más píxeles pesa más de
    # 10 MB y ya se rechaza por tamaño (la de 48 MP del banco pesa 21 MB). 50 MP son 150 MB al decodificarla.
    MAX_MEGAPIXELES: float = float(os.getenv("MAX_MEGAPIXELES", 50))
    # Peticiones a /process por IP. Como mucho LIMITE_PETICIONES_IP en cualquier intervalo de VENTANA_LIMITE_IP
    # segundos. Una persona manda dos fotos por DNI y a veces repite alguna, pero detrás de una misma IP puede haber
    # mucha gente (una red móvil, la wifi de la universidad o una sala entera en unas jornadas, que comparten una IPv4
    # o una misma red IPv6 /64). Con 120 por minuto unas 60 personas pueden subir sus dos fotos en el mismo minuto, y
    # quien manda fotos sin parar desde una conexión se queda en dos por segundo. Con 0 no se limita.
    LIMITE_PETICIONES_IP: int = int(os.getenv("LIMITE_PETICIONES_IP", 120))
    VENTANA_LIMITE_IP: int = int(os.getenv("VENTANA_LIMITE_IP", 60))
    # Recortes a la vez. Ultralytics pasa las fotos por el modelo de una en una, así que con 2 una foto se decodifica,
    # se endereza o se codifica mientras otra pasa por el modelo. En un Mac de 18 núcleos, con 24 fotos del banco, un
    # turno recorta 5,6 fotos por segundo, dos 10,5 y tres 13,4. El servidor tiene muchos menos núcleos y cada
    # turno puede ocupar 150 MB con una foto de 50 MP, así que se deja en 2. Con más núcleos se puede subir.
    RECORTES_SIMULTANEOS: int = max(1, int(os.getenv("RECORTES_SIMULTANEOS", 2)))
    # Fotos que pueden esperar turno. Con la cola llena se contesta 503 al momento, en vez de dejar esperando a todos.
    # Cada foto en la cola ocupa como mucho 10 MB de memoria y tarda unas décimas de segundo en recortarse, así que la
    # última de la cola espera unos segundos, muy lejos de los 90 s que aguanta la web. Si una sala entera sube las
    # fotos a la vez, esperan un poco en vez de recibir un error.
    COLA_RECORTES: int = max(0, int(os.getenv("COLA_RECORTES", 20)))
    # Segundos que se piden en Retry-After cuando la cola está llena
    ESPERA_SI_OCUPADO: int = 5
    # Cabecera con la IP real del visitante. La pone Caddy (ver Caddyfile) y solo se cree si la conexión llega desde
    # la propia máquina o desde la red interna de Docker, que es por donde entra Caddy.
    CABECERA_IP_CLIENTE: str = "x-ip-cliente"

    # Servidor
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = "0.0.0.0"
    
    # Entorno. Se lee de ENVIRONMENT (docker-compose ya pone "production").
    # Si no está definida se asume producción, así un despliegue sin variables
    # nunca deja la documentación abierta.
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production").strip().lower()

    @property
    def is_production(self) -> bool:
        """True salvo que ENVIRONMENT sea "development" """
        return self.ENVIRONMENT != "development"

    @property
    def environment(self) -> str:
        """Retorna el nombre del entorno"""
        return "production" if self.is_production else "development"

    @property
    def max_cuerpo(self) -> int:
        """Bytes que puede tener el cuerpo de una petición, la foto más la parte multipart"""
        return self.MAX_FILE_SIZE + self.MARGEN_MULTIPART

    @property
    def max_pixeles(self) -> int:
        """Píxeles que puede tener la foto recibida"""
        return int(self.MAX_MEGAPIXELES * 1_000_000)

    @property
    def cors_origins(self) -> list[str]:
        """Orígenes permitidos. Los de localhost solo en desarrollo"""
        if self.is_production:
            return self.PRODUCTION_ORIGINS
        return self.PRODUCTION_ORIGINS + self.DEV_ORIGINS


# Instancia única de configuración
settings = Settings()

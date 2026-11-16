"""
Protección de /process frente a abusos y sobrecargas

- Tamaño del cuerpo antes de leerlo (413)
- Límite de peticiones por IP real (429)
- Turnos de recorte, con una cola máxima (503)
"""
import asyncio
import ipaddress
import logging
import math
import time
from collections import deque
from itertools import islice
from typing import Optional

from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config.settings import settings
from detection.validators import detalle_error, error_archivo_grande

logger = logging.getLogger(__name__)

# IP que el límite de peticiones recuerda a la vez. Cada una ocupa menos de 1 KB y en un minuto normal hay unas pocas
MAX_IPS_VIGILADAS = 10_000


class CuerpoDemasiadoGrande(HTTPException):
    """El cuerpo de la petición ha pasado del máximo mientras llegaba"""

    def __init__(self):
        error = error_archivo_grande()
        super().__init__(status_code=error.status_code, detail=error.detail)


def _normalizar_ip(texto: str) -> Optional[str]:
    """
    Clave con la que se cuenta una IP, o None si el texto no es una IP.

    Las IPv6 se cuentan por su red /64, porque cada conexión de casa o del móvil suele tener una red /64 entera y
    podría ir cambiando de dirección dentro de ella.
    """
    try:
        ip = ipaddress.ip_address(texto.strip())
    except ValueError:
        return None
    if ip.version == 6:
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def _es_interna(texto: str) -> bool:
    """True si la conexión viene de la propia máquina o de una red privada, como la de Docker"""
    try:
        ip = ipaddress.ip_address(texto)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def ip_cliente(scope: Scope) -> Optional[str]:
    """
    IP del visitante para el límite de peticiones, o None si no se puede saber.

    En el servidor al contenedor solo llega Caddy, porque docker-compose publica el puerto solo en 127.0.0.1, y Caddy
    sustituye siempre la cabecera X-IP-Cliente por la IP real. Por eso esa cabecera solo se cree cuando la conexión
    viene de la propia máquina o de la red interna de Docker. Si llegara cualquier otra conexión, cuenta su propia IP,
    que no se puede falsificar. En desarrollo, sin Caddy, vale la IP de la conexión.

    La IP de la conexión tiene que llegar tal cual, sin que uvicorn la cambie por la de X-Forwarded-For. Por eso el
    Dockerfile arranca uvicorn con --no-proxy-headers.
    """
    cliente = scope.get("client")
    conexion = cliente[0] if cliente else ""
    if _es_interna(conexion):
        for nombre, valor in scope.get("headers", []):
            if nombre.decode("latin-1").lower() == settings.CABECERA_IP_CLIENTE:
                ip = _normalizar_ip(valor.decode("latin-1"))
                if ip:
                    return ip
                break
        if settings.is_production:
            # Un Caddy sin la cabecera (por ejemplo con el Caddyfile antiguo) hace que todas las peticiones parezcan
            # de la misma IP. Antes que frenar a todo el mundo a la vez, no se limita.
            return None
    return _normalizar_ip(conexion)


class LimitePorIP:
    """Como mucho `maximo` peticiones de cada IP en cualquier intervalo de `ventana` segundos"""

    def __init__(self, maximo: int, ventana: float, max_ips: int = MAX_IPS_VIGILADAS):
        self.maximo = maximo
        self.ventana = ventana
        self.max_ips = max(1, max_ips)
        self._marcas: dict[str, deque] = {}
        self._proxima_limpieza = 0.0

    def espera(self, clave: Optional[str], ahora: Optional[float] = None) -> float:
        """Apunta la petición y devuelve 0 si puede pasar o, si no, los segundos que tiene que esperar"""
        if self.maximo <= 0 or not clave:
            return 0.0
        ahora = time.monotonic() if ahora is None else ahora
        if ahora >= self._proxima_limpieza or (clave not in self._marcas and len(self._marcas) >= self.max_ips):
            self._limpiar(ahora)
        marcas = self._marcas.setdefault(clave, deque())
        while marcas and ahora - marcas[0] >= self.ventana:
            marcas.popleft()
        if len(marcas) >= self.maximo:
            return self.ventana - (ahora - marcas[0])
        marcas.append(ahora)
        return 0.0

    def _limpiar(self, ahora: float) -> None:
        """Olvida las IP que llevan una ventana entera sin enviar nada, para que la memoria no crezca sin fin"""
        caducadas = [clave for clave, marcas in self._marcas.items()
                     if not marcas or ahora - marcas[-1] >= self.ventana]
        for clave in caducadas:
            del self._marcas[clave]
        # Con muchísimas IP distintas en una misma ventana (por ejemplo, alguien que va cambiando de dirección IPv6)
        # se olvidan las que se vieron primero, hasta dejar sitio. Para ellas la cuenta vuelve a empezar, pero la
        # memoria queda acotada. Del exceso de peticiones ya protegen los turnos de recorte
        sobran = len(self._marcas) - self.max_ips * 9 // 10
        if sobran > 0:
            for clave in list(islice(self._marcas, sobran)):
                del self._marcas[clave]
            logger.warning(f"🧹 Demasiadas IP distintas en el límite de peticiones, se olvidan {sobran}")
        self._proxima_limpieza = ahora + self.ventana


class TurnosRecorte:
    """
    Cuántas fotos se recortan a la vez y cuántas pueden esperar turno.

    Con todos los turnos ocupados y la cola llena, la foto nueva se rechaza con un 503 en vez de esperar sin fin.
    """

    def __init__(self, simultaneos: int, cola: int):
        self.simultaneos = max(1, simultaneos)
        self.cola = max(0, cola)
        self._semaforo = asyncio.Semaphore(self.simultaneos)
        self._esperando = 0

    async def __aenter__(self) -> None:
        if self._semaforo.locked() and self._esperando >= self.cola:
            logger.warning(f"🚦 Servidor ocupado, sin turno libre y con la cola llena ({self._esperando} esperando)")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detalle_error(
                    "server_busy",
                    "El servidor está recortando muchas fotos a la vez.",
                    "Espera unos segundos y vuelve a intentarlo."
                ),
                headers={"Retry-After": str(settings.ESPERA_SI_OCUPADO)}
            )
        self._esperando += 1
        try:
            await self._semaforo.acquire()
        finally:
            self._esperando -= 1

    async def __aexit__(self, *excepcion) -> None:
        self._semaforo.release()


class ProteccionMiddleware:
    """
    Middleware ASGI que corta las peticiones antes de leer el cuerpo.

    - Límite de peticiones por IP en POST /process (429 con Retry-After)
    - Tamaño del cuerpo en todas las rutas. Si trae Content-Length se mira antes de leer nada y, si no la trae
      (cuerpo troceado), se va contando lo que llega y se corta al pasar del máximo (413)

    Tiene que ir dentro del middleware de CORS para que sus respuestas lleven las cabeceras de CORS. Sin ellas el
    navegador no deja leer la respuesta y la web solo vería «Failed to fetch».
    """

    def __init__(self, app: ASGIApp, limite: "LimitePorIP"):
        self.app = app
        self.limite = limite

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["method"] == "POST" and scope["path"] == "/process":
            espera = self.limite.espera(ip_cliente(scope))
            if espera > 0:
                segundos = max(1, math.ceil(espera))
                # No se registra la IP, es un dato personal
                logger.warning(f"⏱️ Límite de peticiones superado, faltan {segundos} s")
                await self._responder(scope, receive, send, status.HTTP_429_TOO_MANY_REQUESTS, detalle_error(
                    "too_many_requests",
                    "Se han enviado demasiadas fotos en poco tiempo desde tu conexión.",
                    f"Espera {segundos} segundo{'' if segundos == 1 else 's'} y vuelve a intentarlo."
                ), {"Retry-After": str(segundos)})
                return

        maximo = settings.max_cuerpo
        for nombre, valor in scope.get("headers", []):
            if nombre == b"content-length":
                try:
                    longitud = int(valor)
                except ValueError:
                    break
                if longitud > maximo:
                    await self._responder(scope, receive, send, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                          error_archivo_grande().detail)
                    return
                break

        # Aunque haya Content-Length se cuenta lo que llega, así se cubre también el cuerpo troceado
        recibido = 0
        respuesta_empezada = False

        async def receive_limitado() -> Message:
            nonlocal recibido
            mensaje = await receive()
            if mensaje["type"] == "http.request":
                recibido += len(mensaje.get("body", b""))
                if recibido > maximo:
                    # FastAPI deja pasar las HTTPException que salen al leer el cuerpo, así que la contesta él
                    raise CuerpoDemasiadoGrande()
            return mensaje

        async def send_vigilado(mensaje: Message) -> None:
            nonlocal respuesta_empezada
            if mensaje["type"] == "http.response.start":
                respuesta_empezada = True
            await send(mensaje)

        try:
            await self.app(scope, receive_limitado, send_vigilado)
        except CuerpoDemasiadoGrande as e:
            # Por si la excepción llega hasta aquí sin que nadie haya contestado
            if not respuesta_empezada:
                await self._responder(scope, receive, send, e.status_code, e.detail)

    @staticmethod
    async def _responder(scope: Scope, receive: Receive, send: Send, codigo: int, detalle: dict,
                         cabeceras: Optional[dict] = None) -> None:
        respuesta = JSONResponse({"detail": detalle}, status_code=codigo, headers=cabeceras)
        await respuesta(scope, receive, send)


# Instancias únicas, compartidas por el middleware y la ruta
limite_peticiones = LimitePorIP(settings.LIMITE_PETICIONES_IP, settings.VENTANA_LIMITE_IP)
turnos_recorte = TurnosRecorte(settings.RECORTES_SIMULTANEOS, settings.COLA_RECORTES)

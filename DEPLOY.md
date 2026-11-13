# Guía de despliegue en VPS

Pasos para desplegar y actualizar la API de detección de DNI en un servidor privado virtual (VPS).

## Prerrequisitos en el VPS

Docker con Docker Compose v2 (el comando `docker compose`, con espacio) y buildx.

```bash
# Ejemplo para Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
sudo systemctl enable --now docker
```

> **No uses el paquete antiguo `docker-compose` (versión 1.x, con guion).** Con las versiones actuales de Docker falla al recrear un contenedor con `KeyError: 'ContainerConfig'`. Para entonces ya ha parado el contenedor anterior, así que la API se queda caída. Si el servidor todavía lo tiene, mira el último apartado.

## Primer despliegue

### 1. Llevar el código al VPS

**Opción A, con Git.**

```bash
git clone <url-de-tu-repo>
cd dni-detection-api
```

**Opción B, copia manual** con `scp` o `rsync`. Hacen falta estos archivos y carpetas.

- `api/`
- `config/`
- `detection/`
- `model/`
- `main.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`

### 2. Construir y levantar el contenedor

Desde la carpeta del proyecto.

```bash
sudo docker compose up -d --build
```

- `-d` lo deja en segundo plano.
- `--build` reconstruye la imagen con el código actual.

### 3. Comprobar que funciona

```bash
sudo docker ps
curl -s http://localhost:8000/health
```

El contenedor `dni-api` debe aparecer como `Up` con el puerto `127.0.0.1:8000->8000/tcp`. Ese puerto solo escucha dentro del servidor. Desde fuera se entra por Caddy (ver `Caddyfile`), que pone el HTTPS.

```bash
curl -s https://ia.defiendetuidentidad.es/health
```

La documentación interactiva (`/docs`) solo existe con `ENVIRONMENT=development`. En producción devuelve 404.

### 4. Ver los registros

```bash
sudo docker compose logs -f
```

## Actualizar a una versión nueva

Desde la carpeta del proyecto en el VPS.

```bash
git pull
sudo docker compose up -d --build
curl -s https://ia.defiendetuidentidad.es/health
```

La imagen se construye antes de sustituir el contenedor, así que la API solo se para unos segundos. Comprueba siempre el `health` al terminar.

## Parar la aplicación

```bash
sudo docker compose down
```

## Si el servidor todavía tiene `docker-compose` 1.x

`docker-compose up -d --build` no sirve para actualizar, falla como se explica en los prerrequisitos. Hasta instalar Compose v2 hay que borrar el contenedor antes de crearlo de nuevo.

```bash
git pull
sudo docker-compose build
sudo docker-compose down
sudo docker-compose up -d
```

Si ya ha fallado y la API está caída, se recupera con `sudo docker-compose down && sudo docker-compose up -d`.

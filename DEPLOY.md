# Guía de Despliegue en VPS

Este documento detalla los pasos para desplegar la API de Detección de DNI en un Servidor Privado Virtual (VPS).

## Prerrequisitos en el VPS

Asegúrate de tener instalados `docker` y `docker-compose` en tu servidor.

```bash
# Ejemplo para Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker
```

## Pasos de Despliegue

### 1. Transferir el Código al VPS

Puedes usar `git clone` si subes este código a un repositorio, o copiar los archivos usando `scp` o `rsync`.

**Opción A (Git):**
```bash
git clone <url-de-tu-repo>
cd dni-detection-api
```

**Opción B (Copia manual):**
Sube los siguientes archivos/carpetas:
- `api/`
- `config/`
- `detection/`
- `model/`
- `main.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`

### 2. Construir y Levantar el Contenedor

Ejecuta el siguiente comando en el directorio del proyecto:

```bash
sudo docker-compose up -d --build
```

- `-d`: Ejecuta en segundo plano (detached mode).
- `--build`: Fuerza la reconstrucción de la imagen para asegurar que tienes los últimos cambios.

### 3. Verificar el Estado

Comprueba que el contenedor está corriendo:

```bash
sudo docker ps
```

Deberías ver algo como:
```
CONTAINER ID   IMAGE                 STATUS          PORTS                    NAMES
xxxxxxxxxxxx   dni-detection-api...  Up X seconds    0.0.0.0:8000->8000/tcp   dni-api
```

### 4. Ver Logs (Debugging)

Si algo falla, revisa los logs:

```bash
sudo docker-compose logs -f
```

## Verificación de API

Una vez desplegado, puedes probar la API accediendo a:

- Swagger UI: `http://<TU_IP_VPS>:8000/docs`
- Health Check (si existe): `http://<TU_IP_VPS>:8000/`

## Mantenimiento

Para detener la aplicación:
```bash
sudo docker-compose down
```

Para actualizar (después de traer nuevos cambios de código):
```bash
sudo docker-compose up -d --build
```

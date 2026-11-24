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

## Límites de `/process`

La API se protege sola frente a fotos enormes y avalanchas de peticiones. Cada límite contesta con un error en español, con el mismo formato que los demás, y la web lo enseña tal cual.

| Límite | Respuesta | Variable de entorno | Por defecto |
|---|---|---|---|
| Cuerpo de la petición de más de 10 MB. Se mira antes de leerlo y, si llega troceado, se corta al pasarse | 413 | – | 10 MB |
| Foto con demasiados píxeles. Se mira en su cabecera, sin decodificarla, contra las bombas de descompresión | 413 | `MAX_MEGAPIXELES` | 50 |
| Peticiones a `/process` desde una misma IP en cada ventana. Con 0 no se limita | 429 con `Retry-After` | `LIMITE_PETICIONES_IP` | 120 |
| Duración de esa ventana, en segundos | – | `VENTANA_LIMITE_IP` | 60 |
| Fotos que se recortan a la vez, en un hilo aparte. Mientras tanto la API sigue atendiendo `/health` y las subidas | – | `RECORTES_SIMULTANEOS` | 2 |
| Fotos que pueden esperar turno. Con la cola llena se contesta al momento | 503 con `Retry-After` | `COLA_RECORTES` | 20 |

Caddy corta además cualquier cuerpo de más de 32 MiB antes de que llegue a la API (ver `Caddyfile`). Va por encima de los 30 MB que la web deja elegir para que, entre 10 y 32 MiB, el 413 lo dé la API en español y con CORS (la cámara de la web manda la foto sin reducir y la de un móvil de 48 o 50 MP pasa de 10 MiB). El 413 de Caddy no lleva cuerpo ni CORS y el navegador solo enseñaría «Failed to fetch». Las cuentas se llevan en la memoria del proceso, así que se pierden al reiniciar el contenedor. Con varios procesos de uvicorn cada uno llevaría las suyas, pero el `Dockerfile` arranca uno solo.

Los valores por defecto están pensados para que no frenen a nadie de verdad. Una persona manda dos fotos por DNI y a veces repite alguna, pero detrás de una misma IP puede haber una sala entera en unas jornadas. Con 120 por minuto caben unas 60 personas subiendo a la vez, y la cola de 20 hace que esperen unos segundos en vez de recibir un error.

`RECORTES_SIMULTANEOS` depende de los núcleos del servidor (`nproc`). En un Mac de 18 núcleos, con un turno se recortan 5,6 fotos por segundo, con dos 10,5 y con tres 13,4, porque el modelo pasa las fotos de una en una pero el resto del recorte se puede hacer a la vez. Con 2 núcleos conviene dejarlo en 2. Con 4 o más se puede probar 3, teniendo en cuenta que cada turno puede ocupar 150 MB de memoria con una foto de 50 MP.

### Cambiar un valor

Sin tocar los archivos del repo, con un `docker-compose.override.yml` en la carpeta del proyecto. Docker Compose lo junta solo con `docker-compose.yml` y git no lo sigue.

```yaml
services:
  dni-api:
    environment:
      - LIMITE_PETICIONES_IP=300
```

```bash
sudo docker compose up -d
sudo docker compose exec dni-api env | grep LIMITE_PETICIONES_IP
```

### Ver si salta algún límite

```bash
sudo docker compose logs --since 24h | grep -E "Límite de peticiones|Servidor ocupado|demasiados píxeles"
```

### La IP real del visitante

La API no ve al visitante. Ve a Caddy, y Caddy ve a Cloudflare. La IP real llega así y nadie la puede falsificar.

1. Cloudflare pone en `CF-Connecting-IP` la IP de quien se conecta a él y sustituye la que mande el visitante.
2. Caddy solo cree esa cabecera si la conexión viene de un rango de Cloudflare (`trusted_proxies` y `client_ip_headers` en el `Caddyfile`). Con cualquier otra conexión, `{client_ip}` es la IP de la propia conexión.
3. Caddy se la pasa a la API en `X-IP-Cliente` y sustituye siempre la que traiga la petición.
4. El contenedor solo publica el puerto en `127.0.0.1`, así que a la API solo llega Caddy. Aun así, la API solo cree `X-IP-Cliente` si la conexión viene de la propia máquina o de la red interna de Docker. Para eso uvicorn arranca con `--no-proxy-headers` (ver `Dockerfile`) y deja la IP de la conexión tal cual.

Si la cabecera no llega, por ejemplo con el Caddyfile antiguo, la API no limita por IP, en vez de contar a todo el mundo como si fuera una sola IP. En desarrollo, sin Caddy, cuenta la IP de la conexión. Las IPv6 se cuentan por su red /64, porque cada casa o cada móvil suele tener una entera.

## Cómo se recorta el DNI

Las esquinas salen, como siempre, de simplificar a cuatro puntos el contorno que da el modelo, pero sin las manchas sueltas de la máscara. Con el DNI girado la máscara se come la sombra de la tarjeta en trozos aparte, y antes el contorno saltaba a ellos y el recorte salía torcido (ver `detection/yolo_processor.py`). Además se calculan otras esquinas con las rectas de los cuatro lados (ver `detection/esquinas.py`). Si el DNI está girado o sale pequeño en la foto, para ellas su zona pasa otra vez por el modelo, derecha y de cerca. Solo se usan si se separan de las de siempre más del 5 % de la diagonal, que es cuando el polígono falla de verdad, por ejemplo con un dedo en una esquina. Por debajo el recorte queda como siempre, porque el modelo de campos de la web se entrenó con esos recortes. Cuesta unos milisegundos más, y una pasada más del modelo en las fotos con el DNI girado o lejos.

| Qué | Variable de entorno | Por defecto |
|---|---|---|
| Calcular también las esquinas por las rectas de los lados. Con 0 no se calculan | `ESQUINAS_POR_LADOS` | 1 |
| Separación desde la que se usan las esquinas por los lados, en fracción de la diagonal del DNI | `SEPARACION_ESQUINAS` | 0,05 |
| Giro desde el que se repite la pasada con el DNI derecho, en grados. Con 0 no se mira el giro | `GIRO_SEGUNDA_PASADA` | 4 |
| Tamaño por debajo del cual se repite la pasada de cerca (lado mayor del DNI entre lado mayor de la foto) | `DNI_PEQUENO` | 0,5 |
| Fondo que se deja alrededor del DNI en esa pasada, en fracción de su lado mayor | `MARGEN_SEGUNDA_PASADA` | 0,25 |
| Lo que tiene que ocupar el DNI en la foto para aceptar el segundo intento con margen (escaneos y fotos con poca luz) | `OCUPACION_MINIMA_REINTENTO` | 0,25 |

Se cambian igual que los límites, con el `docker-compose.override.yml`. Para ver cuántas fotos pasan por la segunda pasada y en cuántas se usan las esquinas por los lados:

```bash
sudo docker compose logs --since 24h | grep -c "Pasada de cerca con"
sudo docker compose logs --since 24h | grep -c "Esquinas por los lados"
```

## Atender solo a Cloudflare

`ia.defiendetuidentidad.es` apunta a Cloudflare, pero el servidor también contesta a quien lo llama directamente por su IP. Así se salta Cloudflare y sus protecciones. Los pasos de este apartado lo cierran con `scripts/solo_cloudflare.sh`.

### Qué se ha valorado

- **Regla en el Caddyfile.** Caddy corta sin contestar cualquier petición que no venga de un rango de Cloudflare (`@fuera_de_cloudflare` y `abort`). No toca el SSH ni la renovación del certificado y se deshace con la copia del Caddyfile anterior. Ya está en el repo. **Se aplica.**
- **Cortafuegos del sistema (ufw).** Deja los puertos 80 y 443 solo para los rangos de Cloudflare y el SSH abierto como ahora. Por la IP ya no se completa ni la conexión. El script lo hace con una orden, guarda antes el estado de ufw, abre el SSH antes de activar nada y, si `/health` deja de contestar por Cloudflare, deja ufw como estaba. **Se aplica.**
- **Cortafuegos de Hetzner Cloud.** Hace lo mismo que ufw, pero fuera del servidor. Hay que meter a mano los 22 rangos en las reglas de los puertos 80 y 443, y un rango mal copiado deja sin servicio a los visitantes que entren por él. Con ufw ya puesto aporta poco. **Opcional**, al final de este apartado.
- **Authenticated Origin Pulls de Cloudflare.** Caddy solo aceptaría conexiones con un certificado de cliente de Cloudflare. Con el certificado común lo presenta cualquier cuenta de Cloudflare, así que no protege más que los rangos. Con uno propio sí, pero hay que crearlo, subirlo con la API de Cloudflare, instalarlo en Caddy y activarlo en los dos lados en el orden correcto, o la API se cae. **Se deja para más adelante.** Con los rangos, alguien todavía podría llegar a través de su propia cuenta de Cloudflare. Llegaría con su IP real en `CF-Connecting-IP`, que Cloudflare no deja cambiar, así que el límite por IP le seguiría aplicando.

### Requisitos

- Caddy 2.7 o posterior (`caddy version`). El Caddyfile usa `client_ip_headers` y `{client_ip}`, que Caddy 2.6 no entiende. Está probado con las imágenes oficiales de Caddy 2.6.4 (falla), 2.7.6 y 2.11.9. El paquete `caddy` de los repositorios de Ubuntu es la 2.6, así que, si el servidor lo tiene, primero hay que instalar el oficial (ver más abajo). El script lo comprueba y para sin cambiar nada.
- Caddy como servicio de systemd (`systemctl status caddy`). El script instala el Caddyfile en el archivo que carga el servicio, normalmente `/etc/caddy/Caddyfile`, y antes guarda una copia del que hay.
- `ufw` filtrando también IPv6 (`IPV6=yes` en `/etc/default/ufw`, lo normal en Ubuntu).
- Lanzar el script como root desde la carpeta del repo en el servidor. Sin `--aplicar` no cambia nada, solo comprueba y enseña los comandos.

### Pasos

Hay que hacerlos en este orden. Cada paso deja la API funcionando aunque no se haga el siguiente.

1. Traer el código y ver el estado. No cambia nada.

   ```bash
   cd ~/yolo-api && git pull
   sudo bash scripts/solo_cloudflare.sh comprobar
   ```

2. Caddy, primero en modo de prueba y después de verdad. Comprueba que los rangos del Caddyfile coinciden con los que publica Cloudflare en ese momento y lo valida. Después guarda el anterior en `/etc/caddy/Caddyfile.antes-solo-cloudflare-FECHA`, instala el del repo, recarga Caddy y mira `/health` a través de Cloudflare. Si Caddy no lo acepta o `/health` no contesta, vuelve solo a la configuración anterior.

   ```bash
   sudo bash scripts/solo_cloudflare.sh caddy
   sudo bash scripts/solo_cloudflare.sh caddy --aplicar
   ```

3. La API nueva, como cualquier despliegue. Va después de Caddy porque es Caddy quien le pasa la IP real. Al revés también funciona, pero mientras tanto la API no limita por IP y alguien podría hacerse pasar por otra IP con la cabecera `X-IP-Cliente`.

   ```bash
   sudo docker compose up -d --build && sleep 8 && curl -s http://localhost:8000/health
   ```

4. Cortafuegos, también primero en modo de prueba. Enseña las reglas y avisa si hay otros puertos abiertos hacia fuera que se van a cerrar, por ejemplo el de un programa de monitorización. Si ufw ya está activo con alguna regla que abre 80 o 443 a todo el mundo y que el script no sabe quitar (por ejemplo `80,443/tcp` o el perfil `WWW Full`), lo avisa, y con `--aplicar` para sin cambiar nada hasta que se quite a mano con `sudo ufw status numbered` y `sudo ufw delete NÚMERO`. Las sueltas (`80`, `443`, `80/tcp`, `443/tcp`) las quita él. Con `--aplicar` guarda primero en `/var/backups/ufw-antes-solo-cloudflare/FECHA` si ufw estaba activo y sus archivos de reglas (`/etc/ufw/user.rules`, `user6.rules`, `ufw.conf` y `/etc/default/ufw`). Después permite el SSH, abre 80 y 443 para cada rango de Cloudflare, cierra el resto y activa ufw. Si `/health` deja de contestar a través de Cloudflare, deja ufw como estaba con esa copia. Al terminar vuelve a buscar reglas que abran 80 o 443 a todo el mundo y, si queda alguna, para y la enseña.

   ```bash
   sudo bash scripts/solo_cloudflare.sh cortafuegos
   sudo bash scripts/solo_cloudflare.sh cortafuegos --aplicar
   ```

   Antes de lanzarlo, abre una segunda sesión SSH y no la cierres hasta comprobar que puedes volver a entrar. Si te quedaras fuera, la consola web de Hetzner Cloud da acceso al servidor sin SSH, y desde ella se desactiva con `ufw disable`.

### Comprobar que por la IP ya no contesta

Desde otro equipo, nunca desde el propio servidor. `IP_DEL_SERVIDOR` es la IPv4 pública del servidor. Está en la consola de Hetzner Cloud y el script la enseña al terminar el paso 4.

```bash
# Por Cloudflare tiene que contestar {"status":"healthy","model_loaded":true}
curl -sS --max-time 10 https://ia.defiendetuidentidad.es/health

# Directamente a la IP no tiene que contestar
curl -sS --max-time 10 --resolve ia.defiendetuidentidad.es:443:IP_DEL_SERVIDOR https://ia.defiendetuidentidad.es/health
```

Antes del paso 2 la llamada directa devuelve `healthy`, que es el problema. Tras el paso 2 Caddy la corta sin contestar y curl da el error 92 («HTTP/2 stream 1 was not closed cleanly») o, con `--http1.1`, el 52 («Empty reply from server»). Tras el paso 4 ni siquiera conecta y curl se rinde a los 10 segundos con el error 28. Si el servidor tiene IPv6, repite la prueba con `[IPv6_DEL_SERVIDOR]` en lugar de la IPv4.

### Deshacer

```bash
# Deja ufw como estaba antes de la primera vez que se aplicó el paso 4. Si ya estaba activo, con sus reglas de
# entonces y recargado. Si no, desactivado. Sin copia, solo lo desactiva
sudo bash scripts/solo_cloudflare.sh deshacer-cortafuegos --aplicar

# Vuelve a la última copia del Caddyfile anterior y recarga Caddy
sudo bash scripts/solo_cloudflare.sh deshacer-caddy --aplicar
```

La API nueva funciona igual con el Caddyfile anterior, solo que sin límite por IP.

### Mantenimiento

- **Si Cloudflare cambia sus rangos**, el script lo detecta y para. Hay que poner al día las dos listas del `Caddyfile` (`trusted_proxies` y `remote_ip`), subirlo y repetir los pasos 1, 2 y 4. Para comparar los del repo con los publicados en cualquier momento, desde cualquier equipo, `bash scripts/solo_cloudflare.sh rangos`. Repetir el paso 4 añade los rangos nuevos, pero no quita los que Cloudflare haya dejado de usar. Esos se borran con `sudo ufw status numbered` y `sudo ufw delete NÚMERO`.
- **Si algún día se quita el proxy de Cloudflare** (la nube gris en el DNS de `ia`), antes hay que deshacer los dos pasos. Si no, nadie llega a la API y Caddy no puede renovar el certificado.
- **La renovación del certificado no cambia.** Los retos de Let's Encrypt llegan a través de Cloudflare, porque el dominio apunta a Cloudflare, así que entran por sus rangos. Además la regla de Caddy solo está en el sitio HTTPS y los retos HTTP-01 van al puerto 80.

### Opcional, cortafuegos de Hetzner Cloud

En la consola de Hetzner Cloud, en Firewalls, se crea uno con estas reglas de entrada y se aplica al servidor.

- TCP 22 desde cualquier IPv4 e IPv6, o solo desde las IP desde las que se entra.
- TCP 80 desde cada uno de los rangos de Cloudflare.
- TCP 443 desde los mismos rangos.

Los rangos salen, uno por línea, con `bash scripts/solo_cloudflare.sh rangos`. Un firewall de Hetzner cierra todo lo que no tenga regla, así que cualquier otro servicio que deba verse desde fuera necesita la suya. Se deshace quitando el firewall del servidor en la misma consola.

### Si el servidor tiene Caddy 2.6 o anterior

Hay que instalar el paquete oficial de Caddy, con los pasos de https://caddyserver.com/docs/install para Debian y Ubuntu. Al actualizarlo Caddy se reinicia y la API se corta unos segundos. Guarda antes una copia del Caddyfile y comprueba después que sigue siendo el mismo.

```bash
sudo cp -a /etc/caddy/Caddyfile /root/Caddyfile.antes-de-actualizar-caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
caddy version
sudo diff /root/Caddyfile.antes-de-actualizar-caddy /etc/caddy/Caddyfile && curl -s https://ia.defiendetuidentidad.es/health
```

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

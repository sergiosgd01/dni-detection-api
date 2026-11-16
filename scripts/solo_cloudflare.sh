#!/usr/bin/env bash
# Deja el servidor de la API atendiendo solo a lo que llega por Cloudflare.
#
# Se lanza en el servidor, desde la carpeta del repo (~/yolo-api) y como root. Sin --aplicar no cambia nada, solo
# hace comprobaciones de lectura y muestra los comandos que lanzaría. Estas son las órdenes:
#
#   bash scripts/solo_cloudflare.sh comprobar                      estado actual, sin cambiar nada
#   bash scripts/solo_cloudflare.sh caddy [--aplicar]              instala el Caddyfile del repo y recarga Caddy
#   bash scripts/solo_cloudflare.sh cortafuegos [--aplicar]        80 y 443 solo para Cloudflare con ufw, SSH abierto
#   bash scripts/solo_cloudflare.sh deshacer-cortafuegos [--aplicar] ufw como estaba antes del primer cortafuegos
#   bash scripts/solo_cloudflare.sh deshacer-caddy [--aplicar]
#   bash scripts/solo_cloudflare.sh rangos                         rangos de Cloudflare, uno por línea
#
# El orden seguro está en DEPLOY.md (git pull, caddy, docker compose, cortafuegos). Cada paso deja la API funcionando
# aunque no se haga el siguiente. Los dos que cambian algo comprueban /health a través de Cloudflare al terminar y,
# si no contesta, se deshacen solos. El del cortafuegos guarda antes el estado de ufw para volver a él.

set -euo pipefail

DOMINIO="ia.defiendetuidentidad.es"
ADMIN_CADDY="http://localhost:2019"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CADDYFILE_REPO="$REPO/Caddyfile"
SUFIJO_COPIA="antes-solo-cloudflare"
# Copias del estado de ufw antes de tocarlo, una carpeta por cada vez que se aplica el paso del cortafuegos
COPIAS_UFW="/var/backups/ufw-$SUFIJO_COPIA"
# Archivos donde ufw guarda su estado. Las reglas añadidas, si está activo y la política por defecto
ARCHIVOS_UFW=(/etc/ufw/user.rules /etc/ufw/user6.rules /etc/ufw/ufw.conf /etc/default/ufw)
CARGADA=""

APLICAR=0
ORDEN="${1:-}"
[[ "${2:-}" == "--aplicar" || "${1:-}" == "--aplicar" ]] && APLICAR=1
[[ "$ORDEN" == "--aplicar" ]] && ORDEN="${2:-}"

titulo() { printf '\n== %s ==\n' "$*"; }
bien() { printf '  [bien] %s\n' "$*"; }
aviso() { printf '  [aviso] %s\n' "$*"; }
parar() { printf '\n  [parado] %s\n' "$*" >&2; exit 1; }

# Muestra el comando y solo lo lanza con --aplicar
ejecutar() {
  local arg linea="  +"
  for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Za-z0-9_./:,=+@%-]+$ ]]; then linea+=" $arg"; else linea+=" '$arg'"; fi
  done
  printf '%s\n' "$linea"
  if (( APLICAR )); then "$@"; fi
}

exigir_root() {
  if (( APLICAR )) && [[ "$(id -u)" != "0" ]]; then
    parar "Con --aplicar hay que lanzarlo como root (o con sudo)."
  fi
}

# --------------------------------------------------------------------------------------------- rangos de Cloudflare

rangos_trusted_proxies() {
  { grep -E '^[[:space:]]*trusted_proxies static ' "$CADDYFILE_REPO" || true; } \
    | sed -E 's/^[[:space:]]*trusted_proxies static //' | tr -s ' \t' '\n' | sed '/^$/d' | sort -u
}

rangos_remote_ip() {
  { grep -E '^[[:space:]]*remote_ip ' "$CADDYFILE_REPO" || true; } \
    | sed -E 's/^[[:space:]]*remote_ip //' | tr -s ' \t' '\n' | sed '/^$/d' | sort -u
}

rangos_publicados() {
  { curl -fsS --max-time 20 https://www.cloudflare.com/ips-v4 && echo
    curl -fsS --max-time 20 https://www.cloudflare.com/ips-v6 && echo; } | sed '/^$/d' | sort -u
}

# Para si las dos listas del Caddyfile no coinciden entre sí o con las que publica Cloudflare ahora mismo. Si
# Cloudflare hubiera añadido un rango, sus conexiones desde ahí se cortarían, así que antes hay que poner el repo al día
comprobar_rangos() {
  local trusted remote publicados
  trusted="$(rangos_trusted_proxies)"
  remote="$(rangos_remote_ip)"
  [[ -n "$trusted" ]] || parar "No encuentro la línea trusted_proxies en $CADDYFILE_REPO."
  if [[ "$trusted" != "$remote" ]]; then
    diff <(echo "$trusted") <(echo "$remote") || true
    parar "Los rangos de trusted_proxies y de remote_ip del Caddyfile no son los mismos."
  fi
  bien "El Caddyfile tiene los mismos $(echo "$trusted" | wc -l | tr -d ' ') rangos en trusted_proxies y en remote_ip."
  if ! publicados="$(rangos_publicados)" || [[ -z "$publicados" ]]; then
    parar "No he podido descargar los rangos de https://www.cloudflare.com/ips/ para compararlos."
  fi
  if [[ "$trusted" != "$publicados" ]]; then
    echo "  Diferencias (< Caddyfile, > Cloudflare)"
    diff <(echo "$trusted") <(echo "$publicados") || true
    parar "Cloudflare ha cambiado sus rangos. Pon al día las dos listas del Caddyfile en el repo antes de seguir."
  fi
  bien "Coinciden con los que publica Cloudflare ahora."
}

rangos() { rangos_trusted_proxies; }

# ------------------------------------------------------------------------------------------------------------ Caddy

# Archivo que carga el servicio de Caddy (el --config de su ExecStart). Si no se ve, el de siempre
caddyfile_sistema() {
  local ruta=""
  if command -v systemctl >/dev/null 2>&1; then
    ruta="$(systemctl show caddy -p ExecStart --value 2>/dev/null | grep -oE -- '--config[= ][^ ;]+' | head -n1 \
      | sed -E 's/--config[= ]//')" || true
  fi
  echo "${ruta:-/etc/caddy/Caddyfile}"
}

version_caddy() { { caddy version 2>/dev/null || true; } | { grep -oE '^v[0-9]+\.[0-9]+' || true; } | tr -d v; }

comprobar_version_caddy() {
  command -v caddy >/dev/null 2>&1 || parar "No encuentro el programa caddy."
  local version mayor menor
  version="$(version_caddy)"
  mayor="${version%%.*}"
  menor="${version#*.}"
  if [[ -z "$version" ]] || (( mayor < 2 || (mayor == 2 && menor < 7) )); then
    parar "Caddy ${version:-desconocido} es demasiado antiguo. El Caddyfile necesita 2.7 o posterior (ver DEPLOY.md)."
  fi
  bien "Caddy $version."
}

comprobar_servicio_caddy() {
  if systemctl is-active --quiet caddy 2>/dev/null; then
    bien "Caddy corre como servicio de systemd y carga $(caddyfile_sistema)."
  elif (( APLICAR )); then
    parar "Caddy no corre como servicio de systemd (caddy.service). Hay que instalar el Caddyfile y recargarlo a mano."
  else
    aviso "Caddy no corre como servicio de systemd (caddy.service). Con --aplicar se pararía aquí."
  fi
}

validar_caddyfile_repo() {
  if caddy validate --adapter caddyfile --config "$CADDYFILE_REPO" >/tmp/caddy-validate.log 2>&1; then
    bien "caddy validate da por bueno el Caddyfile del repo."
  else
    tail -n 5 /tmp/caddy-validate.log
    parar "caddy validate no da por bueno el Caddyfile del repo. No se ha cambiado nada."
  fi
}

salud_por_cloudflare() {
  curl -fsS --max-time 20 "https://$DOMINIO/health" 2>/dev/null | grep -q '"healthy"'
}

ip_publica() {
  ip -4 -o route get 1.1.1.1 2>/dev/null | grep -oE 'src [0-9.]+' | cut -d' ' -f2 || true
}

# Llamada directa a la IP del servidor desde el propio servidor, sin pasar por Cloudflare. Con el Caddyfile nuevo
# Caddy corta la conexión sin contestar. El cortafuegos no se nota aquí, porque el tráfico local no pasa por él
comprobar_corte_por_ip() {
  local ip
  ip="$(ip_publica)"
  [[ -n "$ip" ]] || { aviso "No he podido averiguar la IP pública para probar la llamada directa."; return; }
  if curl -fsS --max-time 10 --resolve "$DOMINIO:443:$ip" "https://$DOMINIO/health" >/dev/null 2>&1; then
    aviso "Llamando directamente a $ip la API todavía contesta."
  else
    bien "Llamando directamente a $ip ya no contesta (lo corta Caddy)."
  fi
}

# Vuelve a la configuración de antes. Con copia del archivo se restaura y se recarga. Sin ella (el servicio carga
# directamente el Caddyfile del repo) se vuelve a cargar en Caddy la configuración que tenía, guardada con su API
volver_caddy_anterior() {
  local destino="$1" copia="$2" cargada="$3"
  if [[ -n "$copia" && -f "$copia" ]]; then
    cp "$copia" "$destino"
    systemctl reload caddy || true
    aviso "Se ha vuelto a dejar el Caddyfile anterior ($copia) y Caddy lo ha cargado."
  elif [[ -n "$cargada" && -s "$cargada" ]]; then
    curl -fsS --max-time 10 -X POST -H 'Content-Type: application/json' --data-binary @"$cargada" \
      "$ADMIN_CADDY/load" >/dev/null 2>&1 || true
    aviso "Caddy vuelve a tener la configuración de antes, pero $destino sigue siendo el nuevo."
    aviso "Antes de reiniciar Caddy o el servidor, deja en $destino el Caddyfile anterior."
  else
    aviso "No hay copia de la configuración anterior. El Caddyfile de antes era solo el bloque que enseña deshacer-caddy."
  fi
}

orden_caddy() {
  exigir_root
  titulo "Caddyfile del repo"
  comprobar_version_caddy
  comprobar_servicio_caddy
  comprobar_rangos
  validar_caddyfile_repo

  local destino real copia cargada=""
  destino="$(caddyfile_sistema)"
  real="$(readlink -f "$destino" 2>/dev/null || echo "$destino")"
  copia="$destino.$SUFIJO_COPIA-$(date +%Y%m%d-%H%M%S)"
  if (( APLICAR )); then
    # La configuración que Caddy tiene cargada ahora, por si hay que volver a ella. Se borra al terminar
    CARGADA="$(mktemp /tmp/caddy-cargada-XXXXXX)"
    trap '[[ -z "${CARGADA:-}" ]] || rm -f "$CARGADA"' EXIT
    if curl -fsS --max-time 5 "$ADMIN_CADDY/config/" -o "$CARGADA" 2>/dev/null; then cargada="$CARGADA"; fi
  fi
  titulo "Instalar y recargar Caddy ($destino)"
  if [[ "$real" == "$(readlink -f "$CADDYFILE_REPO")" ]]; then
    bien "$destino ya es el Caddyfile del repo, no hace falta copiarlo."
    copia=""
  elif [[ -f "$destino" ]] && cmp -s "$destino" "$CADDYFILE_REPO"; then
    bien "$destino ya es igual que el del repo."
    copia=""
  else
    if [[ -f "$destino" ]]; then ejecutar cp -a "$destino" "$copia"; else copia=""; fi
    ejecutar cp "$CADDYFILE_REPO" "$destino"
  fi
  # Si la configuración nueva no carga, Caddy sigue con la anterior. Aun así se deja también el archivo anterior,
  # para que Caddy no arranque con el nuevo en el próximo reinicio
  if ! ejecutar systemctl reload caddy; then
    volver_caddy_anterior "$destino" "$copia" "$cargada"
    parar "Caddy no ha aceptado el Caddyfile nuevo. Revisa 'journalctl -u caddy -n 50' antes de repetirlo."
  fi

  (( APLICAR )) || { echo; echo "Modo de prueba, no se ha cambiado nada. Para hacerlo, añade --aplicar."; return; }
  sleep 3
  titulo "Comprobaciones"
  if salud_por_cloudflare; then
    bien "https://$DOMINIO/health contesta a través de Cloudflare."
  else
    aviso "https://$DOMINIO/health no contesta a través de Cloudflare. Se vuelve a la configuración anterior."
    volver_caddy_anterior "$destino" "$copia" "$cargada"
    parar "Revisa 'journalctl -u caddy -n 50' antes de repetirlo."
  fi
  comprobar_corte_por_ip
  [[ -n "$copia" ]] && echo "  Copia del Caddyfile anterior en $copia"
  return 0
}

orden_deshacer_caddy() {
  exigir_root
  local destino copia
  destino="$(caddyfile_sistema)"
  copia="$(ls -1t "$destino.$SUFIJO_COPIA-"* 2>/dev/null | head -n1 || true)"
  titulo "Volver al Caddyfile anterior"
  if [[ -z "$copia" ]]; then
    aviso "No hay ninguna copia $destino.$SUFIJO_COPIA-*. El Caddyfile de antes era solo esto."
    printf '\n    ia.defiendetuidentidad.es {\n        reverse_proxy localhost:8000\n    }\n\n'
    parar "Escríbelo en $destino y recarga con 'systemctl reload caddy'."
  fi
  ejecutar cp "$copia" "$destino"
  ejecutar systemctl reload caddy
  (( APLICAR )) || echo "Modo de prueba, no se ha cambiado nada. Para hacerlo, añade --aplicar."
}

# ------------------------------------------------------------------------------------------------------ Cortafuegos

puertos_ssh() {
  {
    [[ -n "${SSH_CONNECTION:-}" ]] && echo "${SSH_CONNECTION##* }"
    sshd -T 2>/dev/null | awk '$1 == "port" {print $2}' || true
    ss -Htlnp 2>/dev/null | awk '/"sshd"/ {print $4}' | sed -E 's/.*:([0-9]+)$/\1/' || true
  } | grep -E '^[0-9]+$' | sort -un || true
}

# Sockets que escuchan fuera de la propia máquina y que el cortafuegos va a cerrar
otros_puertos_abiertos() {
  local ssh
  ssh="$(puertos_ssh | tr '\n' '|')"
  ss -Htulnp 2>/dev/null | awk '{print $1, $5, $7}' \
    | grep -vE ' (127\.[0-9.]+|\[::1\]|\[::ffff:127\.[0-9.]+\])(%[a-z0-9]+)?:[0-9]+ ' \
    | grep -vE ":(80|443|${ssh%|}) " || true
}

ufw_activo() { ufw status 2>/dev/null | head -n1 | grep -q 'Status: active'; }

# Si una lista de puertos de ufw (80, 80,443, 8000:9000...) incluye el 80 o el 443
puertos_incluyen_web() {
  local item desde hasta
  for item in ${1//,/ }; do
    if [[ "$item" =~ ^([0-9]+):([0-9]+)$ ]]; then
      desde="${BASH_REMATCH[1]}"; hasta="${BASH_REMATCH[2]}"
      if (( (desde <= 80 && 80 <= hasta) || (desde <= 443 && 443 <= hasta) )); then return 0; fi
    elif [[ "$item" == 80 || "$item" == 443 ]]; then
      return 0
    fi
  done
  return 1
}

# Si el destino de una regla de 'ufw status' (columna To) incluye el 80 o el 443. Puede ser Anywhere, unos puertos
# con o sin protocolo, una dirección con o sin puertos o el nombre de un perfil de aplicación (WWW Full, Nginx Full)
destino_incluye_web() {
  local destino="$1" ultimo puertos
  destino="${destino% (v6)}"
  destino="$(sed -E 's/ on [^ ]+$//' <<<"$destino")"
  [[ "$destino" == Anywhere ]] && return 0
  ultimo="${destino##* }"
  ultimo="${ultimo%/tcp}"
  ultimo="${ultimo%/udp}"
  if [[ "$ultimo" =~ ^[0-9]+([:,][0-9]+)*$ && "$ultimo" != *::* ]]; then
    puertos_incluyen_web "$ultimo"
    return
  fi
  # Una dirección sin puertos abre todos
  [[ "$destino" =~ ^[0-9A-Fa-f.:/]+$ ]] && return 0
  # Perfil de aplicación. Si no se pueden leer sus puertos, se da por hecho que los incluye
  puertos="$(ufw app info "$destino" 2>/dev/null | sed -n '/^Ports:/,$p' | sed '1d' | tr -d ' ' | sed -E 's#/(tcp|udp)$##')"
  [[ -n "$puertos" ]] || return 0
  local linea
  while IFS= read -r linea; do
    puertos_incluyen_web "$linea" && return 0
  done <<<"$puertos"
  return 1
}

# Reglas que dejan entrar en 80 o 443 desde cualquier sitio. Con alguna de ellas el resto del cortafuegos no sirve
reglas_web_abiertas() {
  local linea destino origen
  { ufw status 2>/dev/null | grep -E '[[:space:]](ALLOW|LIMIT)( IN)?[[:space:]]' || true; } | while IFS= read -r linea; do
    destino="$(sed -E 's/[[:space:]]+(ALLOW|LIMIT)( IN)?[[:space:]].*//' <<<"$linea")"
    origen="$(sed -E 's/.*[[:space:]](ALLOW|LIMIT)( IN)?[[:space:]]+//' <<<"$linea")"
    [[ "$origen" == Anywhere* ]] || continue
    if destino_incluye_web "$destino"; then echo "$linea"; fi
  done
}

# Las que el script quita solo, porque abren 80 o 443 sueltos (las que salen de 'ufw allow 80', 'ufw allow http'...)
regla_que_se_quita() {
  local destino
  destino="$(sed -E 's/[[:space:]]+(ALLOW|LIMIT)( IN)?[[:space:]].*//' <<<"$1")"
  destino="${destino% (v6)}"
  [[ "$destino" =~ ^(80|443)(/tcp)?$ ]]
}

# Guarda el estado de ufw antes de tocarlo, para poder volver a él tal cual. Deja la ruta de la copia en COPIA_UFW
guardar_ufw() {
  COPIA_UFW="$COPIAS_UFW/$(date +%Y%m%d-%H%M%S)"
  local estado=inactivo archivo
  ufw_activo && estado=activo
  titulo "Copia del estado de ufw (ahora está $estado)"
  ejecutar mkdir -p "$COPIA_UFW"
  for archivo in "${ARCHIVOS_UFW[@]}"; do
    [[ -f "$archivo" ]] && ejecutar cp -a "$archivo" "$COPIA_UFW/$(echo "${archivo#/}" | tr / _)"
  done
  printf '  + echo %s > %s\n' "$estado" "$COPIA_UFW/estado"
  if (( APLICAR )); then echo "$estado" >"$COPIA_UFW/estado"; fi
}

# Deja ufw como estaba al hacer la copia. Si estaba activo, con sus reglas de entonces y recargado. Si no, desactivado
restaurar_ufw() {
  local copia="$1" estado archivo
  estado="$(cat "$copia/estado" 2>/dev/null || echo inactivo)"
  [[ "$estado" == activo ]] || ejecutar ufw disable
  for archivo in "${ARCHIVOS_UFW[@]}"; do
    [[ -f "$copia/$(echo "${archivo#/}" | tr / _)" ]] && ejecutar cp -a "$copia/$(echo "${archivo#/}" | tr / _)" "$archivo"
  done
  if [[ "$estado" == activo ]]; then ejecutar ufw reload; fi
  if (( APLICAR )); then aviso "ufw vuelve a estar como antes ($estado, copia en $copia)."; fi
}

orden_cortafuegos() {
  exigir_root
  titulo "Comprobaciones previas"
  command -v ufw >/dev/null 2>&1 || parar "ufw no está instalado. En Ubuntu se instala con 'apt-get install -y ufw'."
  comprobar_rangos
  if ! grep -qE '^IPV6=yes' /etc/default/ufw 2>/dev/null; then
    parar "ufw no filtra IPv6 (falta IPV6=yes en /etc/default/ufw) y la IP v6 del servidor quedaría abierta."
  fi
  bien "ufw filtra también IPv6."

  local ssh
  ssh="$(puertos_ssh)"
  [[ -n "$ssh" ]] || parar "No veo en qué puerto escucha sshd. No se toca el cortafuegos para no cortar el SSH."
  bien "SSH escucha en el puerto $(echo "$ssh" | tr '\n' ' ')y se queda abierto para todo el mundo."
  local otros
  otros="$(otros_puertos_abiertos)"
  if [[ -n "$otros" ]]; then
    aviso "Estos puertos escuchan hacia fuera y el cortafuegos los dejará cerrados."
    echo "$otros" | sed 's/^/      /'
  fi
  if command -v docker >/dev/null 2>&1; then
    # Docker abre sus puertos por su cuenta, sin pasar por ufw. Por eso la API solo debe publicarse en 127.0.0.1
    if docker ps --format '{{.Names}} {{.Ports}}' | grep -E '0\.0\.0\.0:|\[::\]:|:::' ; then
      aviso "Algún contenedor publica puertos hacia fuera. ufw no los protege, revisa docker-compose.yml."
    else
      bien "Ningún contenedor publica puertos hacia fuera."
    fi
  fi
  echo "  Estado actual de ufw"
  ufw status verbose 2>/dev/null | sed 's/^/      /' || true
  # Una regla que abre 80 o 443 a todo el mundo y que el script no sabe quitar deja el cortafuegos sin efecto.
  # Hay que quitarla a mano antes. Solo se ven si ufw está activo, las demás se buscan otra vez al terminar
  local amplias
  amplias="$(reglas_web_abiertas | while IFS= read -r r; do regla_que_se_quita "$r" || echo "$r"; done)"
  if [[ -n "$amplias" ]]; then
    echo "$amplias" | sed 's/^/      /'
    if (( APLICAR )); then
      parar "Estas reglas abren 80 o 443 a todo el mundo. Quítalas con 'ufw status numbered' y 'ufw delete NÚMERO' y repite. No se ha cambiado nada."
    fi
    aviso "Estas reglas abren 80 o 443 a todo el mundo y el script no las quita. Con --aplicar pararía aquí."
  fi

  local COPIA_UFW=""
  guardar_ufw

  titulo "Reglas, primero las que dejan entrar"
  local puerto rango
  for puerto in $ssh; do
    ejecutar ufw allow "$puerto/tcp" comment 'SSH'
  done
  for rango in $(rangos); do
    ejecutar ufw allow proto tcp from "$rango" to any port 80,443 comment 'Cloudflare'
  done
  titulo "Quitar las reglas que abren 80 y 443 a todo el mundo, si las hay"
  for regla in 80 443 80/tcp 443/tcp http https; do
    printf '  + %s\n' "ufw delete allow $regla"
    if (( APLICAR )); then ufw delete allow "$regla" >/dev/null 2>&1 || true; fi
  done
  titulo "Cerrar lo demás y activar"
  ejecutar ufw default deny incoming
  ejecutar ufw default allow outgoing
  ejecutar ufw --force enable

  (( APLICAR )) || { echo; echo "Modo de prueba, no se ha cambiado nada. Para hacerlo, añade --aplicar."; return; }
  sleep 3
  titulo "Comprobaciones"
  if salud_por_cloudflare; then
    bien "https://$DOMINIO/health contesta a través de Cloudflare."
  else
    aviso "https://$DOMINIO/health no contesta a través de Cloudflare. Se vuelve al estado anterior de ufw."
    restaurar_ufw "$COPIA_UFW" || true
    parar "Revisa las reglas con 'ufw status numbered' antes de repetirlo."
  fi
  ufw status verbose | sed 's/^/      /'
  amplias="$(reglas_web_abiertas)"
  if [[ -n "$amplias" ]]; then
    echo "$amplias" | sed 's/^/      /'
    parar "La API sigue funcionando, pero estas reglas abren todavía 80 o 443 a todo el mundo. Quítalas con 'ufw status numbered' y 'ufw delete NÚMERO'."
  fi
  bien "Ninguna regla abre 80 o 443 a todo el mundo."
  echo
  echo "Desde otro equipo, la llamada directa a la IP tiene que quedarse sin respuesta hasta agotar el tiempo."
  echo "  curl -sS --max-time 10 --resolve $DOMINIO:443:$(ip_publica) https://$DOMINIO/health"
}

# Vuelve al estado de ufw de antes de la primera vez que se aplicó el paso del cortafuegos (la copia más antigua).
# Si se ha repetido el paso, por ejemplo al cambiar los rangos, las copias siguientes ya llevan reglas de Cloudflare
orden_deshacer_cortafuegos() {
  exigir_root
  local copia
  copia="$(ls -1d "$COPIAS_UFW"/*/ 2>/dev/null | sort | head -n1 || true)"
  copia="${copia%/}"
  if [[ -n "$copia" ]]; then
    titulo "Volver al estado de ufw de antes ($copia)"
    restaurar_ufw "$copia"
  else
    titulo "Desactivar ufw"
    aviso "No hay ninguna copia en $COPIAS_UFW, así que solo se puede desactivar ufw."
    # Las reglas se quedan guardadas. Con 'ufw enable' vuelven tal cual
    ejecutar ufw disable
  fi
  (( APLICAR )) || echo "Modo de prueba, no se ha cambiado nada. Para hacerlo, añade --aplicar."
}

# ---------------------------------------------------------------------------------------------------------- Rangos

# Para el cortafuegos de Hetzner Cloud, si se quiere poner además de ufw (ver DEPLOY.md). Se puede lanzar en cualquier
# equipo con el repo, no cambia nada
orden_rangos() {
  comprobar_rangos >&2
  rangos
}

# ------------------------------------------------------------------------------------------------------- Comprobar

orden_comprobar() {
  titulo "Caddy"
  comprobar_version_caddy
  comprobar_servicio_caddy
  local destino
  destino="$(caddyfile_sistema)"
  if [[ -f "$destino" ]] && cmp -s "$destino" "$CADDYFILE_REPO"; then
    bien "$destino es igual que el Caddyfile del repo."
  else
    aviso "$destino no es igual que el Caddyfile del repo (con 'caddy' se instala)."
  fi
  validar_caddyfile_repo
  titulo "Rangos de Cloudflare"
  comprobar_rangos
  titulo "Cortafuegos"
  if command -v ufw >/dev/null 2>&1; then
    ufw status verbose 2>/dev/null | sed 's/^/      /' || aviso "ufw status necesita root."
  else
    aviso "ufw no está instalado."
  fi
  echo "  Puertos que escuchan hacia fuera"
  ss -Htulnp 2>/dev/null | awk '{print $1, $5, $7}' \
    | grep -vE ' (127\.[0-9.]+|\[::1\]|\[::ffff:127\.[0-9.]+\])(%[a-z0-9]+)?:[0-9]+ ' | sed 's/^/      /' || true
  titulo "API"
  if salud_por_cloudflare; then bien "https://$DOMINIO/health contesta a través de Cloudflare."
  else aviso "https://$DOMINIO/health no contesta a través de Cloudflare."; fi
  comprobar_corte_por_ip
}

case "$ORDEN" in
  comprobar) orden_comprobar ;;
  caddy) orden_caddy ;;
  cortafuegos) orden_cortafuegos ;;
  deshacer-caddy) orden_deshacer_caddy ;;
  deshacer-cortafuegos) orden_deshacer_cortafuegos ;;
  rangos) orden_rangos ;;
  *) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

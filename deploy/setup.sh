#!/usr/bin/env bash
# One-time install of the web visualizer on dikarya.us (Ubuntu, nginx, certbot).
# Copy it to the server and run it as root:
#   scp deploy/setup.sh dikarya.us:inatviz-setup.sh
#   ssh -t dikarya.us sudo bash inatviz-setup.sh [--branch main] [--data DIR]
# --data names a directory holding observations.parquet and taxonomy.parquet
# (rsync them there from a workstation first). Safe to run again: each step
# skips what is already in place. Later updates go through deploy/deploy.sh.
set -euo pipefail

DOMAIN=visualizer.dikarya.us
APP_DIR=/opt/inat-visualizer
DATA_DIR=/srv/inat-visualizer/data
REPO=https://github.com/AlanRockefeller/inat.visualizer.py
VHOST=/etc/nginx/sites-enabled/$DOMAIN.conf
WEBROOT=/var/www/letsencrypt
BRANCH=main
DATA_SRC=

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --data) DATA_SRC="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "Run as root (sudo)." >&2; exit 1; }
for tool in git python3 nginx certbot curl; do
  command -v "$tool" >/dev/null || { echo "Missing $tool." >&2; exit 1; }
done

echo "== User, code and Python environment"
id inatviz >/dev/null 2>&1 ||
  useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin inatviz
mkdir -p "$APP_DIR" "$DATA_DIR"
chown inatviz: "$APP_DIR" "$DATA_DIR"
if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u inatviz git clone --branch "$BRANCH" "$REPO" "$APP_DIR"
else
  sudo -u inatviz git -C "$APP_DIR" pull --ff-only
fi
cd "$APP_DIR"
# A branch from before the subdomain layout has no setup.sh and an nginx file
# that is not a vhost; installing its units and nginx file would fail late.
if [ ! -f deploy/setup.sh ] || ! grep -q "server_name $DOMAIN;" deploy/nginx-visualizer.conf; then
  echo "The $(sudo -u inatviz git rev-parse --abbrev-ref HEAD) checkout in $APP_DIR predates the" \
       "$DOMAIN layout. Merge it into that branch, or switch the checkout:" >&2
  echo "  sudo -u inatviz git -C $APP_DIR switch BRANCH" >&2
  exit 1
fi
[ -x .venv-web/bin/python ] || sudo -u inatviz python3 -m venv .venv-web
sudo -u inatviz .venv-web/bin/pip install -q -r requirements-web.txt

echo "== Data"
if [ -n "$DATA_SRC" ]; then
  for name in taxonomy.parquet observations.parquet; do
    install -o inatviz -g inatviz -m 644 "$DATA_SRC/$name" "$DATA_DIR/$name.new"
    mv "$DATA_DIR/$name.new" "$DATA_DIR/$name"
  done
fi
for name in taxonomy.parquet observations.parquet; do
  [ -s "$DATA_DIR/$name" ] || { echo "$DATA_DIR/$name is missing; pass --data DIR." >&2; exit 1; }
done

echo "== Service"
install -m 644 deploy/inat-visualizer.service deploy/inat-visualizer-update.service \
  deploy/inat-visualizer-update.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable inat-visualizer
systemctl restart inat-visualizer
# The weekly rebuild downloads a ~30 GB archive and needs far more space to
# extract it than this server has free, so its timer stays disabled here.
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8765/api/stats >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://127.0.0.1:8765/api/stats >/dev/null ||
  { systemctl --no-pager status inat-visualizer; exit 1; }

echo "== TLS certificate"
mkdir -p /var/cache/nginx/osm-tiles
chown www-data: /var/cache/nginx/osm-tiles
if [ ! -e "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  # Serve only the ACME challenge until the certificate exists.
  [ -e "$VHOST" ] && { echo "$VHOST exists but the certificate does not." >&2; exit 1; }
  cat >"$VHOST" <<CONF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    location ^~ /.well-known/acme-challenge/ {
        root $WEBROOT;
        default_type text/plain;
        try_files \$uri =404;
    }
}
CONF
  if ! nginx -t || ! systemctl reload nginx; then
    rm -f "$VHOST"; echo "nginx rejected the bootstrap vhost; removed it." >&2; exit 1
  fi
  if ! certbot certonly --webroot -w "$WEBROOT" -d "$DOMAIN" \
      --non-interactive --agree-tos --deploy-hook "systemctl reload nginx"; then
    rm -f "$VHOST"; systemctl reload nginx; exit 1
  fi
fi

echo "== nginx vhost"
backup=
if [ -e "$VHOST" ]; then backup=$(mktemp); cp -p "$VHOST" "$backup"; fi
install -m 644 deploy/nginx-visualizer.conf "$VHOST"
if ! nginx -t || ! systemctl reload nginx; then
  if [ -n "$backup" ]; then mv "$backup" "$VHOST"; else rm -f "$VHOST"; fi
  systemctl reload nginx || true
  echo "nginx rejected $VHOST; put the previous file back." >&2
  exit 1
fi
[ -n "$backup" ] && rm -f "$backup"

echo "== Check"
curl -fsS "https://$DOMAIN/api/stats"; echo
curl -fsS -o /dev/null -D - "https://$DOMAIN/tiles/2/1/1.png" | grep -i '^x-cache-status' || true
echo "Done: https://$DOMAIN/"

#!/usr/bin/env bash
# Update the web visualizer on the server and smoke-test it.
#   deploy/deploy.sh [ssh-host] [public-url]
#   defaults: dikarya.us  https://dikarya.us/visualizer
# Deploys whatever is pushed on the branch the server checkout tracks, installs
# any systemd or nginx files that changed in deploy/, reloads what they affect,
# restarts the service and checks it answers at the public URL.
set -euo pipefail

HOST="${1:-dikarya.us}"
URL="${2:-https://dikarya.us/visualizer}"
APP_DIR=/opt/inat-visualizer

ssh "$HOST" "APP_DIR=$APP_DIR bash -s" <<'REMOTE'
set -euo pipefail
cd "$APP_DIR"
sudo -u inatviz git pull --ff-only
sudo -u inatviz .venv-web/bin/pip install -q -r requirements-web.txt

# git pull does not touch the installed copies of the systemd units and nginx
# files (see deploy/README.md for where they live), so install the ones that
# changed and reload the daemon that reads them.
units_changed=0
backups=()
while read -r source installed; do
  if ! sudo cmp -s "$source" "$installed"; then
    echo "Installing $installed from $source"
    case "$installed" in
      /etc/systemd/*) units_changed=1 ;;
      /etc/nginx/*)
        # Keep the previous copy so a snippet that fails `nginx -t` can be
        # put back instead of waiting to break the next nginx restart.
        if sudo test -e "$installed"; then
          sudo cp -p "$installed" "$installed.deploy-previous"
          backups+=("$installed")
        fi
        ;;
    esac
    sudo install -m 644 "$source" "$installed"
  fi
done <<'FILES'
deploy/inat-visualizer.service /etc/systemd/system/inat-visualizer.service
deploy/inat-visualizer-update.service /etc/systemd/system/inat-visualizer-update.service
deploy/inat-visualizer-update.timer /etc/systemd/system/inat-visualizer-update.timer
deploy/nginx-visualizer-http.conf /etc/nginx/conf.d/inat-visualizer.conf
deploy/nginx-visualizer.conf /etc/nginx/snippets/inat-visualizer.conf
FILES
if [ "$units_changed" -eq 1 ]; then
  sudo systemctl daemon-reload
  sudo systemctl restart inat-visualizer-update.timer
fi
# Validate and reload nginx on every run, not only when a file changed here:
# a previous run may have installed a file without managing to reload it.
if ! sudo nginx -t; then
  for installed in "${backups[@]}"; do
    echo "Restoring $installed" >&2
    sudo mv "$installed.deploy-previous" "$installed"
  done
  echo "ERROR: nginx rejected the configuration; previous nginx files restored." >&2
  exit 1
fi
for installed in "${backups[@]}"; do
  sudo rm -f "$installed.deploy-previous"
done
sudo systemctl reload nginx

sudo systemctl restart inat-visualizer
healthy=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/api/stats >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
sudo systemctl --no-pager --lines=5 status inat-visualizer || true
if [ "$healthy" -ne 1 ]; then
  echo "ERROR: the service did not answer on 127.0.0.1:8765 within 30 seconds." >&2
  exit 1
fi
REMOTE

echo "Checking $URL ..."
curl -fsS "$URL/api/stats"
echo
curl -fsS -o /dev/null -w "index: %{http_code}\n" "$URL/"
curl -fsS -o /dev/null -D - "$URL/tiles/2/1/1.png" | grep -i '^x-cache-status' || true

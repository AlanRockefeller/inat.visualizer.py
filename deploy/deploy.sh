#!/usr/bin/env bash
# Update the web visualizer on the server and smoke-test it.
#   deploy/deploy.sh [ssh-host]        (default host: dikarya.us)
# Deploys whatever is pushed on the branch the server checkout tracks.
set -euo pipefail

HOST="${1:-dikarya.us}"
APP_DIR=/opt/inat-visualizer
URL="https://dikarya.us/visualizer"

# Remote step exit codes: 0 ok, 2 ok but installed config files are stale.
remote_status=0
ssh "$HOST" "APP_DIR=$APP_DIR bash -s" <<'REMOTE' || remote_status=$?
set -euo pipefail
cd "$APP_DIR"
sudo -u inatviz git pull --ff-only
sudo -u inatviz .venv-web/bin/pip install -q -r requirements-web.txt
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
# git pull does not touch the installed copies of the systemd units and nginx
# files, so warn when the repository versions differ from what is running.
stale=0
while read -r source installed; do
  if ! sudo cmp -s "$source" "$installed"; then
    echo "WARNING: $installed differs from $source; install it and reload (see deploy/README.md)." >&2
    stale=1
  fi
done <<'FILES'
deploy/inat-visualizer.service /etc/systemd/system/inat-visualizer.service
deploy/inat-visualizer-update.service /etc/systemd/system/inat-visualizer-update.service
deploy/inat-visualizer-update.timer /etc/systemd/system/inat-visualizer-update.timer
deploy/nginx-visualizer-http.conf /etc/nginx/conf.d/inat-visualizer.conf
deploy/nginx-visualizer.conf /etc/nginx/snippets/inat-visualizer.conf
FILES
[ "$stale" -eq 0 ] || exit 2
REMOTE
if [ "$remote_status" -ne 0 ] && [ "$remote_status" -ne 2 ]; then
  exit "$remote_status"
fi

echo "Checking $URL ..."
curl -fsS "$URL/api/stats"
echo
curl -fsS -o /dev/null -w "index: %{http_code}\n" "$URL/"
curl -fsS -o /dev/null -D - "$URL/tiles/2/1/1.png" | grep -i '^x-cache-status' || true
if [ "$remote_status" -eq 2 ]; then
  echo "Deployed, but installed configuration files are out of date (see warnings above)." >&2
  exit 1
fi

#!/usr/bin/env bash
# Update the web visualizer on the server and smoke-test it.
#   deploy/deploy.sh [ssh-host]        (default host: dikarya.us)
# Deploys whatever is pushed on the branch the server checkout tracks.
set -euo pipefail

HOST="${1:-dikarya.us}"
APP_DIR=/opt/inat-visualizer
URL="https://dikarya.us/visualizer"

ssh "$HOST" "set -e
  cd $APP_DIR
  sudo -u inatviz git pull --ff-only
  sudo -u inatviz .venv-web/bin/pip install -q -r requirements-web.txt
  sudo systemctl restart inat-visualizer
  for i in \$(seq 1 30); do
    curl -fsS http://127.0.0.1:8765/api/stats >/dev/null 2>&1 && break
    sleep 1
  done
  sudo systemctl --no-pager --lines=5 status inat-visualizer"

echo "Checking $URL ..."
curl -fsS "$URL/api/stats"
echo
curl -fsS -o /dev/null -w "index: %{http_code}\n" "$URL/"
curl -fsS -o /dev/null -D - "$URL/tiles/2/1/1.png" | grep -i '^x-cache-status' || true

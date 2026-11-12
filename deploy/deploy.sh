#!/usr/bin/env bash
# Update the web visualizer on the server and smoke-test it.
#   deploy/deploy.sh [ssh-host] [public-url]
#   defaults: dikarya.us  https://visualizer.dikarya.us
# First install: deploy/setup.sh.
# Deploys whatever is pushed on the branch the server checkout tracks, installs
# any systemd or nginx files that changed in deploy/, reloads what they affect,
# restarts the service and checks it answers at the public URL.
set -euo pipefail

HOST="${1:-dikarya.us}"
APP_DIR=/opt/inat-visualizer
if [ -n "${2:-}" ]; then
  URL="$2"
elif [ "$HOST" = "dikarya.us" ]; then
  URL="https://visualizer.dikarya.us"
else
  # Smoke-testing dikarya.us would say nothing about a deployment elsewhere.
  echo "usage: deploy/deploy.sh [ssh-host] [public-url]" >&2
  echo "Give the public URL when deploying to a host other than dikarya.us." >&2
  exit 2
fi

# Passed as an argument rather than on stdin so `ssh -t` gives sudo a terminal
# to ask for the password (once; sudo caches it for the rest of the run).
remote_script=$(cat <<'REMOTE'
set -euo pipefail
sudo -v
cd "$APP_DIR"
sudo -u inatviz git pull --ff-only
sudo -u inatviz .venv-web/bin/pip install -q -r requirements-web.txt

# git pull does not touch the installed copies of the systemd units and nginx
# files (see deploy/README.md for where they live), so install the ones that
# changed and reload the daemon that reads them.
units_changed=0
replaced=()   # nginx files replaced this run; the previous copy is in $backups
added=()      # nginx files that did not exist before this run
# Not beside the originals: nginx loads every file in sites-enabled.
backups=$(mktemp -d)
while read -r source installed; do
  if ! sudo cmp -s "$source" "$installed"; then
    echo "Installing $installed from $source"
    case "$installed" in
      /etc/systemd/*) units_changed=1 ;;
      /etc/nginx/*)
        # Remember what was there so a file that fails `nginx -t` or the
        # reload can be undone instead of breaking the next nginx restart.
        if sudo test -e "$installed"; then
          sudo cp -p "$installed" "$backups/$(basename "$installed")"
          replaced+=("$installed")
        else
          added+=("$installed")
        fi
        ;;
    esac
    sudo install -m 644 "$source" "$installed"
  fi
done <<'FILES'
deploy/inat-visualizer.service /etc/systemd/system/inat-visualizer.service
deploy/inat-visualizer-update.service /etc/systemd/system/inat-visualizer-update.service
deploy/inat-visualizer-update.timer /etc/systemd/system/inat-visualizer-update.timer
deploy/nginx-visualizer.conf /etc/nginx/sites-enabled/visualizer.dikarya.us.conf
FILES
if [ "$units_changed" -eq 1 ]; then
  sudo systemctl daemon-reload
  # try-restart: the timer is left disabled where the disk cannot hold the
  # rebuild (see deploy/README.md), and a deployment must not start it.
  sudo systemctl try-restart inat-visualizer-update.timer
fi
undo_nginx_files() {
  for installed in "${replaced[@]}"; do
    echo "Restoring $installed" >&2
    sudo mv "$backups/$(basename "$installed")" "$installed"
  done
  for installed in "${added[@]}"; do
    echo "Removing $installed" >&2
    sudo rm -f "$installed"
  done
}
# Validate and reload nginx on every run, not only when a file changed here:
# a previous run may have installed a file without managing to reload it.
if ! sudo nginx -t; then
  undo_nginx_files
  echo "ERROR: nginx rejected the configuration; the nginx files are back as they were." >&2
  exit 1
fi
if ! sudo systemctl reload nginx; then
  undo_nginx_files
  echo "ERROR: nginx did not reload; the nginx files are back as they were." >&2
  exit 1
fi
sudo rm -rf "$backups"

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
)
ssh -t "$HOST" "APP_DIR=$APP_DIR bash -c $(printf '%q' "$remote_script")"

echo "Checking $URL ..."
curl -fsS "$URL/api/stats"
echo
curl -fsS -o /dev/null -w "index: %{http_code}\n" "$URL/"
curl -fsS -o /dev/null -D - "$URL/tiles/2/1/1.png" | grep -i '^x-cache-status' || true

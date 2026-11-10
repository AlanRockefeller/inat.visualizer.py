# Deploying the web version to dikarya.us

The web app is a FastAPI server (`webapp/server.py`) behind nginx at
`https://dikarya.us/visualizer/`. It runs the same DuckDB queries as the desktop
app against `observations.parquet` and `taxonomy.parquet`. The browser looks up
taxon and place names directly from `api.inaturalist.org`. nginx caches
OpenStreetMap tiles on disk as they are requested.

```
browser ──https──> nginx ─┬─ /visualizer/        → uvicorn 127.0.0.1:8765 (static page)
                          ├─ /visualizer/api/    → uvicorn (rate limited)
                          └─ /visualizer/tiles/  → tile.openstreetmap.org (disk cache)
```

## One-time setup (on the server)

```bash
# 1. System user, code, and Python environment
sudo apt install -y python3-venv git
sudo useradd --system --home /opt/inat-visualizer --shell /usr/sbin/nologin inatviz
sudo mkdir -p /opt/inat-visualizer /srv/inat-visualizer/data
sudo chown inatviz: /opt/inat-visualizer /srv/inat-visualizer/data
sudo -u inatviz git clone https://github.com/AlanRockefeller/inat.visualizer.py /opt/inat-visualizer
cd /opt/inat-visualizer
sudo -u inatviz python3 -m venv .venv-web
sudo -u inatviz .venv-web/bin/pip install -r requirements-web.txt

# 2. Data. Either copy the existing files from your workstation:
#      rsync -P observations.parquet taxonomy.parquet dikarya.us:/tmp/
#      sudo mv /tmp/*.parquet /srv/inat-visualizer/data/ && sudo chown inatviz: /srv/inat-visualizer/data/*
#    or build them on the server (downloads several GB from iNaturalist):
sudo -u inatviz .venv-web/bin/python update_database.py --output-dir /srv/inat-visualizer/data

# 3. Services
sudo cp deploy/inat-visualizer.service deploy/inat-visualizer-update.service \
        deploy/inat-visualizer-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now inat-visualizer inat-visualizer-update.timer
curl -s http://127.0.0.1:8765/api/stats

# 4. nginx
sudo cp deploy/nginx-visualizer-http.conf /etc/nginx/conf.d/inat-visualizer.conf
sudo cp deploy/nginx-visualizer.conf /etc/nginx/snippets/inat-visualizer.conf
sudo mkdir -p /var/cache/nginx/osm-tiles && sudo chown www-data: /var/cache/nginx/osm-tiles
#    Add this line inside the existing dikarya.us `server { listen 443 ssl; ... }` block
#    (e.g. /etc/nginx/sites-enabled/dikarya.us):
#        include snippets/inat-visualizer.conf;
sudo nginx -t && sudo systemctl reload nginx
```

The existing TLS certificate for dikarya.us already covers the `/visualizer/` path.

## Updating the code

Push your changes, then run `deploy/deploy.sh` from your workstation. It pulls,
installs dependencies, installs any changed files from `deploy/` into
`/etc/systemd/system` and `/etc/nginx` (reloading systemd and nginx as needed),
restarts the service, and smoke-tests the public URL.

## Database updates

`inat-visualizer-update.timer` rebuilds the Parquet files every Sunday morning
with `update_database.py`. Each new file is swapped in atomically (taxonomy
first, then observations, so a query in the second or so between the two swaps
can pair the new taxonomy with the old observations), and the server clears its
caches when their size or mtime changes, so no restart is needed. Results
computed while a swap lands are not cached. To run it now: `sudo systemctl start inat-visualizer-update` and follow
it with `journalctl -fu inat-visualizer-update`.

## Tile cache

- Tiles are cached **on demand only**. Do not bulk-download or pre-seed tiles:
  the [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/)
  forbids it and OSM blocks servers that do it.
- The cache is capped by `max_size` (30 GB) in
  `/etc/nginx/conf.d/inat-visualizer.conf`. Tiles unused for 90 days are
  evicted. Expired tiles are refreshed with conditional requests, which are
  cheap `304 Not Modified` replies when nothing changed.
- Check the size with `du -sh /var/cache/nginx/osm-tiles`. To see whether a
  tile came from the cache, run
  `curl -sI https://dikarya.us/visualizer/tiles/3/1/3.png | grep -i x-cache`.

## Tuning (environment variables in the service unit)

| Variable | Default | Meaning |
|---|---|---|
| `INAT_DATA_DIR` | `.` | Directory holding the two Parquet files |
| `INAT_MAX_CONCURRENT_QUERIES` | `2` | DuckDB queries running at once; others wait |
| `INAT_DUCKDB_MEMORY` | `4GB` (unit sets `3GB`) | DuckDB memory limit per query |
| `INAT_DUCKDB_THREADS` | `4` | DuckDB threads per query |
| `INAT_QUERY_TIMEOUT` | `90` | Seconds before a query is interrupted (HTTP 504) |
| `INAT_MAX_CSV_ROWS` | `500000` | Largest CSV download |
| `INAT_RESULT_CACHE_ENTRIES` | `256` | Cached query results |
| `INAT_RESULT_CACHE_MB` | `64` | Budget for cached results, measured as JSON size |

## Running locally

```bash
python -m venv .venv-web && .venv-web/bin/pip install -r requirements-web.txt
INAT_DATA_DIR=. .venv-web/bin/uvicorn webapp.server:app --port 8765
# open http://127.0.0.1:8765/  (tiles fall back to OSM directly without nginx)
```

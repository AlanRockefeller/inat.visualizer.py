# Deploying the web version to dikarya.us

The web app is a FastAPI server (`webapp/server.py`) behind nginx at
`https://visualizer.dikarya.us/`. It runs the same DuckDB queries as the desktop
app against `observations.parquet` and `taxonomy.parquet`. The browser looks up
taxon and place names directly from `api.inaturalist.org`. nginx caches
OpenStreetMap tiles on disk as they are requested.

```
browser ──https──> nginx ─┬─ /        → uvicorn 127.0.0.1:8765 (static page)
                          ├─ /api/    → uvicorn (rate limited)
                          └─ /tiles/  → tile.openstreetmap.org (disk cache)
```

| Repository file | Installed as |
|---|---|
| `deploy/inat-visualizer.service` | `/etc/systemd/system/inat-visualizer.service` |
| `deploy/inat-visualizer-update.service` | `/etc/systemd/system/inat-visualizer-update.service` |
| `deploy/inat-visualizer-update.timer` | `/etc/systemd/system/inat-visualizer-update.timer` (not enabled) |
| `deploy/nginx-visualizer.conf` | `/etc/nginx/sites-enabled/visualizer.dikarya.us.conf` |

The server's `nginx.conf` includes `sites-enabled/*` but not `conf.d/`, so the
vhost file also carries the http-level tile cache and rate-limit zones.

## One-time setup

DNS for `visualizer.dikarya.us` must point at the server. Then, from a
workstation that has the two Parquet files:

```bash
rsync -P observations.parquet taxonomy.parquet dikarya.us:inatviz-data/
scp deploy/setup.sh dikarya.us:inatviz-setup.sh
ssh -t dikarya.us 'sudo bash inatviz-setup.sh --data "$HOME/inatviz-data"'
```

`setup.sh` creates the `inatviz` user, clones the repository into
`/opt/inat-visualizer` (`--branch` picks the branch, default `main`), builds the
Python environment, installs the data into `/srv/inat-visualizer/data`, installs
and starts the service, obtains a Let's Encrypt certificate through the shared
`/var/www/letsencrypt` webroot, installs the vhost and smoke-tests the site. It
can be run again; finished steps are skipped. The Parquet copies in
`~/inatviz-data` can be deleted afterwards.

## Updating the code

Push your changes to the branch the server checkout tracks, then run
`deploy/deploy.sh` from your workstation. sudo asks for your password once. The
script pulls, installs dependencies, installs any changed files from `deploy/`
(table above), tests and reloads nginx, restarts the service, and smoke-tests
the public URL. If nginx rejects a changed vhost, the previous file is put back.

## Database updates

The data is built on a workstation and copied to the server, because the
server's disk cannot hold iNaturalist's ~30 GB export while it is processed.
`inat-visualizer-update.timer` (a server-side rebuild) stays disabled on
dikarya.us. One command does the whole update:

```bash
python update_database.py --upload dikarya.us
```

1. It rebuilds from the GBIF export only when iNaturalist has published a new
   one since the last build (`--force` rebuilds anyway).
2. If `backfill/` exists, it fetches the research-grade observations the export
   leaves out (see below), up to the day's API budget.
3. It merges the two into `observations.parquet`, the export winning for any
   observation in both.
4. `--upload` copies both Parquet files to the server and swaps them in (sudo
   asks for your password once). Taxonomy goes first, then observations; each
   swap is atomic and the server clears its caches when the files change.

### Observations without an open license

iNaturalist's export only includes observations licensed CC0, CC-BY or
CC-BY-NC: about 175 of 227 million research-grade observations.
`backfill_observations.py` fetches the rest (about 45 million unlicensed and 7
million CC-BY-SA/-ND) from the API, keeping only the columns the graphs count:

```bash
python backfill_observations.py            # runs until today's 10,000 requests are spent
python backfill_observations.py --status   # progress
```

The first pass takes about four weeks at iNat's recommended pace (one request
a second, 10,000 a day). It saves its place every few pages, so it can be
stopped with Ctrl-C and restarted, and `update_database.py` merges whatever has
been fetched so far. After the first pass, each run fetches only observations
changed since the previous one, whatever its license (iNat updates about 3.5
million observations a week, so about 18,000 requests: two days of budget). It drops observations that lost research grade, and those that moved
to a license the export carries, since the export counts them. Observations
deleted on iNaturalist stay counted: the API only lists deletions to their
owners. Only one backfill or update can use the store at a time.

Once backfilled rows are merged, the site's iNaturalist links drop their
license filter, so they show the same observations the graph counts.

## Tile cache

- Tiles are cached **on demand only**. Do not bulk-download or pre-seed tiles:
  the [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/)
  forbids it and OSM blocks servers that do it.
- The cache is capped by `max_size` (5 GB) in
  `/etc/nginx/sites-enabled/visualizer.dikarya.us.conf`. Tiles unused for 90 days are
  evicted. Expired tiles are refreshed with conditional requests, which are
  cheap `304 Not Modified` replies when nothing changed.
- Check the size with `du -sh /var/cache/nginx/osm-tiles`. To see whether a
  tile came from the cache, run
  `curl -sI https://visualizer.dikarya.us/tiles/3/1/3.png | grep -i x-cache`.

## Tuning (environment variables in the service unit)

| Variable | Default | Meaning |
|---|---|---|
| `INAT_DATA_DIR` | `.` | Directory holding the two Parquet files |
| `INAT_MAX_CONCURRENT_QUERIES` | `2` | DuckDB queries running at once; others wait |
| `INAT_DUCKDB_MEMORY` | `4GB` (unit sets `3GB`) | DuckDB memory limit per query |
| `INAT_DUCKDB_THREADS` | `4` (unit sets `2`) | DuckDB threads per query |
| `INAT_QUERY_TIMEOUT` | `90` | Seconds before a query is interrupted (HTTP 504) |
| `INAT_MAX_CSV_ROWS` | `500000` | Largest CSV download |
| `INAT_DUCKDB_EXTENSION_DIR` | DuckDB's default (unit sets `/opt/inat-visualizer/.duckdb-extensions`) | Where the spatial extension for region searches is installed; `deploy.sh` and `setup.sh` install it |
| `INAT_RESULT_CACHE_ENTRIES` | `256` | Cached query results |
| `INAT_RESULT_CACHE_MB` | `64` | Budget for cached results, measured as JSON size |

## Running locally

```bash
python -m venv .venv-web && .venv-web/bin/pip install -r requirements-web.txt
INAT_DATA_DIR=. .venv-web/bin/uvicorn webapp.server:app --port 8765
# open http://127.0.0.1:8765/  (tiles fall back to OSM directly without nginx)
```

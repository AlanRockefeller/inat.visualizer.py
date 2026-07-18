# iNaturalist Seasonal Visualizer

# By Alan Rockefeller - July 17, 2026

Current source version: **1.0.2**. See [CHANGELOG.md](CHANGELOG.md) for release
details.

A desktop GUI app for exploring **seasonal patterns in iNaturalist observations** within a geographic radius. Search by organism (anything from a genus/species to higher taxa like _Agaricales_), choose a date range, and plot observation frequency by **day**, **week**, or **month** of the year.

The app supports two search modes:

- **Graph with local data (fast, offline-ish):** Queries a local `observations.parquet` file using DuckDB (recommended).
- **Graph with live iNat data (online):** Queries the iNaturalist API via `pyinaturalist` (slower and rate-limited, but works without local data).

It also includes an **interactive map dialog** (OpenStreetMap tiles) to set coordinates and radius visually, plus export options for both graphs and data.

---

## Features

- **Interactive GUI** (PyQt6 + Matplotlib) with a sidebar of search controls and a live plot.
- **Local database mode** using DuckDB against `observations.parquet` for fast queries.
- **Taxonomy expansion** using `taxonomy.parquet` to include _all descendant taxa_ of a selected organism (recursive query).
- **Taxon cache** (`taxon_cache.json`) to avoid repeated API lookups and repeated descendant expansion.
- **Interactive map picker** for latitude/longitude + radius:
  - OpenStreetMap tile fetching
  - **RAM LRU cache** and **disk cache** with pruning
  - Pan/zoom controls
- **Progress widget** for long operations:
  - Download progress for required Parquet files
  - API pagination and rate limiting feedback
  - Local query “estimate” + completion messaging
- **Theme and appearance controls**
  - Dark mode by default, with a persistent light/dark mode toggle
  - Graph color, graph background color, window background color
  - Adjustable app font and graph font sizes (saved via QSettings)
- **Export**
  - Export current plot as JPG/PNG with metadata
  - Export observation data to CSV

---

## Screenshots

- Optional splash screen: `splash_screen.jpg` in the current working directory (CWD).

---

## Download (prebuilt apps for Windows & macOS)

The easiest way to get started — no Python required. Grab the latest build from the
[**Releases**](https://github.com/AlanRockefeller/inat.visualizer.py/releases) page:

- **Windows:** download `iNat-Seasonal-Visualizer.exe` and double-click it.
- **macOS (Apple Silicon):** download and unzip
  `iNat-Seasonal-Visualizer-macOS-Apple-Silicon.zip`.
- **macOS (Intel):** download and unzip
  `iNat-Seasonal-Visualizer-macOS-Intel.zip`.
  The first time, right-click the app and choose **Open** (it is unsigned).
- **Linux:** download and extract `iNat-Seasonal-Visualizer-Linux.tar.gz`, then run the binary.

On first launch the app offers to download the approximately 1 GB local database
into its per-user application data directory. The download is optional; without
it, searches use the online iNaturalist API instead of the local-data graph.

### Creating a release

After updating `inat_visualizer_version.py` and `CHANGELOG.md`, commit and push
`main`, then run:

```bash
./build-release.sh
```

The script reads the version from the codebase, validates that local `main`
matches `origin/main`, and pushes the matching version tag. That tag triggers
the GitHub workflow that builds, smoke-tests, and publishes every platform
artifact. Use `./build-release.sh --dry-run` to perform the checks without
creating a tag.

---

## Requirements (running from source)

The app is cross-platform (Windows, macOS, Linux) and targets **Python 3.12**.

- Qt backend: PyQt6
- Matplotlib backend: `QtAgg`

On **Linux** only, Qt runs under XWayland to avoid a Wayland protocol crash
(the app sets `QT_QPA_PLATFORM=xcb` automatically) and needs a couple of system libs:

```bash
sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0
```

Windows and macOS need no extra system packages.

---

## Installation (from source)

### Option A — pip (any OS)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda create -n inat_env python=3.12
conda activate inat_env
pip install -r requirements.txt
```

> The program includes an environment self-check. Missing packages stop startup;
> version differences are logged as warnings but do not prevent the app from running.

---

## Data files (optional; enables local-data graphing and taxonomy expansion)

The application uses two Parquet files in its runtime data directory. Sizes
depend on the source snapshot; the July 2026 DWCA rebuild produced:

- `observations.parquet` (~1.6 GB)
- `taxonomy.parquet` (~6.2 MB)

The older hosted snapshots downloaded by the app are approximately 1.02 GB and
8.7 MB, respectively.

If either is missing at startup, the app offers two choices:

- **Download Local Database:** uses approximately 1 GB of disk space and enables
  faster local observation searches. The taxonomy file also expands higher taxa
  such as orders and families to their descendant species.
- **Use iNaturalist API Only:** starts immediately without the large download.
  Searches require an internet connection and may be slower or rate-limited.
  **Graph with local data** remains disabled until `observations.parquet` is
  installed.

If `observations.parquet` is already installed and only the much smaller taxonomy
file is missing, **Graph with local data** remains available; higher-taxon
expansion may be limited until the taxonomy file is downloaded.

Packaged apps store mutable files in these writable per-user locations:

- **macOS:** `~/Library/Application Support/iNat Seasonal Visualizer/`
- **Windows:** `%LOCALAPPDATA%\iNat Seasonal Visualizer\`
- **Linux:** `$XDG_DATA_HOME/iNat Seasonal Visualizer/`, or
  `~/.local/share/iNat Seasonal Visualizer/` when `XDG_DATA_HOME` is unset

Runs from source continue to use the current working directory.

The download URLs are:

- `https://images.mushroomobserver.org/observations.parquet`
- `https://images.mushroomobserver.org/taxonomy.parquet`

When the observation database is installed, startup uses a lightweight HEAD
request to compare the hosted `observations.parquet` `Content-Length` with its
installed file size. A different size means a coordinated database update is
available; the app then checks the companion taxonomy file, asks before
downloading the changed files, and atomically replaces each installed copy only
after its download completes. A matching observation file does not prompt,
regardless of `Last-Modified` timestamps or a different older taxonomy snapshot
on the server. API-only installations do not perform this update check.

### Rebuilding the local databases from iNaturalist

Run the repository's updater to rebuild both Parquet files from the current
iNaturalist Darwin Core archives:

```bash
./update_database.py
```

The script uses `gbif-observations-dwca.zip` from the repository directory when
it is already present (including a file downloaded separately with `wget`). If
it is absent, the script downloads it. It also downloads and rebuilds the
separate iNaturalist taxonomy archive required for higher-taxon searches.

Close the visualizer before running the update. The observation archive is
about 25 GB and its required CSV currently expands to more than 100 GB, so allow
roughly 140 GB of free space. The updater extracts only the two CSV members the
app needs; it does not retain the media or DNA extension data. New databases are
validated before they atomically replace the installed files. Extracted CSVs
and source archives are removed after success. Use `--keep-archives` to retain
the ZIP files, or `--help` for paths and other options.

### Expected columns

`observations.parquet` must contain:

- `eventDate`
- `decimalLatitude`
- `decimalLongitude`
- `taxonID`

If these columns are missing, local-data graphing will fail with an explanatory
error.

---

## Running

With your environment activated, run:

```bash
python visualizer.py
```

### Command-line options

```bash
python visualizer.py --lat 37.7749 --lon -122.4194 --radius 25 --scale-factor 1.5 --debug
```

Flags:

- `--lat` Latitude (default comes from saved settings)
- `--lon` Longitude (default comes from saved settings)
- `--radius` Radius in km (default comes from saved settings)
- `--scale-factor` Manual UI scale multiplier (useful for 4K/HiDPI)
- `--debug` Enable debug logging + extra console prints

Logs go to:

- Packaged app: `inat_visualizer.log` inside the per-user application data
  directory listed above
- Source run: `inat_visualizer.log` in the current working directory

---

## Using the app

1. **Set location**
   - Type latitude/longitude (or paste `"lat, lon"` into the latitude field)
   - Or click **🗺️ Map** to pick a point and radius interactively.

2. **Choose an organism**
   - Examples: `Boletus`, `Russula brevipes`, `Agaricales`
   - Leave blank to search all organisms.

3. **Optional: exclude a taxon**
   - Example: exclude `Boletus regineus` (also expands descendants)

4. **Pick a date range**
   - Default `Date From`: `2000-01-01`
   - Default `Date To`: auto-filled from the most recent date in `observations.parquet` (if available)

5. Choose view: **Daily / Weekly / Monthly**

6. Click:
   - **Graph with local data** (fast, when the optional database is installed)
   - **Graph with live iNat data** (online, rate-limited, and available without local data)

---

## Taxon ID caching

To reduce API calls, the app stores cached results in:

- `taxon_cache.json`

This cache includes:

- Name → taxon ID
- Name → list of descendant taxon IDs (`<name>_descendants`)

There is also optional support for a manual descendant file:

- `descendant_taxons.txt`

Format:

```text
Agaricales: 117159, 48723, 12345
```

If present, it can be used as a fallback when descendant expansion via `taxonomy.parquet` fails.

---

## Map tile caching

The map dialog fetches OpenStreetMap tiles and caches them:

- In RAM: LRU cache up to `MAX_CACHE_SIZE` tiles
- On disk: `tile_cache/` inside the runtime data directory, with pruning to
  ~`200 MB`

---

## Notes on iNaturalist API limits

Anonymous API usage can be rate-limited (HTTP 429 / 403). The app uses pagination and backoff, but large queries may still be slow. Local searches are better - they are much faster, and don't hit the API at all.

If you plan to do lots of API queries, you can increase limits by configuring credentials (if supported by your setup). The script references:

- `INATURALIST_APP_ID`
- `INATURALIST_APP_SECRET`

(Place them in `~/.bashrc` and restart your shell.)

---

## Troubleshooting

### Wayland / Qt crashes (Linux)

On Linux the app forces Qt onto XWayland (`QT_QPA_PLATFORM=xcb`) automatically.
If you still have rendering issues, ensure the required libs are installed:

```bash
sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0
```

You can override the platform plugin by setting `QT_QPA_PLATFORM` yourself before
launching. (This forcing does not apply on Windows or macOS, which use their
native Qt platform plugins.)

### Environment check fails

At startup the app verifies that the required Python packages are importable.
A missing package stops startup with an install hint
(`pip install -r requirements.txt`). Version differences from the tested set are
logged as warnings only and do not block the app.

---

## Project structure (runtime artifacts)

When you run the program, it may create the following files in its runtime data
directory:

- `inat_visualizer.log` (log file)
- `taxon_cache.json` (API/taxon cache)
- `descendant_taxons.txt` (optional manual descendant list)
- `tile_cache/` (map tile disk cache)
- `observations.parquet` + `taxonomy.parquet` (large; size varies by snapshot)

The application log rotates at 2 MiB and retains at most two backups, preventing
debug sessions or long-running installations from growing it without bound.

---

## License

MIT

[GitHub Repository](https://github.com/AlanRockefeller/inat.visualizer.py)

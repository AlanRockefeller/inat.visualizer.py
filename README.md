# iNaturalist Seasonal Visualizer

# By Alan Rockefeller - January 5, 2026

A desktop GUI app for exploring **seasonal patterns in iNaturalist observations** within a geographic radius. Search by organism (anything from a genus/species to higher taxa like *Agaricales*), choose a date range, and plot observation frequency by **day**, **week**, or **month** of the year.

The app supports two search modes:

- **Local Search (fast, offline-ish):** Queries a local `observations.parquet` file using DuckDB (recommended).
- **Search with API (online):** Queries the iNaturalist API via `pyinaturalist` (slower and rate-limited, but works without local data).

It also includes an **interactive map dialog** (OpenStreetMap tiles) to set coordinates and radius visually, plus export options for both graphs and data.

---

## Features

- **Interactive GUI** (PyQt6 + Matplotlib) with a sidebar of search controls and a live plot.
- **Local database mode** using DuckDB against `observations.parquet` for fast queries.
- **Taxonomy expansion** using `taxonomy.parquet` to include *all descendant taxa* of a selected organism (recursive query).
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
  - Light/dark mode toggle
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
- **macOS:** download and unzip `iNat-Seasonal-Visualizer-macOS.zip`, then open the app.
  The first time, right-click the app and choose **Open** (it is unsigned).
- **Linux:** download and extract `iNat-Seasonal-Visualizer-Linux.tar.gz`, then run the binary.

On first launch the app offers to download the required data files
(`observations.parquet` ~1 GB and `taxonomy.parquet`) into its working directory.

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

## Data files (required for Local Search and taxonomy expansion)

The application uses two Parquet files in the **current working directory**:

- `observations.parquet` (~1.02 GB)
- `taxonomy.parquet` (~8.7 MB)

If either is missing at startup, the app will prompt to download them automatically into the CWD:

- `http://images.mushroomobserver.org/observations.parquet`
- `http://images.mushroomobserver.org/taxonomy.parquet`

### Expected columns

`observations.parquet` must contain:

- `eventDate`
- `decimalLatitude`
- `decimalLongitude`
- `taxonID`

If these columns are missing, Local Search will fail with an explanatory error.

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

- `inat_visualizer.log`

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
   - **Local Search** (fast, recommended)
   - **Search with API** (online, rate-limited)

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
- On disk: `./tile_cache/` (relative to CWD) with pruning to ~`200 MB`

---

## Notes on iNaturalist API limits

Anonymous API usage can be rate-limited (HTTP 429 / 403). The app uses pagination and backoff, but large queries may still be slow.    Local searches are better - they are much faster, and don't hit the API at all.

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

When you run the program, it may create:

- `inat_visualizer.log` (log file)
- `taxon_cache.json` (API/taxon cache)
- `descendant_taxons.txt` (optional manual descendant list)
- `tile_cache/` (map tile disk cache)
- `observations.parquet` + `taxonomy.parquet` (Quite large (1 gb), if downloaded)

---

## License

MIT



[GitHub Repository](https://github.com/AlanRockefeller/inat.visualizer.py)

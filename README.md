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

## Requirements

### OS packages (Linux)
This app uses Qt via X11/XWayland and needs a couple system libs:

```bash
sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0
```

### Python environment
The script is designed for a conda environment named `inat_env` and expects Python 3.12.

The program also forces Qt to run under XWayland to avoid a Wayland protocol crash:

- `QT_QPA_PLATFORM=xcb`
- Matplotlib backend: `QtAgg`

---

## Installation

### 1) Create the conda environment

```bash
conda deactivate
conda env remove -n inat_env
conda create -n inat_env python=3.12
conda activate inat_env

conda install numpy=1.26.4 pandas=2.2.2 pyarrow=17.0.0 matplotlib=3.9.2 pyinaturalist=0.19.0
pip install PyQt6==6.8.1 duckdb
```

> The program includes an environment self-check and will warn if versions don’t match its expectations.

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

Activate the environment first:

```bash
conda activate inat_env
```

Run the script:

```bash
python inat_seasonal_visualizer.py
```

(Replace the filename with whatever you saved it as.)

### Command-line options

```bash
python inat_seasonal_visualizer.py   --lat 37.7749   --lon -122.4194   --radius 25   --scale-factor 1.5   --debug
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

### Wayland / Qt crashes
This app forces Qt onto XWayland:

```bash
export QT_QPA_PLATFORM=xcb
```

If you still have rendering issues, ensure required libs are installed:

```bash
sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0
```

### Environment check fails
The script checks:
- correct interpreter path
- matplotlib backend is `QtAgg`
- required package versions

If it prints an environment fix recipe, follow it exactly and restart.


---

## Project structure (runtime artifacts)

When you run the program, it may create:

- `inat_visualizer.log` (log file)
- `taxon_cache.json` (API/taxon cache)
- `descendant_taxons.txt` (optional manual descendant list)
- `tile_cache/` (map tile disk cache)
- `observations.parquet` + `taxonomy.parquet` (Quite large, if downloaded)

---

## License

MIT



GitHub: https://github.com/AlanRockefeller/inat.visualizer.py

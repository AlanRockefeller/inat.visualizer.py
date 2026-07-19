#!/usr/bin/env python3
# iNaturalist Seasonal Visualizer -- cross-platform (Windows, macOS, Linux).
# Install dependencies with: pip install -r requirements.txt
# On Linux, Qt also needs the system libraries: libxcb-cursor0 libxkbcommon-x11-0

import os
import sqlite3
import sys

# On Linux, route Qt through XWayland to avoid Wayland protocol crashes.
# This must be done before PyQt6 initializes. On Windows ("windows" plugin)
# and macOS ("cocoa" plugin) there is no "xcb" platform plugin, so forcing it
# would prevent the app from starting -- leave Qt's default in place there.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- stdlib ---
import argparse
import importlib
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from inat_visualizer_version import __version__
from startup_config import APP_DATA_DIRECTORY_NAME

# --- third-party ---
import duckdb
import matplotlib

matplotlib.use("QtAgg")  # Must be set before importing matplotlib.pyplot.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyinaturalist
import requests
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from PyQt6.QtCore import (
    QMutex,
    QMutexLocker,
    QSettings,
    Qt,
    QThread,
    QTimer,
    QWaitCondition,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QGuiApplication,
    QKeyEvent,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplashScreen,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from requests.exceptions import HTTPError, RequestException
from pyinaturalist.session import CACHE_FILE as PYINATURALIST_DEFAULT_CACHE_FILE


# Configure pyinaturalist with User-Agent
def build_app_user_agent() -> str:
    """Build a proper User-Agent string for application requests."""
    # Build email from parts to avoid simple scraping
    user_part = "alan" + "rockefeller"
    domain_part = "gmail.com"
    email = f"{user_part}@{domain_part}"

    app_name = "iNaturalist Seasonal Visualizer"
    repo_url = "https://github.com/AlanRockefeller/inat.visualizer.py"

    return f"{app_name} ({repo_url}; contact: {email})"


pyinaturalist.user_agent = build_app_user_agent()  # type: ignore[attr-defined]


def resource_path(filename: str) -> str:
    """Locate a bundled resource, working both from source and from a frozen build.

    PyInstaller unpacks bundled data files into a temporary directory exposed as
    ``sys._MEIPASS``. When running from source that attribute is absent, so we
    fall back to the current working directory and then the script's directory.
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, filename))
    candidates.append(os.path.join(os.getcwd(), filename))
    candidates.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


DEFAULT_THEME = "dark"
DARK_WINDOW_BACKGROUND = "#2e2e2e"
DARK_GRAPH_BACKGROUND = "#3e3e3e"
LIGHT_BACKGROUND = "#ffffff"
PLACEHOLDER_DETAILS_FONT_SCALE = 0.5
PLACEHOLDER_DETAILS_MIN_FONT_SIZE = 6.0
LOCAL_GRAPH_ACTION_TEXT = "Graph with local data"
LIVE_GRAPH_ACTION_TEXT = "Graph with live iNat data"
SPLASH_WINDOW_FLAGS = Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
DATABASE_FILE_INFO: dict[str, dict[str, Any]] = {
    "observations.parquet": {
        "url": "https://images.mushroomobserver.org/observations.parquet",
        "size": 1025327222,
        "description": (
            "iNaturalist observation dates, locations, and taxon IDs for fast "
            "local searches"
        ),
    },
    "taxonomy.parquet": {
        "url": "https://images.mushroomobserver.org/taxonomy.parquet",
        "size": 8697166,
        "description": (
            "the iNaturalist taxonomy hierarchy used to include descendants of "
            "higher taxa"
        ),
    },
}
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 2
HTTP_CACHE_FILE_NAME = "inat_api_cache.db"
DATABASE_STATS_CACHE_FILE_NAME = "database_stats.json"
DATABASE_STATS_CACHE_VERSION = 1
DEFAULT_HTTP_CACHE_MAX_MB = 128
MIN_HTTP_CACHE_MAX_MB = 16
HTTP_CACHE_MAX_MB_ENV = "INAT_VISUALIZER_HTTP_CACHE_MAX_MB"


def configured_http_cache_max_bytes(
    max_size_mb: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve the HTTP cache budget from an argument or environment variable."""
    environment = os.environ if environ is None else environ
    raw_value: Any = (
        max_size_mb
        if max_size_mb is not None
        else environment.get(HTTP_CACHE_MAX_MB_ENV, DEFAULT_HTTP_CACHE_MAX_MB)
    )
    try:
        resolved_mb = int(raw_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid %s=%r; using the %d MB default.",
            HTTP_CACHE_MAX_MB_ENV,
            raw_value,
            DEFAULT_HTTP_CACHE_MAX_MB,
        )
        resolved_mb = DEFAULT_HTTP_CACHE_MAX_MB
    if resolved_mb < MIN_HTTP_CACHE_MAX_MB:
        logging.warning(
            "HTTP cache limit %d MB is too small; using the %d MB minimum.",
            resolved_mb,
            MIN_HTTP_CACHE_MAX_MB,
        )
        resolved_mb = MIN_HTTP_CACHE_MAX_MB
    return resolved_mb * 1024 * 1024


def sqlite_cache_disk_usage(cache_path: str | Path) -> int:
    """Return SQLite database usage including WAL and shared-memory sidecars."""
    path = Path(cache_path)
    total = 0
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            pass
    return total


def vacuum_http_cache(cache, cache_path: str | Path) -> None:
    """Close pooled cache connections, checkpoint WAL, and reclaim free pages."""
    path = Path(cache_path)
    cache.responses.close()
    cache.redirects.close()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")


def maintain_http_cache(
    request_session,
    cache_path: str | Path,
    max_size_bytes: int,
    *,
    clear_valid_if_oversize: bool = True,
) -> None:
    """Remove expired responses and keep a requests-cache SQLite file bounded.

    Expired rows are deleted on every maintenance pass. SQLite is vacuumed only
    when the physical file exceeds the budget. If valid responses alone still
    exceed the budget, an app-owned cache is cleared; legacy shared caches keep
    their valid entries.
    """
    path = Path(cache_path)
    if str(cache_path) == ":memory:" or not path.exists():
        return

    try:
        cache = request_session.cache
        entries_before = len(cache.responses)
        size_before = sqlite_cache_disk_usage(path)
        cache.delete(expired=True, vacuum=False)
        entries_after = len(cache.responses)
        expired_removed = max(0, entries_before - entries_after)
        vacuumed = False
        cleared = False

        if sqlite_cache_disk_usage(path) > max_size_bytes:
            # requests-cache keeps separate pooled connections for its tables.
            # Close both before VACUUM so SQLite can actually truncate the file.
            vacuum_http_cache(cache, path)
            vacuumed = True

        if sqlite_cache_disk_usage(path) > max_size_bytes and clear_valid_if_oversize:
            cache.clear()
            vacuum_http_cache(cache, path)
            cleared = True

        size_after = sqlite_cache_disk_usage(path)
        if expired_removed or vacuumed or cleared:
            logging.info(
                "HTTP cache maintenance: expired=%d, cleared=%s, size=%.1f->%.1f MB, path=%s",
                expired_removed,
                cleared,
                size_before / (1024 * 1024),
                size_after / (1024 * 1024),
                path,
            )
        if size_after > max_size_bytes:
            logging.warning(
                "HTTP cache remains above its %.1f MB budget (size=%.1f MB, path=%s).",
                max_size_bytes / (1024 * 1024),
                size_after / (1024 * 1024),
                path,
            )
    except Exception as error:
        logging.warning("Could not maintain HTTP cache %s: %s", path, error)


def maintain_legacy_pyinaturalist_cache(
    app_cache_path: str | Path, max_size_bytes: int
) -> None:
    """Reclaim expired data from pyinaturalist's former shared default cache."""
    legacy_path = Path(PYINATURALIST_DEFAULT_CACHE_FILE)
    app_path = Path(app_cache_path)
    try:
        if (
            legacy_path.resolve() == app_path.resolve()
            or not legacy_path.exists()
            or sqlite_cache_disk_usage(legacy_path) <= max_size_bytes
        ):
            return
    except OSError as error:
        logging.warning(
            "Could not inspect legacy HTTP cache %s: %s", legacy_path, error
        )
        return

    session = None
    try:
        session = pyinaturalist.ClientSession(
            cache_file=str(legacy_path), max_retries=0, timeout=10
        )
        maintain_http_cache(
            session,
            legacy_path,
            max_size_bytes,
            clear_valid_if_oversize=False,
        )
    except Exception as error:
        logging.warning("Could not open legacy HTTP cache %s: %s", legacy_path, error)
    finally:
        if session is not None:
            session.close()


def normalize_theme(value: Any) -> str:
    """Return a supported theme, defaulting clean installations to dark mode."""
    return value if value in {"dark", "light"} else DEFAULT_THEME


def application_data_dir(
    *,
    frozen: bool | None = None,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home_dir: str | Path | None = None,
    current_dir: str | Path | None = None,
) -> Path:
    """Return the directory for mutable application data.

    A Finder-launched macOS app does not have a useful writable working
    directory, so frozen builds use each platform's conventional per-user data
    location. Source runs retain the historical current-directory behavior.
    Optional arguments make the platform selection deterministic in tests.
    """
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        return Path.cwd() if current_dir is None else Path(current_dir)

    target_platform = sys.platform if platform_name is None else platform_name
    environment = os.environ if environ is None else environ
    home = Path.home() if home_dir is None else Path(home_dir)

    if target_platform == "darwin":
        base_dir = home / "Library" / "Application Support"
    elif target_platform.startswith("win"):
        configured_dir = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        base_dir = (
            Path(configured_dir) if configured_dir else home / "AppData" / "Local"
        )
    else:
        configured_dir = environment.get("XDG_DATA_HOME")
        base_dir = Path(configured_dir) if configured_dir else home / ".local" / "share"

    return base_dir / APP_DATA_DIRECTORY_NAME


def ensure_application_data_dir(**kwargs: Any) -> Path:
    """Resolve and create the writable application data directory."""
    data_dir = application_data_dir(**kwargs)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def database_file_signature(database_path: str | Path) -> dict[str, int]:
    """Return inexpensive fields that change whenever the database changes."""
    stat = Path(database_path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_database_stats_cache(
    cache_path: str | Path,
    database_path: str | Path,
) -> tuple[int, int | str] | None:
    """Return cached database counts when they match the installed snapshot."""
    try:
        with open(cache_path, encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        if not isinstance(cached, dict):
            return None
        if cached.get("version") != DATABASE_STATS_CACHE_VERSION:
            return None
        if cached.get("database") != database_file_signature(database_path):
            return None
        total_observations = cached.get("total_observations")
        unique_taxa = cached.get("unique_taxa")
        if not isinstance(total_observations, int) or isinstance(
            total_observations, bool
        ):
            return None
        if not (
            isinstance(unique_taxa, int)
            and not isinstance(unique_taxa, bool)
            or unique_taxa == "N/A"
        ):
            return None
        return total_observations, unique_taxa
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_database_stats_cache(
    cache_path: str | Path,
    database_path: str | Path,
    total_observations: int,
    unique_taxa: int | str,
) -> None:
    """Atomically persist database counts for reuse on subsequent launches."""
    path = Path(cache_path)
    temp_path = path.with_name(f"{path.name}.part")
    try:
        payload = {
            "version": DATABASE_STATS_CACHE_VERSION,
            "database": database_file_signature(database_path),
            "total_observations": total_observations,
            "unique_taxa": unique_taxa,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, indent=2)
        os.replace(temp_path, path)
    except OSError as error:
        logging.warning("Could not save database statistics cache %s: %s", path, error)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def missing_database_files(
    working_dir: str | Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Return database files that are not present in the runtime data directory."""
    data_dir = Path(working_dir)
    return [
        (filename, info)
        for filename, info in DATABASE_FILE_INFO.items()
        if not (data_dir / filename).exists()
    ]


def available_database_updates(
    working_dir: str | Path,
    cancel_callback=None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return changed files from a differently sized hosted database release.

    The observation database is the opt-in marker for local mode. If it is not
    installed, avoid all network requests so API-only users are not repeatedly
    asked about the optional database.
    """
    data_dir = Path(working_dir)
    if not (data_dir / "observations.parquet").exists() or (
        cancel_callback is not None and cancel_callback()
    ):
        return []

    def differently_sized_remote_file(
        filename: str, configured_info: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        local_path = data_dir / filename
        if not local_path.exists() or (
            cancel_callback is not None and cancel_callback()
        ):
            return None

        local_stat = local_path.stat()
        try:
            response = requests.head(
                configured_info["url"],
                headers={"User-Agent": build_app_user_agent()},
                allow_redirects=True,
                timeout=5,
            )
            if cancel_callback is not None and cancel_callback():
                return None
            if response.status_code == 304:
                return None
            response.raise_for_status()
        except RequestException as error:
            # An update check must never prevent an offline startup.
            logging.info("Could not check %s for updates: %s", filename, error)
            return None

        try:
            remote_size = int(response.headers.get("content-length", ""))
        except (TypeError, ValueError):
            logging.info("Update check for %s returned no file size.", filename)
            return None

        # Uploading a locally generated file gives the hosted copy a newer
        # Last-Modified timestamp even though its contents are identical. The
        # database build guarantees that real updates have a different size,
        # so Content-Length is the authoritative version signal.
        if remote_size == local_stat.st_size:
            return None

        remote_info = dict(configured_info)
        remote_info["size"] = remote_size
        return filename, remote_info

    # observations.parquet is the release marker for the coordinated database
    # files. In particular, do not offer an old hosted taxonomy snapshot when a
    # freshly built observation database has already been uploaded and matches.
    observations_update = differently_sized_remote_file(
        "observations.parquet", DATABASE_FILE_INFO["observations.parquet"]
    )
    if observations_update is None:
        return []

    updates = [observations_update]
    for filename, configured_info in DATABASE_FILE_INFO.items():
        if filename == "observations.parquet":
            continue
        companion_update = differently_sized_remote_file(filename, configured_info)
        if companion_update is not None:
            updates.append(companion_update)

    return updates


def database_download_choice_message(
    files_to_download: list[tuple[str, dict[str, Any]]],
) -> str:
    """Explain the storage and search tradeoffs for missing database files."""
    missing_names = {filename for filename, _info in files_to_download}
    observations_missing = "observations.parquet" in missing_names
    taxonomy_missing = "taxonomy.parquet" in missing_names

    if observations_missing:
        download_size = "approximately 1 GB"
    else:
        total_bytes = sum(int(info["size"]) for _filename, info in files_to_download)
        download_size = f"approximately {total_bytes / (1024 * 1024):.1f} MB"

    download_benefits = []
    if observations_missing:
        download_benefits.append(
            f"Enables fast '{LOCAL_GRAPH_ACTION_TEXT}' searches without requesting "
            "every search from the iNaturalist API"
        )
    if taxonomy_missing:
        download_benefits.append(
            "Includes descendant species when searching higher taxa such as Agaricales"
        )

    if observations_missing:
        alternative_heading = "Use iNaturalist API Only"
        without_download = (
            f"The app will start immediately in API-only mode. "
            f"'{LIVE_GRAPH_ACTION_TEXT}' requires an internet connection and may "
            f"be slower or rate-limited. '{LOCAL_GRAPH_ACTION_TEXT}' will be "
            "unavailable."
        )
    else:
        alternative_heading = "Continue Without Taxonomy Download"
        without_download = (
            f"The existing observation database will still support "
            f"'{LOCAL_GRAPH_ACTION_TEXT}', but searches for higher taxa may include "
            "only the selected parent taxon unless descendants are already cached."
        )

    benefits_text = "\n".join(f"• {benefit}." for benefit in download_benefits)
    return (
        f"Download Local Database ({download_size}):\n"
        f"{benefits_text}\n"
        "• Uses disk space and may take several minutes to download.\n\n"
        f"{alternative_heading}:\n"
        f"• {without_download}"
    )


# --- Constants for MapDialog ---
INATURALIST_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
INATURALIST_GET_QUERY_LIMIT = 1800
OBSERVATION_TAXON_FILTER_KEYS = ("taxon_id", "not_in_taxon_id")
HTTP_ERROR_BODY_LOG_LIMIT = 500
MAX_TILES_PER_REQUEST = 100
MAX_CACHE_SIZE = 500  # Max number of tiles to keep in memory (RAM)
TILE_CACHE_DIR_NAME = "tile_cache"
MAX_DISK_CACHE_SIZE_MB = 200
PLACE_SEARCH_RESULT_LIMIT = 8
PLACE_SEARCH_LOG_RESPONSE_LIMIT = 20_000
WEB_MERCATOR_MAX_LATITUDE = 85.05112878


def _coordinate_pairs(value: Any):
    """Yield longitude/latitude pairs from nested GeoJSON coordinates."""
    if not isinstance(value, (list, tuple)):
        return
    if (
        len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    for child in value:
        yield from _coordinate_pairs(child)


def normalize_place_search_results(response: Any) -> list[dict[str, Any]]:
    """Return validated, display-ready place records from an iNaturalist search."""
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        return []

    places = []
    for result in response["results"]:
        if (
            not isinstance(result, dict)
            or str(result.get("type", "")).lower() != "place"
        ):
            continue
        record = result.get("record")
        if not isinstance(record, dict):
            continue

        raw_location = record.get("location")
        if isinstance(raw_location, str):
            raw_location = raw_location.split(",")
        if not isinstance(raw_location, (list, tuple)) or len(raw_location) < 2:
            continue
        try:
            lat, lon = float(raw_location[0]), float(raw_location[1])
        except (TypeError, ValueError):
            continue
        if not (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90.0 <= lat <= 90.0
            and -180.0 <= lon <= 180.0
        ):
            continue

        display_name = str(
            record.get("display_name") or record.get("name") or ""
        ).strip()
        if not display_name:
            continue

        bounds = None
        bounding_geojson = record.get("bounding_box_geojson")
        if isinstance(bounding_geojson, dict):
            pairs = list(_coordinate_pairs(bounding_geojson.get("coordinates")))
            valid_pairs = [
                (
                    lon + ((pair_lon - lon + 180.0) % 360.0) - 180.0,
                    pair_lat,
                )
                for pair_lon, pair_lat in pairs
                if math.isfinite(pair_lon)
                and math.isfinite(pair_lat)
                and -180.0 <= pair_lon <= 180.0
                and -90.0 <= pair_lat <= 90.0
            ]
            if valid_pairs:
                lons, lats = zip(*valid_pairs, strict=True)
                bounds = (min(lons), max(lons), min(lats), max(lats))

        places.append(
            {
                "id": record.get("id"),
                "name": str(record.get("name") or "").strip(),
                "display_name": display_name,
                "admin_level": record.get("admin_level"),
                "ancestor_place_ids": record.get("ancestor_place_ids") or [],
                "lat": lat,
                "lon": lon,
                "bounds": bounds,
            }
        )
    return places


def place_search_response_for_log(response: Any) -> str:
    """Serialize a bounded place-search response for DEBUG diagnostics."""
    try:
        serialized = json.dumps(
            response,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        serialized = repr(response)
    if len(serialized) > PLACE_SEARCH_LOG_RESPONSE_LIMIT:
        omitted = len(serialized) - PLACE_SEARCH_LOG_RESPONSE_LIMIT
        serialized = (
            serialized[:PLACE_SEARCH_LOG_RESPONSE_LIMIT]
            + f"... <{omitted} characters omitted>"
        )
    return serialized


def place_result_view_limits(
    place: Mapping[str, Any], padding: float = 0.1
) -> tuple[float, float, float, float]:
    """Return padded map limits for a normalized place search result."""
    lat = float(place["lat"])
    lon = float(place["lon"])
    bounds = place.get("bounds")
    if bounds is None:
        west, east, south, north = lon - 0.25, lon + 0.25, lat - 0.25, lat + 0.25
    else:
        west, east, south, north = (float(value) for value in bounds)

    # Give point-sized and very small places enough context to be recognizable.
    lon_span = max(east - west, 0.02)
    lat_span = max(north - south, 0.02)
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    half_lon = lon_span * (0.5 + padding)
    half_lat = lat_span * (0.5 + padding)
    south = max(-WEB_MERCATOR_MAX_LATITUDE, center_lat - half_lat)
    north = min(WEB_MERCATOR_MAX_LATITUDE, center_lat + half_lat)
    return center_lon - half_lon, center_lon + half_lon, south, north


def calculate_local_search_bounds(
    lat: float, lon: float, radius: float
) -> tuple[float, float, float, float]:
    """Calculate a latitude/longitude bounding box around a radius in km."""
    lat_rad = math.radians(lat)
    earth_radius_km = 6371
    lat_diff = radius / earth_radius_km

    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        lon_diff = math.pi
    else:
        asin_arg = math.sin(lat_diff) / cos_lat
        asin_arg = max(-1.0, min(1.0, asin_arg))
        lon_diff = math.asin(asin_arg)

    lat_min = max(-90.0, lat - math.degrees(lat_diff))
    lat_max = min(90.0, lat + math.degrees(lat_diff))
    lon_min = lon - math.degrees(lon_diff)
    lon_max = lon + math.degrees(lon_diff)
    return lat_min, lat_max, lon_min, lon_max


def calendar_aligned_week_numbers(dates: pd.Series) -> pd.Series:
    """Return 1-52 week bins with ISO year-boundary dates kept by month.

    ISO week 53 straddles December and January, while some late-December dates
    belong to ISO week 1 and some early-January dates belong to ISO week 52.
    Seasonal plots use a fixed 52-bin calendar, so put January boundary dates
    in week 1 and December boundary dates in week 52 instead of dropping them
    or displaying them at the opposite end of the year.
    """
    weeks = dates.dt.isocalendar().week.astype("int64")
    months = dates.dt.month
    january_boundary = (months == 1) & weeks.isin((52, 53))
    december_boundary = (months == 12) & weeks.isin((1, 53))
    return weeks.mask(january_boundary, 1).mask(december_boundary, 52)


def http_error_details(error: HTTPError) -> tuple[int | None, str | None]:
    """Return an HTTP status and bounded single-line response-body excerpt."""
    response = error.response
    if response is None:
        return None, None
    body = ""
    try:
        body = " ".join(response.text.split())
    except (AttributeError, UnicodeError):
        pass
    if len(body) > HTTP_ERROR_BODY_LOG_LIMIT:
        body = body[: HTTP_ERROR_BODY_LOG_LIMIT - 3] + "..."
    return response.status_code, body or None


def create_log_file_handler(log_path: str | Path) -> RotatingFileHandler:
    """Create the application's bounded, UTF-8 rotating file handler."""
    return RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )


def should_post_observation_search(params: dict[str, Any]) -> bool:
    """Use POST when taxon filters would likely overflow the GET query string."""
    if not any(
        isinstance(params.get(key), list) and params[key]
        for key in OBSERVATION_TAXON_FILTER_KEYS
    ):
        return False

    encoded_params = {
        key: value for key, value in params.items() if value is not None and value != ""
    }
    query_string = urlencode(encoded_params, doseq=True)
    return len(query_string) > INATURALIST_GET_QUERY_LIMIT


def fetch_observations_page(
    params: dict[str, Any], force_post: bool = False, request_session=None
) -> dict[str, Any]:
    """Fetch one page of observations, switching to POST for oversized taxon filters."""
    if force_post or should_post_observation_search(params):
        query_string = urlencode(
            {
                key: value
                for key, value in params.items()
                if value is not None and value != ""
            },
            doseq=True,
        )
        taxon_filter_sizes = {
            key: len(params[key])
            for key in OBSERVATION_TAXON_FILTER_KEYS
            if isinstance(params.get(key), list)
        }
        logging.info(
            "Using POST for observation search (query_length=%s, taxon_filters=%s)",
            len(query_string),
            taxon_filter_sizes,
        )
        response = requests.post(
            INATURALIST_OBSERVATIONS_URL,
            data=params,
            headers={
                "Accept": "application/json",
                "User-Agent": build_app_user_agent(),
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    if request_session is None:
        return pyinaturalist.get_observations(**params)
    return pyinaturalist.get_observations(session=request_session, **params)


class SearchCancelled(Exception):
    """Raised inside a worker when the user cancels a search."""


def raise_if_search_cancelled(cancel_callback=None) -> None:
    if cancel_callback and cancel_callback():
        raise SearchCancelled


def wait_for_search_delay(seconds: float, cancel_callback=None) -> None:
    """Wait for a retry or rate-limit delay while checking for cancellation."""
    deadline = time.monotonic() + seconds
    while True:
        raise_if_search_cancelled(cancel_callback)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def fetch_all_observation_pages(
    params: dict[str, Any],
    progress_callback=None,
    api_call_callback=None,
    cancel_callback=None,
    request_session=None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch every observation page without interacting with Qt widgets.

    Keeping the network and retry loop UI-independent allows live searches to
    run in a worker thread while the main Qt event loop continues repainting
    the window and responding to Windows messages.
    """
    all_results = []
    page = 1
    base_params = params.copy()
    per_page = 500
    max_retries = 3
    post_rate_limit_delay = 1.0
    last_http_status: int | None = None
    force_post = False

    while True:
        raise_if_search_cancelled(cancel_callback)
        page_params = {**base_params, "page": page, "per_page": per_page}
        retries = 0

        while retries <= max_retries:
            try:
                raise_if_search_cancelled(cancel_callback)
                if progress_callback:
                    progress_callback(f"Fetching page {page}...", None)
                response = fetch_observations_page(
                    page_params,
                    force_post=force_post,
                    request_session=request_session,
                )
                if api_call_callback:
                    api_call_callback()
                raise_if_search_cancelled(cancel_callback)

                results = response.get("results", [])
                all_results.extend(results)
                total_results = response.get("total_results", 0)
                progress = None
                if total_results > 0:
                    progress = min(100, int((len(all_results) / total_results) * 100))
                if progress_callback:
                    progress_callback(
                        f"Fetched {len(all_results)} / {total_results} observations",
                        progress,
                    )

                if len(all_results) >= total_results or not results:
                    return all_results, None

                page += 1
                # ClientSession already enforces GET rate limits. The direct
                # POST fallback bypasses that session, so retain a short,
                # cancellation-aware delay only for those oversized filters.
                if force_post or should_post_observation_search(page_params):
                    if progress_callback:
                        progress_callback(
                            "Waiting briefly to respect iNaturalist rate limits...",
                            progress,
                        )
                    wait_for_search_delay(post_rate_limit_delay, cancel_callback)
                break

            except SearchCancelled:
                raise
            except HTTPError as e:
                raise_if_search_cancelled(cancel_callback)
                status_code, response_body = http_error_details(e)
                if status_code == 414 and not force_post:
                    force_post = True
                    logging.warning(
                        "Observation search exceeded GET URL limits on page %s; retrying with POST.",
                        page,
                    )
                    continue
                if status_code in (429, 403):
                    last_http_status = status_code
                    retries += 1
                    wait_time = 2**retries
                    if progress_callback:
                        progress_callback(
                            f"Rate limited, retrying in {wait_time}s "
                            f"(attempt {retries}/{max_retries})...",
                            None,
                        )
                    wait_for_search_delay(wait_time, cancel_callback)
                else:
                    if response_body:
                        logging.error(
                            "API search HTTP %s on page %s: %s",
                            status_code,
                            page,
                            response_body,
                        )
                    else:
                        logging.error(
                            "API search HTTP %s on page %s", status_code, page
                        )
                    return all_results, str(e)
            except Exception as e:
                raise_if_search_cancelled(cancel_callback)
                logging.exception(
                    "API search failed on page %s (%s)", page, type(e).__name__
                )
                return all_results, str(e)

        if retries > max_retries:
            error_msg = (
                f"Max retries exceeded for error {last_http_status}. "
                "You may be using anonymous API access, which has stricter rate "
                "limits. To increase limits, set INATURALIST_APP_ID and "
                "INATURALIST_APP_SECRET in your environment."
            )
            return all_results, error_msg


class DiskTileCache:
    """Persistent disk cache for map tiles with size limit and LRU pruning."""

    def __init__(
        self, parent_dir: str | Path, max_size_mb: int = MAX_DISK_CACHE_SIZE_MB
    ) -> None:
        self.cache_dir = Path(parent_dir) / TILE_CACHE_DIR_NAME
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / str(z) / str(x) / f"{y}.png"

    def get_tile(self, z: int, x: int, y: int) -> "Image.Image | None":
        path = self.get_path(z, x, y)
        if path.exists():
            try:
                # Update mtime for LRU
                path.touch()
                with Image.open(path) as img:
                    return img.convert("RGB").copy()
            except Exception as e:
                logging.warning(f"Corrupt tile in disk cache {path}: {e}")
        return None

    def put_tile(self, z: int, x: int, y: int, image_bytes: bytes) -> None:
        path = self.get_path(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        temp_path = path.with_suffix(f".tmp.{os.getpid()}.{time.time()}")
        try:
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            os.replace(temp_path, path)
            self._prune_if_needed()
        except Exception as e:
            logging.exception(f"Failed to write tile to disk {path}: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _prune_if_needed(self) -> None:
        # Simple probability-based check to avoid scanning on every write
        if np.random.random() > 0.05:
            return

        total_size = 0
        files = []
        for p in self.cache_dir.rglob("*.png"):
            try:
                stat = p.stat()
                total_size += stat.st_size
                files.append((stat.st_mtime, p, stat.st_size))
            except OSError:
                pass

        if total_size > self.max_size_mb * 1024 * 1024:
            # Sort by mtime (oldest first)
            files.sort()
            deleted_size = 0
            target_reduction = total_size * 0.2  # clear 20%
            for _mtime, p, size in files:
                try:
                    p.unlink()
                    deleted_size += size
                    if deleted_size >= target_reduction:
                        break
                except OSError:
                    pass
            # Remove empty dirs; sort deepest-first so children are removed before parents.
            dirs = sorted(
                (p for p in self.cache_dir.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            )
            for p in dirs:
                try:
                    p.rmdir()  # Only removes if empty
                except OSError:
                    pass


class PlaceSearchWorker(QThread):
    """Search iNaturalist places without blocking the map dialog."""

    results_ready = pyqtSignal(object)
    search_failed = pyqtSignal(str)

    def __init__(self, query: str, http_cache_file: str | Path | None = None) -> None:
        super().__init__()
        self.query = query
        self.http_cache_file = (
            str(http_cache_file) if http_cache_file is not None else ":memory:"
        )

    def cancel(self) -> None:
        self.requestInterruption()

    @staticmethod
    def _qualified_query_parts(query: str) -> tuple[str, str] | None:
        if "," not in query:
            return None
        name, qualifier = (part.strip() for part in query.split(",", 1))
        return (name, qualifier) if name and qualifier else None

    @staticmethod
    def _exact_qualifier_id(
        qualifier: str, qualifier_places: list[dict[str, Any]]
    ) -> Any | None:
        normalized_qualifier = qualifier.casefold()
        for place in qualifier_places:
            names = {
                str(place.get("name") or "").casefold(),
                str(place.get("display_name") or "")
                .split(",", 1)[0]
                .strip()
                .casefold(),
            }
            if normalized_qualifier in names:
                return place.get("id")
        return None

    @staticmethod
    def _places_with_ancestor(
        places: list[dict[str, Any]], ancestor_id: Any
    ) -> list[dict[str, Any]]:
        return [
            place
            for place in places
            if ancestor_id == place.get("id")
            or ancestor_id in place.get("ancestor_place_ids", [])
        ]

    def _search(self, query: str, request_session) -> list[dict[str, Any]]:
        request_params = {
            "q": query,
            "sources": "places",
            "per_page": PLACE_SEARCH_RESULT_LIMIT,
        }
        debug_logging = logging.getLogger().isEnabledFor(logging.DEBUG)
        if debug_logging:
            logging.debug(
                "Place search request: GET https://api.inaturalist.org/v1/search params=%r",
                request_params,
            )
        response = pyinaturalist.search(session=request_session, **request_params)
        if debug_logging:
            logging.debug(
                "Place search response for query=%r: %s",
                query,
                place_search_response_for_log(response),
            )
        return normalize_place_search_results(response)

    def run(self) -> None:
        request_session = None
        try:
            request_session = pyinaturalist.ClientSession(
                cache_file=self.http_cache_file,
                max_retries=0,
                timeout=8,
            )
            places = self._search(self.query, request_session)
            qualified_parts = self._qualified_query_parts(self.query)
            if not places and qualified_parts and not self.isInterruptionRequested():
                name, qualifier = qualified_parts
                logging.debug(
                    "Place search query=%r returned no usable results; resolving "
                    "qualifier=%r and retrying name=%r",
                    self.query,
                    qualifier,
                    name,
                )
                qualifier_places = self._search(qualifier, request_session)
                qualifier_id = self._exact_qualifier_id(qualifier, qualifier_places)
                fallback_places = self._search(name, request_session)
                scoped_places = (
                    self._places_with_ancestor(fallback_places, qualifier_id)
                    if qualifier_id is not None
                    else []
                )
                if scoped_places:
                    logging.debug(
                        "Place search qualifier=%r resolved to place_id=%r; "
                        "kept %d of %d name matches",
                        qualifier,
                        qualifier_id,
                        len(scoped_places),
                        len(fallback_places),
                    )
                    places = scoped_places
                else:
                    logging.debug(
                        "Place search qualifier=%r could not narrow %d name matches; "
                        "returning the unqualified matches",
                        qualifier,
                        len(fallback_places),
                    )
                    places = fallback_places
            if not self.isInterruptionRequested():
                self.results_ready.emit(places)
        except Exception as error:
            if not self.isInterruptionRequested():
                logging.exception("Place search failed for %r", self.query)
                self.search_failed.emit(str(error))
        finally:
            if request_session is not None:
                request_session.close()


class TileLoaderWorker(QThread):
    """Background worker to fetch map tiles."""

    view_ready = pyqtSignal(int, object, tuple)  # job_id, composite_image, extent
    network_error = pyqtSignal(str)
    network_recovered = pyqtSignal()
    retry_complete = pyqtSignal()
    tiles_skipped = (
        pyqtSignal()
    )  # Signal when some tiles were skipped due to network suspension

    def __init__(self, working_dir: str | Path) -> None:
        super().__init__()
        self.disk_cache = DiskTileCache(working_dir)
        self.ram_cache = OrderedDict()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._pending_job = None  # (job_id, zoom, x0, x1, y0, y1)
        self._running = True
        self.last_network_request_time = 0.0
        self._network_suspended_until = 0.0
        self._notified_error = False
        self._skipped_tiles = set()

    def request_view(
        self, job_id: int, zoom: int, x0: float, x1: float, y0: float, y1: float
    ) -> None:
        with QMutexLocker(self.mutex):
            self._pending_job = (job_id, zoom, x0, x1, y0, y1)
            self._skipped_tiles.clear()
            self.wait_condition.wakeOne()

    def stop(self) -> None:
        with QMutexLocker(self.mutex):
            self._running = False
            self.wait_condition.wakeOne()
        self.wait()

    def run(self) -> None:
        session = requests.Session()
        session.headers.update({"User-Agent": build_app_user_agent()})

        while True:
            job = None
            with QMutexLocker(self.mutex):
                while self._pending_job is None and self._running:
                    self.wait_condition.wait(self.mutex)
                if not self._running:
                    return
                job = self._pending_job
                self._pending_job = (
                    None  # Clear it so we don't repeat unless new one comes
                )

            if not job:
                continue

            job_id, base_zoom, x0, x1, y0, y1 = job
            self._process_job(session, job_id, base_zoom, x0, x1, y0, y1)

    def _get_tile_range(
        self, z: int, x0: float, x1: float, y0: float, y1: float
    ) -> tuple[int, int, int, int]:
        n = 2.0**z
        xtile_min = math.floor((x0 + 180.0) / 360.0 * n)
        xtile_max = math.floor((x1 + 180.0) / 360.0 * n)
        if xtile_max < xtile_min:
            xtile_max += int(n)

        def lat2y(lat):
            return (
                (
                    1.0
                    - math.asinh(math.tan(math.radians(max(-85.0, min(85.0, lat)))))
                    / math.pi
                )
                / 2.0
                * n
            )

        y_a = int(lat2y(y1))
        y_b = int(lat2y(y0))
        return (
            xtile_min,
            xtile_max,
            min(y_a, y_b),
            max(y_a, y_b),
        )

    def _process_job(
        self,
        session: requests.Session,
        job_id: int,
        base_zoom: int,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> None:
        MAX_UNCACHED_TILES = 16
        MAX_TOTAL_TILES = 100

        zoom = base_zoom
        while zoom >= 0:
            x_min, x_max, y_min, y_max = self._get_tile_range(zoom, x0, x1, y0, y1)
            total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)

            if total_tiles > MAX_TOTAL_TILES:
                zoom -= 1
                continue

            uncached = 0
            for tx in range(x_min, x_max + 1):
                for ty in range(y_min, y_max + 1):
                    with QMutexLocker(self.mutex):
                        if (
                            self._pending_job is not None
                            and self._pending_job[0] > job_id
                        ):
                            return

                    wrapped_x = tx % int(2.0**zoom)
                    key = (zoom, wrapped_x, ty)
                    if key not in self.ram_cache:
                        path_obj = self.disk_cache.get_path(zoom, wrapped_x, ty)
                        if not path_obj.exists():
                            uncached += 1

            if uncached <= MAX_UNCACHED_TILES:
                break
            zoom -= 1

        if zoom < 0:
            zoom = 0

        x_min, x_max, y_min, y_max = self._get_tile_range(zoom, x0, x1, y0, y1)
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)

        tiles = []
        tile_positions = []
        trigger_retry = False
        skipped_any = False

        for x in x_range:
            for y in y_range:
                with QMutexLocker(self.mutex):
                    if self._pending_job is not None and self._pending_job[0] > job_id:
                        return

                wrapped_x = x % int(2.0**zoom)
                tile_key = (zoom, wrapped_x, y)
                img = None

                if tile_key in self.ram_cache:
                    img = self.ram_cache[tile_key]
                    self.ram_cache.move_to_end(tile_key)
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"RAM cache hit: {zoom}/{wrapped_x}/{y}")

                if img is None:
                    img = self.disk_cache.get_tile(zoom, wrapped_x, y)
                    if img is not None:
                        self.ram_cache[tile_key] = img
                        if len(self.ram_cache) > MAX_CACHE_SIZE:
                            self.ram_cache.popitem(last=False)
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(f"Disk cache hit: {zoom}/{wrapped_x}/{y}")

                if img is None:
                    if time.time() < self._network_suspended_until:
                        with QMutexLocker(self.mutex):
                            self._skipped_tiles.add(tile_key)
                        skipped_any = True
                        continue

                    now = time.time()
                    elapsed = now - self.last_network_request_time
                    throttle_delay = 0.15
                    if elapsed < throttle_delay:
                        time.sleep(throttle_delay - elapsed)
                        with QMutexLocker(self.mutex):
                            if (
                                self._pending_job is not None
                                and self._pending_job[0] > job_id
                            ):
                                return

                    self.last_network_request_time = time.time()
                    url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_x}/{y}.png"

                    try:
                        resp = session.get(url, timeout=3.0)
                        if resp.status_code == 200:
                            if self._notified_error:
                                self._notified_error = False
                                self.network_recovered.emit()
                                trigger_retry = True

                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug(f"Network fetch: {zoom}/{wrapped_x}/{y}")
                            img_bytes = resp.content
                            img = Image.open(BytesIO(img_bytes)).convert("RGB")

                            self.ram_cache[tile_key] = img
                            if len(self.ram_cache) > MAX_CACHE_SIZE:
                                self.ram_cache.popitem(last=False)
                            self.disk_cache.put_tile(zoom, wrapped_x, y, img_bytes)

                            with QMutexLocker(self.mutex):
                                self._skipped_tiles.discard(tile_key)
                        else:
                            logging.debug(
                                f"Tile {zoom}/{wrapped_x}/{y} HTTP {resp.status_code}"
                            )
                            if resp.status_code in (
                                403,
                                429,
                                418,
                                408,
                                500,
                                502,
                                503,
                                504,
                            ):
                                if not self._notified_error:
                                    self._notified_error = True
                                    if resp.status_code in (403, 429, 418):
                                        msg = f"Map service throttled requests (HTTP {resp.status_code})."
                                    elif resp.status_code == 408:
                                        msg = "Map service request timed out."
                                    else:
                                        msg = f"Map service error (HTTP {resp.status_code})."
                                    self.network_error.emit(f"{msg} Backing off.")
                                self._network_suspended_until = time.time() + 15.0
                                with QMutexLocker(self.mutex):
                                    self._skipped_tiles.add(tile_key)
                                skipped_any = True
                                continue
                    except RequestException as e:
                        logging.debug(
                            f"Failed to load tile {zoom}/{wrapped_x}/{y}: {e}"
                        )
                        if not self._notified_error:
                            self._notified_error = True
                            self.network_error.emit(
                                f"Network error: {e!s}. Backing off."
                            )
                        self._network_suspended_until = time.time() + 15.0
                        with QMutexLocker(self.mutex):
                            self._skipped_tiles.add(tile_key)
                        skipped_any = True
                        continue

                if img:
                    tiles.append(img)
                    tile_positions.append((x, y))

        w = (x_max - x_min + 1) * 256
        h = (y_max - y_min + 1) * 256
        composite = Image.new("RGB", (w, h), (230, 230, 230))

        for img, (xpos, ypos) in zip(tiles, tile_positions, strict=True):
            composite.paste(img, ((xpos - x_min) * 256, (ypos - y_min) * 256))

        def num2deg(xtile, ytile, zoom):
            n = 2.0**zoom
            lon_deg = xtile / n * 360.0 - 180.0
            lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
            lat_deg = np.degrees(lat_rad)
            return lat_deg, lon_deg

        north, west = num2deg(x_min, y_min, zoom)
        south, east = num2deg(x_max + 1, y_max + 1, zoom)

        if tiles:
            self.view_ready.emit(job_id, composite, (west, east, south, north))
        if skipped_any:
            self.tiles_skipped.emit()
        if trigger_retry:
            if self._retry_skipped_tiles(session):
                self.retry_complete.emit()

    def _retry_skipped_tiles(self, session: requests.Session) -> bool:
        with QMutexLocker(self.mutex):
            skipped = sorted(self._skipped_tiles)
        recovered_any = False
        for tile_key in skipped:
            with QMutexLocker(self.mutex):
                if self._pending_job is not None:
                    return recovered_any

            zoom, wrapped_x, y = tile_key
            if tile_key in self.ram_cache:
                with QMutexLocker(self.mutex):
                    self._skipped_tiles.discard(tile_key)
                continue

            if time.time() < self._network_suspended_until:
                break

            now = time.time()
            elapsed = now - self.last_network_request_time
            if elapsed < 0.15:
                time.sleep(0.15 - elapsed)
            self.last_network_request_time = time.time()

            url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_x}/{y}.png"
            try:
                resp = session.get(url, timeout=3.0)
                if resp.status_code == 200:
                    img_bytes = resp.content
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    self.ram_cache[tile_key] = img
                    if len(self.ram_cache) > MAX_CACHE_SIZE:
                        self.ram_cache.popitem(last=False)
                    self.disk_cache.put_tile(zoom, wrapped_x, y, img_bytes)
                    with QMutexLocker(self.mutex):
                        self._skipped_tiles.discard(tile_key)
                    recovered_any = True
                else:
                    logging.debug(
                        f"Tile {zoom}/{wrapped_x}/{y} HTTP {resp.status_code}"
                    )
                    if resp.status_code in (403, 429, 418, 408, 500, 502, 503, 504):
                        if not self._notified_error:
                            self._notified_error = True
                            if resp.status_code in (403, 429, 418):
                                msg = f"Map service throttled requests (HTTP {resp.status_code})."
                            elif resp.status_code == 408:
                                msg = "Map service request timed out."
                            else:
                                msg = f"Map service error (HTTP {resp.status_code})."
                            self.network_error.emit(f"{msg} Backing off.")
                        self._network_suspended_until = time.time() + 15.0
                    break
            except RequestException as e:
                logging.debug(f"Failed to load tile {zoom}/{wrapped_x}/{y}: {e!s}")
                if not self._notified_error:
                    self._notified_error = True
                    self.network_error.emit(f"Network error: {e!s}. Backing off.")
                self._network_suspended_until = time.time() + 15.0
                break

        with QMutexLocker(self.mutex):
            still_skipped = len(self._skipped_tiles) > 0
        if still_skipped:
            self.tiles_skipped.emit()

        return recovered_any


class DatabaseUpdateCheckWorker(QThread):
    """Check for database updates without delaying or blocking the main window."""

    updates_ready = pyqtSignal(object)

    def __init__(self, working_dir: str | Path) -> None:
        super().__init__()
        self.working_dir = str(working_dir)

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            updates = available_database_updates(
                self.working_dir,
                cancel_callback=self.isInterruptionRequested,
            )
        except Exception:
            logging.exception("Database update check failed")
            updates = []
        if not self.isInterruptionRequested():
            self.updates_ready.emit(updates)


class ApiSearchWorker(QThread):
    """Background worker for taxon lookup and live iNaturalist searches."""

    progress = pyqtSignal(str, object)
    api_call_completed = pyqtSignal()
    taxon_resolved = pyqtSignal(str, object)
    search_finished = pyqtSignal(object, object)
    search_failed = pyqtSignal(str)
    search_cancelled = pyqtSignal()

    def __init__(
        self,
        params: dict[str, Any],
        organism: str,
        exclude: str,
        taxon_cache: Mapping[str, Any],
        http_cache_file: str | Path | None = None,
        http_cache_max_bytes: int = DEFAULT_HTTP_CACHE_MAX_MB * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.params = params.copy()
        self.organism = organism
        self.exclude = exclude
        self.taxon_cache = dict(taxon_cache)
        self.http_cache_file = (
            str(http_cache_file) if http_cache_file is not None else ":memory:"
        )
        self.http_cache_max_bytes = http_cache_max_bytes
        self._cancel_requested = False
        self.request_session = None
        self._started_at = 0.0
        self._api_call_count = 0
        self._page_count = 0
        self._cache_hit_count = 0
        self._observation_count = 0

    def cancel(self) -> None:
        """Ask the worker to stop at the next safe cancellation point."""
        self._cancel_requested = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancel_requested or self.isInterruptionRequested()

    def _record_http_response(self, response, *_args, **_kwargs):
        if getattr(response, "from_cache", False):
            self._cache_hit_count += 1
        return response

    def _record_api_call(self) -> None:
        self._api_call_count += 1
        self.api_call_completed.emit()

    def _record_observation_page(self) -> None:
        self._page_count += 1
        self._record_api_call()

    def _log_search_summary(self, status: str) -> None:
        logging.info(
            "Live search %s: observations=%d, pages=%d, api_calls=%d, cache_hits=%d, duration=%.1fs",
            status,
            self._observation_count,
            self._page_count,
            self._api_call_count,
            self._cache_hit_count,
            max(0.0, time.monotonic() - self._started_at),
        )

    def _get_taxon_id(self, query: str) -> int | None:
        raise_if_search_cancelled(self.is_cancelled)
        cached_taxon_id = self.taxon_cache.get(query)
        if isinstance(cached_taxon_id, int):
            logging.info(
                "Retrieved taxon ID for %s from cache: %s",
                query,
                cached_taxon_id,
            )
            return cached_taxon_id

        self.progress.emit(f"Looking up taxon ID for {query}...", None)
        try:
            taxa = pyinaturalist.get_taxa(
                q=query, limit=1, session=self.request_session
            )
            self._record_api_call()
        except Exception as e:
            raise_if_search_cancelled(self.is_cancelled)
            logging.error("Failed to fetch taxon ID for %s: %s", query, e)
            return None
        raise_if_search_cancelled(self.is_cancelled)

        results = taxa.get("results", [])
        if not results:
            logging.warning("No taxon ID found for %s", query)
            return None

        taxon_id = results[0]["id"]
        self.taxon_cache[query] = taxon_id
        self.taxon_resolved.emit(query, taxon_id)
        logging.info("Fetched taxon ID for %s: %s", query, taxon_id)
        return taxon_id

    def run(self) -> None:
        warnings = []
        self._started_at = time.monotonic()
        logging.info(
            "Live search started: organism=%r, exclude=%r, radius=%s km, dates=%s..%s",
            self.organism or None,
            self.exclude or None,
            self.params.get("radius"),
            self.params.get("d1"),
            self.params.get("d2"),
        )
        try:
            self.request_session = pyinaturalist.ClientSession(
                cache_file=self.http_cache_file,
                max_retries=0,
                timeout=10,
            )
            self.request_session.hooks.setdefault("response", []).append(
                self._record_http_response
            )
            maintain_http_cache(
                self.request_session,
                self.http_cache_file,
                self.http_cache_max_bytes,
            )
            raise_if_search_cancelled(self.is_cancelled)
            if self.organism:
                taxon_id = self._get_taxon_id(self.organism)
                if taxon_id:
                    self.params["taxon_id"] = taxon_id
                else:
                    warnings.append(
                        f"Could not find taxon ID for {self.organism}. "
                        "The search was run without that organism filter."
                    )

            if self.exclude:
                exclude_taxon_id = self._get_taxon_id(self.exclude)
                if exclude_taxon_id:
                    self.params["not_in_taxon_id"] = exclude_taxon_id
                else:
                    warnings.append(
                        f"Could not find taxon ID for {self.exclude}. "
                        "The search was run without that exclusion filter."
                    )

            observations, error = fetch_all_observation_pages(
                self.params,
                progress_callback=self.progress.emit,
                api_call_callback=self._record_observation_page,
                cancel_callback=self.is_cancelled,
                request_session=self.request_session,
            )
            self._observation_count = len(observations)
            raise_if_search_cancelled(self.is_cancelled)
            if error:
                self._log_search_summary("failed")
                self.search_failed.emit(error)
                return
            self._log_search_summary("completed")
            self.search_finished.emit(observations, warnings)
        except SearchCancelled:
            self._log_search_summary("cancelled")
            self.search_cancelled.emit()
        except Exception as e:
            logging.exception("Live iNaturalist search worker failed")
            self._log_search_summary("failed")
            self.search_failed.emit(str(e))
        finally:
            if self.request_session is not None:
                maintain_http_cache(
                    self.request_session,
                    self.http_cache_file,
                    self.http_cache_max_bytes,
                )
                self.request_session.close()
                self.request_session = None


class LocalSearchWorker(QThread):
    """Background worker for local DuckDB searches."""

    progress = pyqtSignal(str)
    search_finished = pyqtSignal(object, int)
    search_failed = pyqtSignal(str)
    search_cancelled = pyqtSignal()

    def __init__(
        self,
        parquet_path: str,
        date_from: str,
        date_to: str,
        lat: float,
        lon: float,
        radius: float,
        taxon_ids: list[int] | None = None,
        exclude_taxon_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.date_from = date_from
        self.date_to = date_to
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.taxon_ids = taxon_ids or []
        self.exclude_taxon_ids = exclude_taxon_ids or []
        self._cancel_requested = False
        self._connection = None
        self._connection_mutex = QMutex()

    def _set_connection(self, connection) -> None:
        with QMutexLocker(self._connection_mutex):
            self._connection = connection

    def cancel(self) -> None:
        """Cancel the search and interrupt an active DuckDB query."""
        self._cancel_requested = True
        self.requestInterruption()
        with QMutexLocker(self._connection_mutex):
            if self._connection is not None:
                try:
                    self._connection.interrupt()
                except Exception as e:
                    logging.debug("Could not interrupt DuckDB query: %s", e)

    def is_cancelled(self) -> bool:
        return self._cancel_requested or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            observations, estimated_count = run_local_observation_query(
                self.parquet_path,
                self.date_from,
                self.date_to,
                self.lat,
                self.lon,
                self.radius,
                self.taxon_ids,
                self.exclude_taxon_ids,
                progress_callback=self.progress.emit,
                cancel_callback=self.is_cancelled,
                connection_callback=self._set_connection,
            )
            raise_if_search_cancelled(self.is_cancelled)
            self.search_finished.emit(observations, estimated_count)
        except SearchCancelled:
            self.search_cancelled.emit()
        except Exception as e:
            if self.is_cancelled():
                self.search_cancelled.emit()
            else:
                logging.error("Local search worker failed: %s", e)
                self.search_failed.emit(str(e))


def run_local_observation_query(
    parquet_path: str,
    date_from: str,
    date_to: str,
    lat: float,
    lon: float,
    radius: float,
    taxon_ids: list[int] | None = None,
    exclude_taxon_ids: list[int] | None = None,
    progress_callback=None,
    cancel_callback=None,
    connection_callback=None,
) -> tuple[list[dict[str, Any]], int]:
    """Query observations.parquet for local observations."""
    taxon_ids = taxon_ids or []
    exclude_taxon_ids = exclude_taxon_ids or []
    escaped_path = parquet_path.replace("'", "''")
    lat_min, lat_max, lon_min, lon_max = calculate_local_search_bounds(lat, lon, radius)
    con = None
    try:
        raise_if_search_cancelled(cancel_callback)
        con = duckdb.connect()
        if connection_callback:
            connection_callback(con)
        raise_if_search_cancelled(cancel_callback)
        if progress_callback:
            progress_callback("Checking observations.parquet schema...")
        schema = con.execute(f"DESCRIBE SELECT * FROM '{escaped_path}'").fetchall()
        raise_if_search_cancelled(cancel_callback)
        column_names = [row[0].lower() for row in schema]
        required_columns = [
            "eventdate",
            "decimallatitude",
            "decimallongitude",
            "taxonid",
        ]
        missing_columns = [col for col in required_columns if col not in column_names]
        if missing_columns:
            raise ValueError(
                "Missing required columns in observations.parquet: "
                + ", ".join(missing_columns)
                + ". The file must include eventDate, decimalLatitude, "
                "decimalLongitude, and taxonID."
            )

        base_where = """
            WHERE eventDate BETWEEN ? AND ?
            AND decimalLatitude BETWEEN ? AND ?
            AND decimalLongitude BETWEEN ? AND ?
        """
        filter_sql = ""
        if taxon_ids:
            filter_sql += (
                " AND taxonID IN (" + ",".join(str(id) for id in taxon_ids) + ")"
            )
        if exclude_taxon_ids:
            filter_sql += (
                " AND taxonID NOT IN ("
                + ",".join(str(id) for id in exclude_taxon_ids)
                + ")"
            )

        bbox_params = [date_from, date_to, lat_min, lat_max, lon_min, lon_max]
        if progress_callback:
            progress_callback("Estimating result count...")
        count_query = f"SELECT COUNT(*) FROM '{escaped_path}' {base_where} {filter_sql}"
        _count_row = con.execute(count_query, bbox_params).fetchone()
        raise_if_search_cancelled(cancel_callback)
        estimated_count = _count_row[0] if _count_row else 0

        if progress_callback:
            progress_callback(
                f"Executing DuckDB query (estimated {estimated_count:,} candidates)..."
            )

        # Use DuckDB-native math instead of a Python UDF so large scans do not
        # call back into Python once per row.
        query = f"""
            SELECT id, decimalLatitude, decimalLongitude, eventDate, taxonID
            FROM '{escaped_path}'
            {base_where}
            {filter_sql}
            AND 2 * 6371 * asin(
                sqrt(
                    pow(sin(radians(decimalLatitude - ?) / 2), 2)
                    + cos(radians(?)) * cos(radians(decimalLatitude))
                    * pow(sin(radians(decimalLongitude - ?) / 2), 2)
                )
            ) <= ?
        """
        query_params = bbox_params + [lat, lat, lon, radius]
        results = con.execute(query, query_params).fetchall()
        raise_if_search_cancelled(cancel_callback)

        if progress_callback:
            progress_callback("Processing results...")
        observations = []
        for index, row in enumerate(results):
            if index % 1000 == 0:
                raise_if_search_cancelled(cancel_callback)
            observations.append(
                {
                    "id": row[0],
                    "decimalLatitude": row[1],
                    "decimalLongitude": row[2],
                    "eventDate": row[3],
                    "taxonID": row[4],
                }
            )
        raise_if_search_cancelled(cancel_callback)
        return observations, estimated_count
    finally:
        if connection_callback:
            connection_callback(None)
        if con is not None:
            con.close()


class DatabaseProgressTracker:
    """Track database operation progress without impacting performance."""

    def __init__(self, progress_widget: "EnhancedProgressWidget") -> None:
        self.progress_widget = progress_widget
        self.operation_start = None
        self.last_update = None
        self.estimated_total = None

    def start_operation(
        self, operation_name: str, estimated_total: int | None = None
    ) -> None:
        """Start tracking a database operation."""
        self.operation_start = time.time()
        self.last_update = self.operation_start
        self.estimated_total = estimated_total
        self.progress_widget.start_progress(
            estimated_total or 0, f"Starting {operation_name}..."
        )

    def update_progress(
        self, current_count: int | None = None, message: str | None = None
    ) -> None:
        """Update progress during database operation."""
        current_time = time.time()

        if current_count is not None and self.estimated_total:
            progress = min(100, int((current_count / self.estimated_total) * 100))
            self.progress_widget.update_progress(progress, message)
        else:
            self.progress_widget.update_progress(message=message)

        self.last_update = current_time

    def finish_operation(
        self, final_count: int | None = None, message: str | None = None
    ) -> None:
        """Finish tracking the operation."""
        if final_count is not None:
            self.progress_widget.finish_progress(
                f"{message or 'Operation completed'}: {final_count} results"
            )
        else:
            self.progress_widget.finish_progress(message or "Operation completed")


class CustomSplashScreen(QSplashScreen):
    """Splash screen that scales to fit the screen without overflowing."""

    def __init__(self, image_path: str, parent=None, scale_factor: float = 2.2) -> None:
        # Load the base image
        original_pixmap = QPixmap(image_path)

        # Get the screen geometry
        _primary = QApplication.primaryScreen()
        if _primary is None:
            logging.warning(
                "No primary screen detected; using fallback splash dimensions"
            )
            screen_x, screen_y, screen_w, screen_h = 0, 0, 1920, 1080
        else:
            _geom = _primary.geometry()
            screen_x, screen_y = _geom.x(), _geom.y()
            screen_w, screen_h = _geom.width(), _geom.height()

        # Compute target size based on desired scale factor
        target_w = original_pixmap.width() * scale_factor
        target_h = original_pixmap.height() * scale_factor

        # If too large, shrink so it never exceeds screen boundaries
        if target_w > screen_w or target_h > screen_h:
            scale_ratio = min(
                screen_w / original_pixmap.width(), screen_h / original_pixmap.height()
            )
            target_w = original_pixmap.width() * scale_ratio
            target_h = original_pixmap.height() * scale_ratio

        # Scale with smooth filtering
        scaled_pixmap = original_pixmap.scaled(
            int(target_w),
            int(target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        super().__init__(scaled_pixmap)

        # Do not make the splash globally topmost. On Windows that can place
        # native startup dialogs behind it and prevent the user from answering
        # them or reaching other applications.
        self.setWindowFlags(SPLASH_WINDOW_FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        # Center on screen
        self.move(
            screen_x + (screen_w - self.width()) // 2,
            screen_y + (screen_h - self.height()) // 2,
        )

        # Status label
        self.status_label = QLabel(self)
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
            }
        """
        )
        self.status_label.setText("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Bottom-centered position
        self.status_label.resize(self.width() - 20, 50)
        self.status_label.move(10, self.height() - 70)

        self.show()
        QApplication.processEvents()

    def update_status(self, message: str) -> None:
        self.status_label.setText(message)
        QApplication.processEvents()


class PlaceSearchLineEdit(QLineEdit):
    """A search field whose Enter key cannot accept its containing dialog."""

    search_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.search_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MapDialog(QDialog):
    """Interactive map dialog for setting coordinates and radius using matplotlib"""

    def __init__(
        self,
        parent: "INatSeasonalVisualizer | None" = None,
        lat: float = 37.7749,
        lon: float = -122.4194,
        radius: float = 10,
    ) -> None:
        super().__init__(parent)
        self._main_window = parent
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.place_search_worker: PlaceSearchWorker | None = None
        self.place_search_results: list[dict[str, Any]] = []

        # Worker setup
        working_dir = parent.working_dir if parent else os.getcwd()
        self.worker = TileLoaderWorker(working_dir)
        self.worker.view_ready.connect(self.on_view_ready)
        self.worker.network_error.connect(self.on_network_error)
        self.worker.network_recovered.connect(self.on_network_recovered)
        self.worker.retry_complete.connect(self.request_tiles_for_current_view)
        self.worker.tiles_skipped.connect(self.on_tiles_skipped)
        self.worker.start()

        self.job_counter = 0
        self.current_job_id = 0

        self.setWindowTitle("Interactive Map - Set Location")
        self.setModal(True)

        _primary = QApplication.primaryScreen()
        if _primary is not None:
            screen = _primary.geometry()
            self.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
        else:
            logging.warning("No primary screen detected; using fallback dialog size")
            self.resize(1200, 800)

        # Create layout
        layout = QVBoxLayout(self)

        # Explicit place search avoids sending a request for every keystroke.
        search_layout = QHBoxLayout()
        self.place_search_input = PlaceSearchLineEdit()
        self.place_search_input.setPlaceholderText(
            "Search for a city, country, or park..."
        )
        self.place_search_input.setClearButtonEnabled(True)
        self.place_search_input.search_requested.connect(self.start_place_search)
        self.place_search_button = QPushButton("Search")
        self.place_search_button.clicked.connect(self.start_place_search)
        search_layout.addWidget(self.place_search_input, stretch=1)
        search_layout.addWidget(self.place_search_button)
        layout.addLayout(search_layout)

        self.place_search_status_label = QLabel()
        self.place_search_status_label.hide()
        layout.addWidget(self.place_search_status_label)

        self.place_search_list = QListWidget()
        self.place_search_list.setMaximumHeight(170)
        self.place_search_list.currentRowChanged.connect(
            self.select_place_search_result
        )
        self.place_search_list.hide()
        layout.addWidget(self.place_search_list)

        # Controls layout
        controls_layout = QHBoxLayout()

        instructions = QLabel("Interactive map")
        instructions.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(instructions)

        radius_label = QLabel("Radius (km):")
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(1, 1000)
        self.radius_spinbox.setValue(int(radius))
        self.radius_spinbox.valueChanged.connect(self.update_radius)

        # Zoom buttons
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(lambda: self.zoom_map(1.0 / 1.5))

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(lambda: self.zoom_map(1.5))

        self.coord_label = QLabel(f"Lat: {lat:.4f}, Lon: {lon:.4f}")

        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            "color: red; font-weight: bold; margin-left: 10px;"
        )
        self.status_label.hide()

        controls_layout.addWidget(radius_label)
        controls_layout.addWidget(self.radius_spinbox)
        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(zoom_out_btn)
        controls_layout.addWidget(self.coord_label)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumSize(800, 600)
        layout.addWidget(self.canvas, stretch=1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Timer for debounced reloads
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200)  # 200ms debounce
        self.update_timer.timeout.connect(self.request_tiles_for_current_view)

        # Dedicated timer for retries after network suspension
        self.retry_timer = QTimer()
        self.retry_timer.setSingleShot(True)
        self.retry_timer.timeout.connect(self.request_tiles_for_current_view)

        # Initialize the map
        self.ax.set_xlim(self.lon - 5, self.lon + 5)
        self.ax.set_ylim(self.lat - 5, self.lat + 5)

        self.ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
        self.ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
        self.ax.set_title(
            "Interactive Map - Click to set location, right-click (or Ctrl+Click) to pan. Scroll or use buttons to zoom.",
            fontsize=14,
            fontweight="bold",
        )
        self.ax.grid(False)

        # Overlays - adjust marker size for 4K visibility if needed (using scale_factor from parent if available)
        marker_size = 12 * (
            parent.scale_factor if parent and hasattr(parent, "scale_factor") else 1.0
        )
        (self.marker,) = self.ax.plot(
            [],
            [],
            "ro",
            markersize=marker_size,
            markeredgecolor="darkred",
            markeredgewidth=3,
            zorder=10,
        )
        (self.circle,) = self.ax.plot([], [], "r-", linewidth=3, alpha=0.8, zorder=5)
        self.update_overlays()

        # OpenStreetMap attribution overlay
        self.ax.text(
            0.99,
            0.01,
            "© OpenStreetMap contributors",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=self.ax.transAxes,
            color="#333333",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
                boxstyle="round,pad=0.3",
            ),
            zorder=20,
        )

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_click)  # type: ignore[arg-type]
        self.canvas.mpl_connect("scroll_event", self.on_scroll)  # type: ignore[arg-type]

        # Pan support
        self._is_panning = False
        self._last_pan_pos: tuple[float, float] | None = None
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)  # type: ignore[arg-type]
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)  # type: ignore[arg-type]
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)  # type: ignore[arg-type]

        # Trigger initial load
        self.request_tiles_for_current_view()

    def _stop_workers(self) -> None:
        if self.place_search_worker and self.place_search_worker.isRunning():
            self.place_search_worker.cancel()
            self.place_search_worker.wait()
        if self.worker.isRunning():
            self.worker.stop()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self._stop_workers()
        super().closeEvent(a0)

    def start_place_search(self) -> None:
        query = self.place_search_input.text().strip()
        if not query:
            self.place_search_status_label.setText("Enter a place name to search.")
            self.place_search_status_label.show()
            self.place_search_list.hide()
            return
        if self.place_search_worker and self.place_search_worker.isRunning():
            return

        self.place_search_results = []
        self.place_search_list.clear()
        self.place_search_list.hide()
        self.place_search_status_label.setText(f'Searching for "{query}"...')
        self.place_search_status_label.show()
        self.place_search_button.setEnabled(False)

        cache_file = (
            self._main_window.http_cache_file
            if self._main_window and hasattr(self._main_window, "http_cache_file")
            else None
        )
        self.place_search_worker = PlaceSearchWorker(query, cache_file)
        self.place_search_worker.results_ready.connect(self.on_place_search_results)
        self.place_search_worker.search_failed.connect(self.on_place_search_failed)
        self.place_search_worker.finished.connect(self.on_place_search_finished)
        self.place_search_worker.start()

    def on_place_search_results(self, results: list[dict[str, Any]]) -> None:
        self.place_search_results = results
        self.place_search_list.clear()
        if not results:
            self.place_search_status_label.setText("No matching places found.")
            self.place_search_status_label.show()
            self.place_search_list.hide()
            return

        for place in results:
            self.place_search_list.addItem(place["display_name"])
        self.place_search_status_label.setText(
            "Select a result to move the map. Your radius will stay unchanged."
        )
        self.place_search_status_label.show()
        self.place_search_list.setCurrentRow(-1)
        self.place_search_list.show()
        self.place_search_list.setFocus()

    def on_place_search_failed(self, message: str) -> None:
        logging.warning("Place search could not be completed: %s", message)
        self.place_search_status_label.setText(
            "Place search is unavailable. Check your network connection and try again."
        )
        self.place_search_status_label.show()
        self.place_search_list.hide()

    def on_place_search_finished(self) -> None:
        self.place_search_button.setEnabled(True)

    def select_place_search_result(self, row: int) -> None:
        if row < 0 or row >= len(self.place_search_results):
            return
        place = self.place_search_results[row]
        self.lat = place["lat"]
        self.lon = place["lon"]
        west, east, south, north = place_result_view_limits(place)
        self.ax.set_xlim(west, east)
        self.ax.set_ylim(south, north)
        self.update_overlays()
        self.canvas.draw_idle()
        self.place_search_input.setText(place["display_name"])
        self.place_search_status_label.setText(
            f'Showing "{place["display_name"]}". Click the map to fine-tune the point.'
        )
        self.place_search_list.hide()
        self.request_tiles_for_current_view()

    def request_tiles_for_current_view(self) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        # Calculate optimal zoom
        try:
            bbox = self.ax.get_window_extent().transformed(
                self.figure.dpi_scale_trans.inverted()
            )
            width_px = bbox.width * self.figure.dpi
        except Exception:
            width_px = 800

        lon_span = abs(x1 - x0)
        if lon_span == 0 or width_px <= 0:
            return

        # Target: pixel density.
        target_scale = 1.5
        needed_2z = (target_scale * width_px * 360.0) / (256.0 * max(0.0001, lon_span))
        zoom = int(math.log2(needed_2z)) if needed_2z > 0 else 0
        zoom = max(0, min(zoom, 15))  # Changed max zoom to 15

        # Check tile budget (now pushed down to Worker for uncached tile budgeting)
        # We just send the target zoom and boundary.
        self.job_counter += 1
        self.current_job_id = self.job_counter
        self.worker.request_view(
            self.current_job_id, zoom, float(x0), float(x1), float(y0), float(y1)
        )

    def on_network_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.show()

    def on_network_recovered(self) -> None:
        self.status_label.hide()
        self.retry_timer.stop()

    def on_tiles_skipped(self) -> None:
        """Called when some tiles were skipped due to network suspension. Schedule a retry."""
        # Schedule a retry after the 15-second suspension period ends.
        # We use 16 seconds to be safe. We always restart to sync with any back-off extension.
        self.retry_timer.start(16000)

    def on_view_ready(
        self, job_id: int, composite, extent: tuple[float, float, float, float]
    ) -> None:
        if job_id < self.current_job_id:
            return

        for img in list(self.ax.images):
            img.remove()

        self.ax.imshow(np.array(composite), extent=extent, origin="upper", alpha=0.9)
        self.ax.set_aspect("equal", adjustable="box")

        # Redraw overlays
        self.update_overlays()
        self.canvas.draw_idle()

    def update_overlays(self) -> None:
        self.marker.set_data([self.lon], [self.lat])

        circle_lons = []
        circle_lats = []
        for angle in np.linspace(0, 2 * np.pi, 100):
            lat_offset = self.radius / 111.0
            lon_offset = self.radius / (111.0 * np.cos(np.radians(self.lat)))
            circle_lats.append(self.lat + lat_offset * np.sin(angle))
            circle_lons.append(self.lon + lon_offset * np.cos(angle))

        self.circle.set_data(circle_lons, circle_lats)
        self.coord_label.setText(f"Lat: {self.lat:.4f}, Lon: {self.lon:.4f}")

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax:
            return
        xdata, ydata = event.xdata, event.ydata
        if event.button == 1 and event.key not in ("control", "ctrl"):
            if xdata is None or ydata is None:
                return
            self.lat = ydata
            self.lon = xdata
            self.update_overlays()
            self.canvas.draw_idle()

    def zoom_map(self, scale: float) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        # Center of view
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        w = x1 - x0
        h = y1 - y0

        new_w = w * scale
        new_h = h * scale

        new_x0 = cx - new_w / 2
        new_x1 = cx + new_w / 2
        new_y0 = cy - new_h / 2
        new_y1 = cy + new_h / 2

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()
        self.update_timer.start()

    def update_radius(self, value: int) -> None:
        self.radius = value
        self.update_overlays()
        self.canvas.draw_idle()

    def accept(self) -> None:
        if self._main_window:
            self._main_window.lat_input.setText(f"{self.lat:.4f}")
            self._main_window.lon_input.setText(f"{self.lon:.4f}")
            self._main_window.radius_input.setText(str(self.radius))
        self._stop_workers()
        super().accept()

    def reject(self) -> None:
        self._stop_workers()
        super().reject()

    def on_mouse_press(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax:
            return
        # Pan on: Right Click (3), OR Left Click (1) with Control OR Shift
        is_pan_click = (event.button == 3) or (
            event.button == 1 and event.key in ("control", "ctrl", "shift")
        )

        xdata, ydata = event.xdata, event.ydata
        if is_pan_click and xdata is not None and ydata is not None:
            self._is_panning = True
            self._last_pan_pos = (xdata, ydata)
            try:
                self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            except Exception:
                logging.debug("Failed to set cursor")

    def on_mouse_release(self, _event: MouseEvent) -> None:
        if self._is_panning:
            self._is_panning = False
            self._last_pan_pos = None
            try:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass
            self.update_timer.start()

    def on_mouse_move(self, event: MouseEvent) -> None:
        if not self._is_panning or event.inaxes != self.ax:
            return
        if self._last_pan_pos is None:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        dx = xdata - self._last_pan_pos[0]
        dy = ydata - self._last_pan_pos[1]

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self.canvas.draw_idle()

    def on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return

        scale = 1.0 / 1.5 if event.button == "up" else 1.5

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        w, h = x1 - x0, y1 - y0

        new_w = w * scale
        new_h = h * scale

        relx = (mx - x0) / w
        rely = (my - y0) / h

        new_x0 = mx - relx * new_w
        new_x1 = new_x0 + new_w
        new_y0 = my - rely * new_h
        new_y1 = new_y0 + new_h

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()
        self.update_timer.start()


class EnhancedProgressWidget(QWidget):
    """Enhanced progress widget with detailed status information."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self._layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setVisible(False)
        self.status_text.setReadOnly(True)
        self.status_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._layout.addWidget(self.status_text)

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)

        # Progress tracking
        self.start_time = None
        self.last_update = None
        self.current_step = 0
        self.total_steps = 0
        self.status_messages = []

        # Create database progress tracker
        self.db_tracker = DatabaseProgressTracker(self)

    def start_progress(
        self, total_steps: int = 0, message: str = "Starting operation..."
    ) -> None:
        """Start progress tracking."""
        self.start_time = time.time()
        self.last_update = self.start_time
        self.current_step = 0
        self.total_steps = total_steps
        self.status_messages = [f"[{self._format_time()}] {message}"]

        if total_steps > 0:
            self.progress_bar.setRange(0, total_steps)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate

        self.progress_bar.setVisible(True)
        self.status_text.setVisible(True)
        self._update_display()
        self.timer.start(100)  # Update every 100ms

    def update_progress(
        self, step: int | None = None, message: str | None = None
    ) -> None:
        """Update progress."""
        current_time = time.time()

        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if message:
            self.status_messages.append(f"[{self._format_time()}] {message}")
            # Keep only last 10 messages
            if len(self.status_messages) > 10:
                self.status_messages = self.status_messages[-10:]

        self.last_update = current_time
        self._update_display()

    def update_percentage(self, percentage: int, message: str) -> None:
        """Switch an indeterminate operation to percentage-based progress."""
        if self.total_steps != 100:
            self.total_steps = 100
            self.progress_bar.setRange(0, 100)
        self.update_progress(max(0, min(100, percentage)), message)

    def finish_progress(self, message: str = "Operation completed") -> None:
        """Finish progress tracking."""
        self.timer.stop()
        self.status_messages.append(f"[{self._format_time()}] {message}")
        self._update_display()

        # Hide after a delay
        QTimer.singleShot(2000, self.hide_progress)

    def hide_progress(self) -> None:
        """Hide progress widget."""
        self.progress_bar.setVisible(False)
        self.status_text.setVisible(False)

    def _format_time(self) -> str:
        """Format current time."""
        return datetime.now().strftime("%H:%M:%S")

    def _update_display(self) -> None:
        """Update the display."""
        if self.total_steps > 0:
            self.progress_bar.setValue(self.current_step)

        # Update status text
        status_display = "\n".join(self.status_messages)

        # Add timing information
        if self.start_time:
            elapsed = time.time() - self.start_time
            if self.total_steps > 0 and self.current_step > 0:
                eta = (elapsed / self.current_step) * (
                    self.total_steps - self.current_step
                )
                status_display += f"\n[{self._format_time()}] Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
            else:
                status_display += f"\n[{self._format_time()}] Elapsed: {elapsed:.1f}s"

        self.status_text.setPlainText(status_display)

    def update_status(self) -> None:
        """Timer callback for status updates."""
        if self.start_time:
            self._update_display()


def check_environment() -> str | None:
    """Check if the environment is correctly set up, return error message if not.

    Only conditions that genuinely prevent the app from running are treated as
    fatal (a missing required package, or a conflicting PyQt5 install). Version
    mismatches are logged as warnings rather than blocking startup, so the app
    runs on Windows, macOS, and Linux regardless of how the dependencies were
    installed (conda, pip, or system packages).
    """
    errors = []
    warnings: list[str] = []

    # Check for PyQt5 conflict (PyQt5 and PyQt6 cannot coexist in one process).
    if "PyQt5" in sys.modules:
        errors.append(
            "PyQt5 detected in sys.modules. Uninstall PyQt5 to avoid conflicts with PyQt6."
        )

    # Check matplotlib backend
    if matplotlib.get_backend() != "QtAgg":
        warnings.append(
            f"Unexpected matplotlib backend: {matplotlib.get_backend()} (expected QtAgg)."
        )

    # Required packages. Versions are the ones the app was developed against;
    # mismatches are warnings, not hard failures.
    required_packages = {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "pyarrow": "17.0.0",
        "matplotlib": "3.9.2",
        "pyinaturalist": "0.19.0",
        "PyQt6": "6.8.1",
        "duckdb": None,  # Version not enforced
    }

    for pkg, expected_version in required_packages.items():
        try:
            module = importlib.import_module(pkg)
            if expected_version and hasattr(module, "__version__"):
                if module.__version__ != expected_version:
                    warnings.append(
                        f"{pkg} version {module.__version__} differs from the "
                        f"tested version {expected_version}."
                    )
        except ImportError:
            errors.append(f"Missing required package: {pkg}")

    for warning in warnings:
        logging.warning("Environment check: %s", warning)

    if errors:
        error_message = (
            "Environment setup issues detected:\n\n"
            + "\n\n".join(errors)
            + "\n\nInstall the required dependencies, e.g.:\n"
            "    pip install -r requirements.txt\n\n"
            "After fixing the environment, restart the program."
        )
        return error_message
    return None


class INatSeasonalVisualizer(QMainWindow):
    """Main application class for iNaturalist Seasonal Visualizer"""

    def get_most_recent_date(self) -> str:
        """Get the most recent date from the observations.parquet file."""
        if not os.path.exists(self.observations_file):
            return datetime.now().strftime("%Y-%m-%d")

        con = None
        try:
            con = duckdb.connect()
            schema = con.execute(
                f"DESCRIBE SELECT * FROM '{self.observations_file}'"
            ).fetchall()
            column_names = [row[0].lower() for row in schema]
            if "eventdate" not in column_names:
                logging.warning(
                    "eventDate column not found in observations.parquet. Cannot get most recent date."
                )
                return datetime.now().strftime("%Y-%m-%d")

            result = con.execute(
                f"SELECT MAX(eventDate) FROM '{self.observations_file}'"
            ).fetchone()
            if result and result[0]:
                # The date might be a datetime object, so we format it
                return pd.to_datetime(result[0]).strftime("%Y-%m-%d")
            else:
                return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            logging.error(f"Failed to get most recent date from database: {e!s}")
            return datetime.now().strftime("%Y-%m-%d")
        finally:
            if con is not None:
                con.close()

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        radius: float | None = None,
        scale_factor: float = 1.0,
        splash_screen=None,
        working_dir: str | Path | None = None,
        http_cache_max_mb: int | None = None,
    ) -> None:
        super().__init__()
        self.splash_screen = splash_screen
        self.settings = QSettings("xAI", "iNatSeasonalVisualizer")
        self.scale_factor = scale_factor

        # Robustly load font sizes with safe positional type argument
        self.app_font_size = self.settings.value("app_font_size", 12, int)
        self.graph_font_size = self.settings.value("graph_font_size", 12, int)

        self.api_call_count = 0  # Initialize API call counter
        self.last_plot_args = None  # Initialize plot arguments cache
        self.api_search_worker = None
        self.local_search_worker = None
        self.database_update_worker = None
        # In-memory cache for taxon IDs and descendants
        # Persisted to taxon_cache.json to minimize API calls and avoid recomputing descendants
        self.taxon_cache = {}
        resolved_working_dir = (
            ensure_application_data_dir() if working_dir is None else Path(working_dir)
        )
        resolved_working_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = str(resolved_working_dir)
        self.http_cache_file = os.path.join(self.working_dir, HTTP_CACHE_FILE_NAME)
        self.database_stats_cache_file = os.path.join(
            self.working_dir, DATABASE_STATS_CACHE_FILE_NAME
        )
        self.http_cache_max_bytes = configured_http_cache_max_bytes(http_cache_max_mb)
        self.taxon_cache_file = os.path.join(self.working_dir, "taxon_cache.json")
        self.descendant_taxons_file = os.path.join(
            self.working_dir, "descendant_taxons.txt"
        )
        self.taxonomy_file = os.path.join(self.working_dir, "taxonomy.parquet")
        self.observations_file = os.path.join(self.working_dir, "observations.parquet")
        self.local_database_available = os.path.exists(self.observations_file)

        if self.splash_screen:
            self.splash_screen.update_status("Maintaining API response cache...")
            QApplication.processEvents()
        maintain_legacy_pyinaturalist_cache(
            self.http_cache_file, self.http_cache_max_bytes
        )

        # Initialize statusBar and enhanced progress widget early for download_missing_files
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.enhanced_progress = EnhancedProgressWidget()

        # Add enhanced progress widget to a temporary layout for visibility during downloads
        self.temp_widget = QWidget()
        self.temp_layout = QVBoxLayout(self.temp_widget)
        self.temp_layout.addWidget(self.enhanced_progress)
        self.setCentralWidget(self.temp_widget)

        if self.splash_screen:
            self.splash_screen.update_status("Loading taxon cache...")
            QApplication.processEvents()
        self.load_taxon_cache()

        if self.splash_screen:
            self.splash_screen.update_status("Checking for missing data files...")
            QApplication.processEvents()
        self.download_missing_files()  # Offer local database or API-only mode.

        if self.splash_screen:
            self.splash_screen.update_status("Initializing application settings...")
            QApplication.processEvents()
        self.init_args(lat, lon, radius)

        # Load database stats for display on startup
        self.total_observations = 0
        self.unique_taxa = 0
        if self.splash_screen:
            self.splash_screen.update_status("Loading database statistics...")
            QApplication.processEvents()
        self.load_database_stats()
        self.most_recent_date = self.get_most_recent_date()
        # The local snapshot may be months behind live iNaturalist data. Using
        # today is harmless for local queries and keeps live searches current.
        self.default_search_end_date = datetime.now().strftime("%Y-%m-%d")

        if self.splash_screen:
            self.splash_screen.update_status("Building user interface...")
            QApplication.processEvents()
        self.init_ui()

        if self.splash_screen:
            self.splash_screen.update_status("Loading user preferences...")
            QApplication.processEvents()
        self.load_settings()
        self.update_status_bar()
        self.update_api_call_count()

        if self.splash_screen:
            self.splash_screen.update_status("Preparing welcome screen...")
            QApplication.processEvents()
        self.show_placeholder()

    def load_database_stats(self) -> None:
        """Query the parquet file to get total observations and unique taxa."""
        if not os.path.exists(self.observations_file):
            return

        cached_stats = read_database_stats_cache(
            self.database_stats_cache_file,
            self.observations_file,
        )
        if cached_stats is not None:
            self.total_observations, self.unique_taxa = cached_stats
            logging.info(
                "Loaded cached database stats: %s observations, %s unique taxa.",
                self.total_observations,
                self.unique_taxa,
            )
            return

        con = None
        try:
            con = duckdb.connect()
            # Check if taxonID column exists before running the main query
            schema = con.execute(
                f"DESCRIBE SELECT * FROM '{self.observations_file}'"
            ).fetchall()
            column_names = [row[0].lower() for row in schema]
            if "taxonid" not in column_names:
                logging.warning(
                    "taxonID column not found in observations.parquet. Cannot calculate unique taxa."
                )
                _row = con.execute(
                    f"SELECT COUNT(*) FROM '{self.observations_file}'"
                ).fetchone()
                self.total_observations = _row[0] if _row else 0
                self.unique_taxa = "N/A"
                write_database_stats_cache(
                    self.database_stats_cache_file,
                    self.observations_file,
                    self.total_observations,
                    self.unique_taxa,
                )
                return

            result = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT taxonID) FROM '{self.observations_file}'"
            ).fetchone()
            if result:
                self.total_observations = result[0]
                self.unique_taxa = result[1]
                write_database_stats_cache(
                    self.database_stats_cache_file,
                    self.observations_file,
                    self.total_observations,
                    self.unique_taxa,
                )
            logging.info(
                f"Loaded database stats: {self.total_observations} observations, {self.unique_taxa} unique taxa."
            )
        except Exception as e:
            logging.error(f"Failed to load database stats: {e!s}")
            self.total_observations = "Error"
            self.unique_taxa = "Error"
        finally:
            if con is not None:
                con.close()

    def human_readable_size(self, size_bytes: int) -> str:
        """Convert file size in bytes to human-readable format (e.g., KB, MB, GB)."""
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = size_bytes
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.2f} {units[i]}"

    def prompt_for_database_download(
        self, files_to_download: list[tuple[str, dict[str, Any]]]
    ) -> bool:
        """Return whether the user chose to download the optional local database."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("Choose How to Search")
        dialog.setText("The local iNaturalist database is optional.")
        dialog.setInformativeText(database_download_choice_message(files_to_download))
        file_details = "\n".join(
            f"{filename}: {info['description']}" for filename, info in files_to_download
        )
        dialog.setDetailedText(f"Missing files:\n{file_details}")

        download_button = dialog.addButton(
            "Download Local Database", QMessageBox.ButtonRole.AcceptRole
        )
        observations_missing = any(
            filename == "observations.parquet" for filename, _info in files_to_download
        )
        alternative_label = (
            "Use iNaturalist API Only"
            if observations_missing
            else "Continue Without Taxonomy Download"
        )
        api_button = dialog.addButton(
            alternative_label, QMessageBox.ButtonRole.ActionRole
        )
        # A large download should require an intentional choice rather than
        # starting when the user presses Enter or closes the dialog.
        dialog.setDefaultButton(api_button)
        dialog.setEscapeButton(api_button)
        self._exec_startup_dialog(dialog)
        return dialog.clickedButton() is download_button

    def _exec_startup_dialog(self, dialog: QDialog) -> int:
        """Run a startup dialog without allowing the splash to cover it."""
        splash = getattr(self, "splash_screen", None)
        restore_splash = bool(splash is not None and splash.isVisible())
        if restore_splash:
            splash.hide()
            QApplication.processEvents()

        try:
            # The main window is not visible while its constructor is running,
            # so make the choice explicitly application-modal.
            dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            return dialog.exec()
        finally:
            if restore_splash:
                splash.show()
                QApplication.processEvents()

    def prompt_for_database_update(
        self, files_to_update: list[tuple[str, dict[str, Any]]]
    ) -> bool:
        """Return whether the user chose to replace installed database files."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("Local Database Update Available")
        dialog.setText("A newer local iNaturalist database is available.")
        total_bytes = sum(int(info["size"]) for _name, info in files_to_update)
        file_names = ", ".join(filename for filename, _info in files_to_update)
        dialog.setInformativeText(
            f"Download approximately {self.human_readable_size(total_bytes)} now? "
            "The app will replace the installed database only after the new "
            "file finishes downloading successfully."
        )
        dialog.setDetailedText(f"Files to update: {file_names}")
        update_button = dialog.addButton(
            "Download Update", QMessageBox.ButtonRole.AcceptRole
        )
        later_button = dialog.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(later_button)
        dialog.setEscapeButton(later_button)
        self._exec_startup_dialog(dialog)
        return dialog.clickedButton() is update_button

    def check_for_database_updates(self) -> None:
        """Synchronously check for updates, primarily for manual/test callers."""
        INatSeasonalVisualizer._offer_database_updates(
            self, available_database_updates(self.working_dir)
        )

    def start_database_update_check(self) -> None:
        """Start the routine update check after the main window is visible."""
        if not self.local_database_available or self.database_update_worker is not None:
            return
        worker = DatabaseUpdateCheckWorker(self.working_dir)
        self.database_update_worker = worker
        worker.updates_ready.connect(self._offer_database_updates)
        worker.finished.connect(self._on_database_update_worker_done)
        worker.start()

    def _offer_database_updates(
        self, files_to_update: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Offer a completed background update check on the GUI thread."""
        if not files_to_update:
            return
        if not self.prompt_for_database_update(files_to_update):
            logging.info("User deferred the available local database update.")
            return
        self.download_database_files(files_to_update, replacing_existing=True)

    def _on_database_update_worker_done(self) -> None:
        self.database_update_worker = None

    def download_missing_files(self) -> None:
        """Offer to download missing local data without making it mandatory."""
        files_to_download = missing_database_files(self.working_dir)
        self.local_database_available = os.path.exists(self.observations_file)
        if not files_to_download:
            return

        if not self.prompt_for_database_download(files_to_download):
            logging.info(
                "User chose to continue without downloading missing local database files."
            )
            return

        self.download_database_files(files_to_download, replacing_existing=False)

    def download_database_files(
        self,
        files_to_download: list[tuple[str, dict[str, Any]]],
        *,
        replacing_existing: bool,
    ) -> None:
        """Download database files and atomically install each completed file."""

        for filename, info in files_to_download:
            file_path = os.path.join(self.working_dir, filename)
            url = info["url"]
            human_size = self.human_readable_size(info["size"])
            logging.info(f"Downloading {filename} ({human_size}) from {url}")
            self.status_bar.showMessage(f"Downloading {filename} ({human_size})...")

            # Start enhanced progress tracking
            self.enhanced_progress.start_progress(
                100, f"Downloading {filename} ({human_size})"
            )
            QApplication.processEvents()  # Ensure UI updates

            # Download to a temporary file and rename only on success, so an
            # interrupted download never leaves a partial file that later looks
            # complete. os.replace is atomic on the same filesystem on every OS.
            temp_path = file_path + ".part"
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", info["size"]))
                block_size = 8192
                downloaded = 0

                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = min(
                                    100, int((downloaded / total_size) * 100)
                                )
                                self.enhanced_progress.update_progress(
                                    progress,
                                    f"Downloaded {self.human_readable_size(downloaded)} / {human_size}",
                                )
                                QApplication.processEvents()  # Update progress bar in real-time

                if total_size > 0 and downloaded != total_size:
                    raise OSError(
                        f"download ended at {downloaded} of {total_size} bytes"
                    )

                remote_modified = response.headers.get("last-modified")
                if remote_modified:
                    try:
                        remote_timestamp = parsedate_to_datetime(
                            remote_modified
                        ).timestamp()
                        os.utime(temp_path, (remote_timestamp, remote_timestamp))
                    except (OSError, TypeError, ValueError, OverflowError):
                        logging.debug(
                            "Could not preserve Last-Modified for %s", filename
                        )

                os.replace(temp_path, file_path)
                logging.info(f"Successfully downloaded {filename} to {file_path}")
                self.status_bar.showMessage(f"Downloaded {filename} ({human_size})")
                self.enhanced_progress.finish_progress(
                    f"Successfully downloaded {filename}"
                )
                QApplication.processEvents()

                if filename == "taxonomy.parquet":
                    self.invalidate_descendant_taxon_cache()

            except Exception as e:
                logging.error(f"Failed to download {filename}: {e!s}")
                # Clean up any partial download so the next run retries cleanly.
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass
                self.enhanced_progress.hide_progress()
                failure_outcome = (
                    "The currently installed file was left unchanged. "
                    if replacing_existing
                    else "The app will continue without this file. "
                )
                QMessageBox.warning(
                    self,
                    "Download Error",
                    f"Failed to download {filename}: {e!s}\n\n"
                    f"{failure_outcome}You can use Search with API, or restart "
                    "later to try the download again.\n\n"
                    f"You can also manually download it from {url} and place it "
                    f"in {self.working_dir}.",
                )
                # A network failure is likely to affect the remaining file too;
                # avoid showing the user the same error twice.
                break

            finally:
                QApplication.processEvents()

        self.local_database_available = os.path.exists(self.observations_file)

    def invalidate_descendant_taxon_cache(self) -> None:
        """Remove taxonomy-derived cache entries after installing new taxonomy."""
        stale_keys = [key for key in self.taxon_cache if key.endswith("_descendants")]
        if not stale_keys:
            return
        for key in stale_keys:
            del self.taxon_cache[key]
        self.save_taxon_cache()
        logging.info(
            "Removed %d stale descendant-taxonomy cache entries.", len(stale_keys)
        )

    def init_args(
        self, lat: float | None, lon: float | None, radius: float | None
    ) -> None:
        """Initialize command-line arguments."""
        default_lat = (
            lat if lat is not None else self.settings.value("latitude", 37.7749)
        )
        default_lon = (
            lon if lon is not None else self.settings.value("longitude", -122.4194)
        )
        self.default_lat = f"{float(default_lat):.4f}"
        self.default_lon = f"{float(default_lon):.4f}"
        self.default_radius = (
            radius if radius is not None else self.settings.value("radius", 10)
        )

    def load_taxon_cache(self) -> None:
        """Load taxon cache from JSON file to avoid repeated API calls and taxonomy queries."""
        try:
            if os.path.exists(self.taxon_cache_file):
                with open(self.taxon_cache_file, "r") as f:
                    self.taxon_cache = json.load(f)
                descendant_lists = sum(
                    key.endswith("_descendants") for key in self.taxon_cache
                )
                logging.info(
                    "Loaded taxon cache from %s (%d entries, %d descendant lists)",
                    self.taxon_cache_file,
                    len(self.taxon_cache),
                    descendant_lists,
                )
                for key, value in self.taxon_cache.items():
                    if key.endswith("_descendants"):
                        logging.debug(
                            f"Loaded {len(value)} descendant taxon IDs for {key[:-11]}"
                        )
            else:
                logging.info(f"No taxon cache found at {self.taxon_cache_file}")
        except Exception as e:
            logging.error(f"Failed to load taxon cache: {e!s}")
            QMessageBox.warning(
                self,
                "Warning",
                f"Invalid or corrupted taxon_cache.json in {self.working_dir}. Starting with empty cache.",
            )
            self.taxon_cache = {}

    def save_taxon_cache(self) -> None:
        """Save taxon cache to JSON file for persistence across sessions."""
        try:
            with open(self.taxon_cache_file, "w") as f:
                json.dump(self.taxon_cache, f, indent=2)
            logging.info(f"Saved taxon cache to {self.taxon_cache_file}")
        except Exception as e:
            logging.error(f"Failed to save taxon cache: {e!s}")

    def load_descendant_taxons_from_file(self, query: str) -> list[int] | None:
        """Load descendant taxon IDs from a user-provided file."""
        try:
            if not os.path.exists(self.descendant_taxons_file):
                return None
            with open(self.descendant_taxons_file, "r") as f:
                for line in f:
                    if line.strip().startswith(f"{query}:"):
                        taxon_ids = [
                            int(id.strip())
                            for id in line.split(":")[1].split(",")
                            if id.strip()
                        ]
                        logging.info(
                            f"Loaded {len(taxon_ids)} descendant taxon IDs for {query} from {self.descendant_taxons_file}"
                        )
                        return taxon_ids
            return None
        except Exception as e:
            logging.error(
                f"Failed to load descendant taxons from {self.descendant_taxons_file}: {e!s}"
            )
            return None

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle(f"iNaturalist Seasonal Visualizer v{__version__}")

        # Use the user-provided or default scale factor
        scale_factor = self.scale_factor

        # The window will be launched in a maximized state, so a hardcoded size is not needed.

        # The application font is now set via stylesheet in the toggle_theme method for consistency.

        # Scale matplotlib fonts globally
        matplotlib.rcParams.update(
            {
                "font.size": self.graph_font_size * scale_factor,
                "axes.labelsize": self.graph_font_size * scale_factor,
                "axes.titlesize": (self.graph_font_size + 2)
                * scale_factor,  # Title is slightly larger
                "xtick.labelsize": (self.graph_font_size - 1)
                * scale_factor,  # Ticks are slightly smaller
                "ytick.labelsize": (self.graph_font_size - 1) * scale_factor,
                "legend.fontsize": self.graph_font_size * scale_factor,
            }
        )

        # Central widget and main vertical layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Horizontal layout for sidebar and graph
        self.top_layout = QHBoxLayout()

        # Sidebar for inputs
        self.sidebar = QWidget()
        self.sidebar_layout = QFormLayout(self.sidebar)
        self.top_layout.addWidget(self.sidebar, 1)

        # Input fields
        self.lat_input = QLineEdit(self.default_lat)
        self.lat_input.setPlaceholderText("e.g., 37.7749")
        self.lat_input.setMinimumWidth(100)
        self.lat_input.textChanged.connect(self.parse_coordinate_input)

        self.lon_input = QLineEdit(self.default_lon)
        self.lon_input.setPlaceholderText("e.g., -122.4194")
        self.lon_input.setMinimumWidth(100)
        self.lon_input.textChanged.connect(self.parse_coordinate_input)

        self.radius_input = QLineEdit(str(self.default_radius))
        self.organism_input = QLineEdit()
        self.organism_input.setPlaceholderText("e.g., Boletus")
        self.organism_input.returnPressed.connect(self.run_preferred_search)
        self.exclude_input = QLineEdit()
        self.exclude_input.setPlaceholderText("e.g., Boletus regineus")
        self.date_from = QLineEdit("2000-01-01")
        self.date_to = QLineEdit(self.default_search_end_date)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Daily", "Weekly", "Monthly"])
        self.view_combo.setCurrentIndex(1)  # Set default to Weekly
        self.color_input = QLineEdit(self.settings.value("graph_color", "#1f77b4"))
        self.color_input.setPlaceholderText("e.g., #1f77b4 or blue")
        self.bg_color_input = QLineEdit(self.settings.value("graph_bg_color", ""))
        self.bg_color_input.setPlaceholderText("Theme Default")

        # Create latitude input layout with map button
        lat_layout = QHBoxLayout()
        lat_layout.addWidget(self.lat_input, 2)

        # Map button
        self.map_button = QPushButton("🗺️ Map")
        self.map_button.clicked.connect(self.open_map_dialog)
        self.map_button.setToolTip("Open interactive map to set location")
        lat_layout.addWidget(self.map_button, 1)

        # Add latitude layout to sidebar
        self.sidebar_layout.addRow("Latitude:", lat_layout)
        self.sidebar_layout.addRow("Longitude:", self.lon_input)
        self.sidebar_layout.addRow("Radius (km):", self.radius_input)
        self.sidebar_layout.addRow("Organism:", self.organism_input)
        self.sidebar_layout.addRow("Exclude:", self.exclude_input)
        self.sidebar_layout.addRow("Date From:", self.date_from)
        self.sidebar_layout.addRow("Date To:", self.date_to)
        self.sidebar_layout.addRow("View:", self.view_combo)
        self.sidebar_layout.addRow("Graph Color:", self.color_input)
        self.sidebar_layout.addRow("Graph BG Color:", self.bg_color_input)

        # Graph with local data button
        self.local_search_button = QPushButton(LOCAL_GRAPH_ACTION_TEXT)
        self.local_search_button.clicked.connect(self.local_search)
        if not self.local_database_available:
            self.local_search_button.setText(
                f"{LOCAL_GRAPH_ACTION_TEXT} (Database Not Installed)"
            )
            self.local_search_button.setEnabled(False)
            self.local_search_button.setToolTip(
                "Download observations.parquet to enable fast local searches. "
                f"Use {LIVE_GRAPH_ACTION_TEXT} in the meantime."
            )
        self.sidebar_layout.addWidget(self.local_search_button)

        # Graph with live iNaturalist data button
        self.search_button = QPushButton(LIVE_GRAPH_ACTION_TEXT)
        self.search_button.clicked.connect(self.search_observations)
        self.search_button.setToolTip(
            "Search online using the iNaturalist API; an internet connection is required."
        )
        self.sidebar_layout.addWidget(self.search_button)

        self.cancel_search_button = QPushButton("Cancel Search")
        self.cancel_search_button.clicked.connect(self.cancel_search)
        self.cancel_search_button.setEnabled(False)
        self.cancel_search_button.setToolTip(
            "Stop the current local or live iNaturalist search."
        )
        self.sidebar_layout.addWidget(self.cancel_search_button)

        # Show URL button
        self.show_url_button = QPushButton("Show URL")
        self.show_url_button.clicked.connect(self.show_search_url)
        self.sidebar_layout.addWidget(self.show_url_button)

        # Enhanced progress widget for API, local searches, and downloads
        self.sidebar_layout.addWidget(self.enhanced_progress)

        # History list
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_history_item)
        self.sidebar_layout.addRow("History:", self.history_list)

        # Keep the sidebar from expanding past a reasonable width
        _primary = QGuiApplication.primaryScreen()
        if _primary is not None:
            avail = _primary.availableGeometry()
            self.sidebar.setMaximumWidth(
                int(avail.width() * 0.35)
            )  # tweak 0.30–0.45 as you like
        else:
            logging.warning("No primary screen detected; skipping sidebar width cap")

        # Prevent long history strings from pushing size hints wider
        self.history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.history_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # API call count label
        self.api_call_label = QLabel(f"API Calls: {self.api_call_count}")
        self.sidebar_layout.addWidget(self.api_call_label)

        # Graph area
        try:
            self.figure, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.figure)
            self.top_layout.addWidget(self.canvas, 3)

            # Make sure the plot area can shrink instead of forcing the window wider
            self.canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.canvas.setMinimumSize(0, 0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize plot: {e!s}")
            self.canvas = None
            error_label = QLabel(f"Plot initialization failed: {e!s}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.top_layout.addWidget(error_label, 3)
            return

        # Add top layout to main layout
        self.main_layout.addLayout(self.top_layout)

        # Info bar (moved to bottom)
        self.info_bar = QLabel()
        self.main_layout.addWidget(self.info_bar, stretch=0)

        # Menu bar
        self.init_menu_bar()

        # Status bar (already set in __init__)

        # Apply initial theme
        self.apply_stylesheet()

    def run_preferred_search(self) -> None:
        """Run a local search when available, otherwise use the online API."""
        if self.local_database_available:
            self.local_search()
        else:
            self.search_observations()

    def parse_coordinate_input(self, text: str) -> None:
        """Parse coordinate string (e.g. 'lat, lon') and update inputs."""
        if "," in text:
            parts = text.split(",")
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())

                    # Block signals to prevent recursive updates
                    self.lat_input.blockSignals(True)
                    self.lon_input.blockSignals(True)

                    self.lat_input.setText(f"{lat}")
                    self.lon_input.setText(f"{lon}")

                    self.lat_input.blockSignals(False)
                    self.lon_input.blockSignals(False)
                except ValueError:
                    pass

    def init_menu_bar(self) -> None:
        """Initialize the menu bar."""
        menu_bar = self.menuBar()
        if menu_bar is None:
            logging.warning("menuBar() returned None; skipping menu bar init")
            return

        # File menu
        file_menu = menu_bar.addMenu("File")
        if file_menu is None:
            logging.warning("addMenu('File') returned None; skipping File menu")
            return
        save_action = QAction("Save Settings", self)
        save_action.triggered.connect(self.save_settings)
        file_menu.addAction(save_action)

        fetch_taxon_ids_action = QAction("Fetch Taxon IDs", self)
        fetch_taxon_ids_action.triggered.connect(self.fetch_taxon_ids)
        file_menu.addAction(fetch_taxon_ids_action)

        export_graph = QAction("Export Graph", self)
        export_graph.triggered.connect(self.export_graph)
        file_menu.addAction(export_graph)

        export_data = QAction("Export Data", self)
        export_data.triggered.connect(self.export_data)
        file_menu.addAction(export_data)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        if edit_menu is None:
            logging.warning("addMenu('Edit') returned None; skipping Edit menu")
            return
        theme_action = QAction("Toggle Dark/Light Mode", self)
        theme_action.triggered.connect(
            lambda: self.toggle_theme(
                "light"
                if normalize_theme(self.settings.value("theme", DEFAULT_THEME))
                == "dark"
                else "dark"
            )
        )
        edit_menu.addAction(theme_action)

        color_action = QAction("Choose Graph Color", self)
        color_action.triggered.connect(self.choose_color)
        edit_menu.addAction(color_action)

        bg_color_action = QAction("Choose Graph Background Color", self)
        bg_color_action.triggered.connect(self.choose_bg_color)
        edit_menu.addAction(bg_color_action)

        window_bg_action = QAction("Choose Window Background Color", self)
        window_bg_action.triggered.connect(self.choose_window_bg_color)
        edit_menu.addAction(window_bg_action)

        self.app_font_size_action = QAction(
            f"Change App Font Size ({self.app_font_size}pt)", self
        )
        self.app_font_size_action.triggered.connect(self.change_app_font_size)
        edit_menu.addAction(self.app_font_size_action)

        self.graph_font_size_action = QAction(
            f"Change Graph Font Size ({self.graph_font_size}pt)", self
        )
        self.graph_font_size_action.triggered.connect(self.change_graph_font_size)
        edit_menu.addAction(self.graph_font_size_action)

    def _refresh_font_menu_labels(self) -> None:
        """Update font size menu actions with current values."""
        if getattr(self, "app_font_size_action", None):
            self.app_font_size_action.setText(
                f"Change App Font Size ({self.app_font_size}pt)"
            )
        if getattr(self, "graph_font_size_action", None):
            self.graph_font_size_action.setText(
                f"Change Graph Font Size ({self.graph_font_size}pt)"
            )

    def change_app_font_size(self) -> None:
        """Prompt user to change the application font size."""
        current_size = self.app_font_size
        new_size, ok = QInputDialog.getInt(
            self,
            "App Font Size",
            "Select app font size (pt):",
            value=current_size,
            min=8,
            max=72,
        )
        if ok:
            self.app_font_size = new_size
            self.settings.setValue("app_font_size", new_size)
            self._refresh_font_menu_labels()
            self.apply_stylesheet()

    def change_graph_font_size(self) -> None:
        """Prompt user to change the graph font size."""
        current_size = self.graph_font_size
        new_size, ok = QInputDialog.getInt(
            self,
            "Graph Font Size",
            "Select graph font size (pt):",
            value=current_size,
            min=8,
            max=72,
        )
        if ok:
            self.graph_font_size = new_size
            self.settings.setValue("graph_font_size", new_size)
            self._refresh_font_menu_labels()

            # Update matplotlib settings
            scale_factor = self.scale_factor
            matplotlib.rcParams.update(
                {
                    "font.size": self.graph_font_size * scale_factor,
                    "axes.labelsize": self.graph_font_size * scale_factor,
                    "axes.titlesize": (self.graph_font_size + 2) * scale_factor,
                    "xtick.labelsize": (self.graph_font_size - 1) * scale_factor,
                    "ytick.labelsize": (self.graph_font_size - 1) * scale_factor,
                    "legend.fontsize": self.graph_font_size * scale_factor,
                }
            )

            # Replot if data exists
            if hasattr(self, "last_plot_args") and self.last_plot_args:
                self.plot_observations(**self.last_plot_args)
                if getattr(self, "figure", None):
                    try:
                        self.figure.tight_layout()
                    except Exception:
                        pass
            elif self.canvas:
                self.canvas.draw_idle()

    def apply_stylesheet(self) -> None:
        """Apply the application stylesheet based on current theme and font settings."""
        mode = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
        font_size_pt = int(self.app_font_size * self.scale_factor)
        font_stylesheet = f"font-size: {font_size_pt}pt;"

        custom_bg_color = self.settings.value("graph_bg_color")
        custom_window_bg = self.settings.value("window_bg_color")
        window_bg_color = custom_window_bg or (
            DARK_WINDOW_BACKGROUND if mode == "dark" else LIGHT_BACKGROUND
        )
        graph_bg_color = custom_bg_color or (
            DARK_GRAPH_BACKGROUND if mode == "dark" else LIGHT_BACKGROUND
        )
        window_text_color = self.get_contrasting_text_color(window_bg_color)
        graph_text_color = self.get_contrasting_text_color(graph_bg_color)

        if mode == "dark":
            # UI dark mode
            self.setStyleSheet(
                font_stylesheet
                + f"background-color: {window_bg_color}; color: {window_text_color};"
            )
            self.central_widget.setStyleSheet(
                "QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, "
                "QLabel { background-color: #3e3e3e; color: #ffffff; }"
            )
        else:
            # UI light mode
            self.setStyleSheet(
                font_stylesheet
                + f"background-color: {window_bg_color}; color: {window_text_color};"
            )
            self.central_widget.setStyleSheet(
                "QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, "
                "QLabel { background-color: #ffffff; color: #000000; }"
            )

        if (
            getattr(self, "canvas", None)
            and getattr(self, "figure", None)
            and getattr(self, "ax", None)
        ):
            self.figure.set_facecolor(window_bg_color)
            self.ax.set_facecolor(graph_bg_color)
            self.ax.tick_params(colors=graph_text_color)
            self.ax.xaxis.label.set_color(graph_text_color)
            self.ax.yaxis.label.set_color(graph_text_color)
            self.ax.title.set_color(graph_text_color)
            for spine in self.ax.spines.values():
                spine.set_color(graph_text_color)

            # Placeholder and annotation text is not covered by axis label
            # properties. Recolor existing artists when the user changes theme.
            for text_artist in self.ax.texts:
                text_artist.set_color(graph_text_color)

            legend = self.ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(graph_bg_color)
                legend.get_frame().set_edgecolor(graph_text_color)
                for legend_text in legend.get_texts():
                    legend_text.set_color(graph_text_color)

        _canvas = getattr(self, "canvas", None)
        if _canvas is not None and getattr(self, "figure", None):
            _canvas.draw_idle()

    def toggle_theme(self, mode: str) -> None:
        """Toggle between dark and light mode for UI and graph."""
        self.settings.setValue("theme", mode)
        self.apply_stylesheet()

    def choose_color(self) -> None:
        """Open a color dialog to choose the graph color."""
        color = QColorDialog.getColor(
            QColor(self.color_input.text()), self, "Choose Graph Color"
        )
        if color.isValid():
            self.color_input.setText(color.name())
            self.settings.setValue("graph_color", color.name())
            # Redraw graph if data is loaded
            if self.history_list.property("data"):
                self.search_observations()  # Re-run last search to update color

    def choose_bg_color(self) -> None:
        """Open a color dialog to choose the graph background color."""
        mode = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
        current_color_hex = self.settings.value("graph_bg_color") or (
            DARK_GRAPH_BACKGROUND if mode == "dark" else LIGHT_BACKGROUND
        )
        color = QColorDialog.getColor(
            QColor(current_color_hex), self, "Choose Graph Background Color"
        )

        if color.isValid():
            color_name = color.name()
            self.bg_color_input.setText(color_name)
            self.settings.setValue("graph_bg_color", color_name)
            # Re-apply the current theme to update the background
            self.apply_stylesheet()

    def choose_window_bg_color(self) -> None:
        """Open a color dialog to choose the main window background color."""
        mode = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
        current_color_hex = self.settings.value("window_bg_color") or (
            DARK_WINDOW_BACKGROUND if mode == "dark" else LIGHT_BACKGROUND
        )
        color = QColorDialog.getColor(
            QColor(current_color_hex), self, "Choose Window Background Color"
        )

        if color.isValid():
            color_name = color.name()
            self.settings.setValue("window_bg_color", color_name)
            # Re-apply the current theme to update the background
            self.apply_stylesheet()

    def open_map_dialog(self) -> None:
        """Open the interactive map dialog."""
        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
            radius = float(self.radius_input.text())
        except ValueError:
            # Use defaults if current values are invalid
            lat = float(self.default_lat)
            lon = float(self.default_lon)
            radius = float(self.default_radius)

        # Open map dialog
        dialog = MapDialog(self, lat, lon, radius)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Coordinates are updated in the dialog's accept method
            pass

    def load_settings(self) -> None:
        """Load saved settings."""
        self.lat_input.setText(self.settings.value("latitude", self.default_lat))
        self.lon_input.setText(self.settings.value("longitude", self.default_lon))
        self.radius_input.setText(
            self.settings.value("radius", str(self.default_radius))
        )
        self.color_input.setText(self.settings.value("graph_color", "#1f77b4"))
        self.bg_color_input.setText(self.settings.value("graph_bg_color", ""))

    def save_settings(self) -> None:
        """Save current settings."""
        try:
            self.settings.setValue("latitude", float(self.lat_input.text()))
            self.settings.setValue("longitude", float(self.lon_input.text()))
            self.settings.setValue("radius", float(self.radius_input.text()))
            self.settings.setValue("graph_color", self.color_input.text())
            self.settings.setValue("graph_bg_color", self.bg_color_input.text())
            self.settings.setValue("app_font_size", self.app_font_size)
            self.settings.setValue("graph_font_size", self.graph_font_size)
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        except ValueError as e:
            QMessageBox.warning(self, "Settings", f"Invalid input: {e!s}")

    def update_api_call_count(self) -> None:
        """Update the API call count display."""
        self.api_call_label.setText(f"API Calls: {self.api_call_count}")

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km using Haversine formula."""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def fetch_all_observations(
        self, params: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Synchronously fetch observations for non-interactive callers."""
        self.enhanced_progress.start_progress(0, "Starting API search...")

        def report_progress(message: str, percentage: int | None) -> None:
            self.status_bar.showMessage(message)
            if percentage is None:
                self.enhanced_progress.update_progress(message=message)
            else:
                self.enhanced_progress.update_percentage(percentage, message)

        def count_api_call() -> None:
            self.api_call_count += 1
            self.update_api_call_count()

        request_session = pyinaturalist.ClientSession(
            cache_file=self.http_cache_file,
            max_retries=0,
            timeout=10,
        )
        maintain_http_cache(
            request_session,
            self.http_cache_file,
            self.http_cache_max_bytes,
        )
        try:
            observations, error = fetch_all_observation_pages(
                params,
                progress_callback=report_progress,
                api_call_callback=count_api_call,
                request_session=request_session,
            )
        finally:
            maintain_http_cache(
                request_session,
                self.http_cache_file,
                self.http_cache_max_bytes,
            )
            request_session.close()
        if error:
            self.enhanced_progress.hide_progress()
        else:
            self.enhanced_progress.finish_progress(
                f"API search completed: {len(observations)} observations"
            )
        return observations, error

    def get_taxon_id(self, query: str) -> int | None:
        """Get taxon ID from query, using cache to minimize API calls."""
        if not query:
            return None
        # Check cache to avoid redundant API calls
        if query in self.taxon_cache and isinstance(self.taxon_cache[query], int):
            logging.info(
                f"Retrieved taxon ID for {query} from cache: {self.taxon_cache[query]}"
            )
            return self.taxon_cache[query]
        request_session = None
        try:
            request_session = pyinaturalist.ClientSession(
                cache_file=self.http_cache_file,
                max_retries=0,
                timeout=10,
            )
            maintain_http_cache(
                request_session,
                self.http_cache_file,
                self.http_cache_max_bytes,
            )
            taxa = pyinaturalist.get_taxa(
                q=query,
                limit=1,
                session=request_session,
            )
            self.api_call_count += 1  # Increment API call counter
            self.update_api_call_count()
            if taxa["results"]:
                taxon_id = taxa["results"][0]["id"]
                self.taxon_cache[query] = taxon_id
                self.save_taxon_cache()
                logging.info(f"Fetched taxon ID for {query}: {taxon_id}")
                return taxon_id
            logging.warning(f"No taxon ID found for {query}")
            return None
        except Exception as e:
            error_msg = (
                f"Failed to fetch taxon ID for {query}: {e!s}. "
                "This may be due to anonymous API access or network issues. "
                "Consider setting INATURALIST_APP_ID and INATURALIST_APP_SECRET in ~/.bashrc for higher limits."
            )
            logging.error(error_msg)
            return None
        finally:
            if request_session is not None:
                maintain_http_cache(
                    request_session,
                    self.http_cache_file,
                    self.http_cache_max_bytes,
                )
                request_session.close()

    def get_descendant_taxon_ids(self, query: str, taxon_id: int) -> list[int]:
        """Get all descendant taxon IDs for a given taxon using taxonomy.parquet."""
        cache_key = f"{query}_descendants"
        # Check cache to avoid recomputing descendants from taxonomy.parquet
        if cache_key in self.taxon_cache:
            logging.info(
                f"Retrieved {len(self.taxon_cache[cache_key])} descendant taxon IDs for {query} from cache"
            )
            return self.taxon_cache[cache_key]

        # Try loading from user-provided file
        file_taxons = self.load_descendant_taxons_from_file(query)
        if file_taxons:
            self.taxon_cache[cache_key] = file_taxons
            self.save_taxon_cache()
            return file_taxons

        # Query taxonomy.parquet
        con = None
        try:
            if not os.path.exists(self.taxonomy_file):
                raise FileNotFoundError(
                    f"taxonomy.parquet not found in {self.working_dir}"
                )

            con = duckdb.connect()
            sql = """
                WITH RECURSIVE descendants AS (
                    SELECT id FROM taxonomy WHERE id = ?
                    UNION ALL
                    SELECT t.id FROM taxonomy t
                    JOIN descendants d ON t.parent_id = d.id
                )
                SELECT id FROM descendants
            """
            con.execute(f"CREATE VIEW taxonomy AS SELECT * FROM '{self.taxonomy_file}'")
            results = con.execute(sql, [taxon_id]).fetchall()
            descendants = [row[0] for row in results]

            # Include the parent taxon ID
            if taxon_id not in descendants:
                descendants.append(taxon_id)

            self.taxon_cache[cache_key] = descendants
            self.save_taxon_cache()
            logging.info(
                f"Retrieved {len(descendants)} descendant taxon IDs for {query} from taxonomy.parquet"
            )
            return descendants

        except Exception as e:
            logging.error(
                f"Failed to fetch descendant taxon IDs for {query} from taxonomy.parquet: {e!s}"
            )

            # Fallback dialog
            error_msg = (
                f"Failed to fetch descendant taxon IDs for {query} from taxonomy.parquet: {e!s}. "
                "You can:\n"
                f"1. Ensure taxonomy.parquet is in {self.working_dir} and contains valid data.\n"
                f"2. Create {self.descendant_taxons_file} with:\n"
                f"   {query}: taxon_id1, taxon_id2, ... (e.g., Agaricales: 117159, 48723)\n"
                "3. Continue with only the parent taxon ID (may yield fewer results)."
            )
            reply = QMessageBox.question(
                self,
                "Descendant Taxon IDs Failed",
                error_msg + "\n\nContinue with parent taxon ID?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                logging.info(f"Continuing with parent taxon ID {taxon_id} for {query}")
                return [taxon_id]
            else:
                logging.info(f"Aborting descendant taxon ID fetch for {query}")
                return []
        finally:
            if con is not None:
                con.close()

    def fetch_taxon_ids(self) -> None:
        """Manually fetch taxon IDs for the current organism and save to cache."""
        organism = self.organism_input.text().strip()
        if not organism:
            QMessageBox.warning(self, "Error", "Please enter an organism name.")
            return

        taxon_id = self.get_taxon_id(organism)
        if not taxon_id:
            QMessageBox.critical(
                self, "Error", f"Could not find taxon ID for {organism}."
            )
            return

        self.enhanced_progress.start_progress(
            0, f"Fetching taxon IDs for {organism}..."
        )
        self.status_bar.showMessage(f"Fetching taxon IDs for {organism}...")

        try:
            self.enhanced_progress.update_progress(
                message="Querying taxonomy database..."
            )
            taxon_ids = self.get_descendant_taxon_ids(organism, taxon_id)
            self.enhanced_progress.finish_progress(
                f"Fetched {len(taxon_ids)} taxon IDs for {organism}"
            )
            if taxon_ids:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Fetched {len(taxon_ids)} taxon IDs for {organism}. Saved to {self.taxon_cache_file}.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"No taxon IDs fetched for {organism}. Check logs or try again.",
                )
        except Exception as e:
            self.enhanced_progress.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to fetch taxon IDs: {e!s}")
        finally:
            self.status_bar.showMessage("Ready")

    def show_search_url(self) -> None:
        """Show the iNaturalist URL for the current search parameters."""
        try:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Invalid latitude or longitude. Please enter valid numbers.",
                )
                return

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()

            # Fetch taxon ID for organism
            taxon_id = self.get_taxon_id(organism)

            # Construct URL
            params = {
                "lat": lat,
                "lng": lon,
                "radius": radius,
                "d1": date_from,
                "d2": date_to,
                "subview": "map",
            }
            if taxon_id:
                params["taxon_id"] = taxon_id

            base_url = "https://www.inaturalist.org/observations"
            url = f"{base_url}?{urlencode(params)}"
            QMessageBox.information(
                self,
                "iNaturalist URL",
                f"Search URL:\n{url}\n\nCopy this URL to verify the search on iNaturalist.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate URL: {e!s}")

    def clamp_to_screen(self) -> None:
        """Clamp the window geometry so it stays within the available screen area."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            logging.warning("No screen available; skipping window clamp")
            return
        avail = screen.availableGeometry()

        g = self.frameGeometry()
        w = min(g.width(), avail.width())
        h = min(g.height(), avail.height())

        if g.width() != w or g.height() != h:
            self.resize(w, h)

        # keep top-left inside the available area
        x = max(avail.left(), min(self.x(), avail.right() - w))
        y = max(avail.top(), min(self.y(), avail.bottom() - h))
        self.move(x, y)

    def plot_observations(
        self,
        observations: list[dict[str, Any]],
        lat: float,
        lon: float,
        radius: float,
        organism: str,
        date_from: str,
        date_to: str,
        view: str,
        source: str = "API",
    ) -> bool:
        """Plot observations (from API or local)."""
        # Save arguments for replotting (e.g., when font size changes)
        self.last_plot_args = {
            "observations": observations,
            "lat": lat,
            "lon": lon,
            "radius": radius,
            "organism": organism,
            "date_from": date_from,
            "date_to": date_to,
            "view": view,
            "source": source,
        }

        if not observations:
            self.show_placeholder(f"No {source} observations found.")
            return False

        # Process data, handling timezone-aware timestamps
        dates = []
        for obs in observations:
            date_str = obs.get("eventDate" if source == "Local" else "observed_on")
            if date_str:
                try:
                    # Convert to UTC and make timezone-naive
                    date = pd.to_datetime(date_str, utc=True).tz_localize(None)
                    dates.append(date)
                except Exception as e:
                    logging.warning(
                        f"Skipping invalid date at observation {obs}: {e!s}"
                    )
                    continue
            else:
                logging.warning(f"Skipping observation with missing date: {obs}")
                continue

        if not dates:
            self.show_placeholder(f"No valid observation dates found in {source} data.")
            return False

        df = pd.DataFrame(dates, columns=["date"])  # type: ignore[call-overload]

        if view == "daily":
            df["group"] = df["date"].dt.dayofyear
            bins = 366  # Account for leap years
            labels = [f"Day {i}" for i in range(1, 367)]
            tick_interval = 30
            tick_positions = range(0, bins, tick_interval)
        elif view == "weekly":
            df["group"] = calendar_aligned_week_numbers(df["date"])
            bins = 52
            labels = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            # Evenly space 12 months across 52 weeks (0 to 51)
            tick_positions = np.linspace(0, 51, 12, endpoint=True)
            tick_interval = None
        else:  # monthly
            df["group"] = df["date"].dt.month
            bins = 12
            labels = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            tick_interval = 1
            tick_positions = range(bins)

        counts = df["group"].value_counts().sort_index()
        data = np.zeros(bins)
        for idx, count in counts.items():
            index = int(idx) - 1  # type: ignore[arg-type]
            if 0 <= index < bins:
                data[index] = count
            else:
                logging.warning(f"Skipping out-of-bounds index: {index}")

        # Update graph
        self.ax.clear()
        self.ax.bar(range(bins), data, color=self.color_input.text())
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(labels, rotation=45)
        self.ax.set_xlabel("Time of Year")
        self.ax.set_ylabel("Observation Count")

        title_text = f"Seasonal Observations for {organism or 'All Organisms'} within {radius} km of ({lat:.3f}, {lon:.3f})"
        title_obj = self.ax.set_title(title_text)

        # Robustly shrink title font using actual renderer measurement
        try:
            # Ensure a renderer exists and extents are valid
            if self.canvas:
                self.canvas.draw()
                renderer = self.canvas.get_renderer()

                if renderer:
                    bbox = title_obj.get_window_extent(renderer=renderer)
                    ax_bbox = self.ax.get_window_extent(renderer=renderer)

                    # Check if title width exceeds 95% of axes width
                    if bbox.width > ax_bbox.width * 0.95:
                        scale_ratio = (ax_bbox.width * 0.95) / bbox.width
                        new_size = max(8, title_obj.get_fontsize() * scale_ratio)
                        title_obj.set_fontsize(new_size)
        except Exception as e:
            logging.warning(f"Failed to resize title: {e}")

        # Apply theme to graph
        self.apply_stylesheet()  # Reapply theme to update graph colors

        if self.canvas:
            self.canvas.draw_idle()
        QTimer.singleShot(0, self.clamp_to_screen)

        # Update history
        history_item = f"{organism or 'All'} | {date_from} to {date_to} ({source})"
        self.history_list.addItem(history_item)
        self.history_list.setProperty("data", observations)

        # Update info bar with observation count and descendant taxa
        observation_count = len(observations)
        descendant_count = 0
        if organism:
            cache_key = f"{organism}_descendants"
            if cache_key in self.taxon_cache:
                descendant_count = len(self.taxon_cache[cache_key])
        descendant_text = (
            f", Descendant Taxa: {descendant_count}" if descendant_count > 0 else ""
        )
        self.info_bar.setText(
            f"Source: {source}, Location: ({lat}, {lon}), Radius: {radius} km, "
            f"Organism: {organism or 'All'}, Dates: {date_from} to {date_to}, "
            f"Observations: {observation_count}{descendant_text}"
        )
        self.update_status_bar(observation_count)
        return True

    def local_search(self) -> None:
        """Query the local observations.parquet file and plot results."""
        if (
            self.local_search_worker is not None
            and self.local_search_worker.isRunning()
        ):
            self.status_bar.showMessage("Local search is already running...")
            return

        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            exclude = self.exclude_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()

            # Check for the parquet file in the application data directory.
            parquet_path = self.observations_file
            if not os.path.exists(parquet_path):
                QMessageBox.critical(
                    self,
                    "Error",
                    f"observations.parquet not found in {self.working_dir}.\n"
                    "Please ensure the file is in the application data directory.",
                )
                return

            # Add taxon filter if organism is specified
            taxon_ids = []
            if organism:
                taxon_id = self.get_taxon_id(organism)
                if taxon_id:
                    taxon_ids = self.get_descendant_taxon_ids(organism, taxon_id)
                    if not taxon_ids:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            f"No taxon IDs found for {organism}. Search may yield no results.",
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Could not find taxon ID for {organism}. Search may yield no results.",
                    )

            # Add exclude filter
            exclude_taxon_ids = []
            if exclude:
                exclude_taxon_id = self.get_taxon_id(exclude)
                if exclude_taxon_id:
                    exclude_taxon_ids = self.get_descendant_taxon_ids(
                        exclude, exclude_taxon_id
                    )

            # Start database progress tracking with estimated progress
            self.enhanced_progress.db_tracker.start_operation("local database search")
            self.status_bar.showMessage("Performing local search...")
            self.local_search_button.setEnabled(False)
            self.search_button.setEnabled(False)
            self.show_url_button.setEnabled(False)
            self.cancel_search_button.setText("Cancel Search")
            self.cancel_search_button.setEnabled(True)

            worker = LocalSearchWorker(
                parquet_path,
                date_from,
                date_to,
                lat,
                lon,
                radius,
                taxon_ids,
                exclude_taxon_ids,
            )
            self.local_search_worker = worker
            search_context = {
                "lat": lat,
                "lon": lon,
                "radius": radius,
                "organism": organism,
                "date_from": date_from,
                "date_to": date_to,
                "view": view,
            }
            worker.progress.connect(self._on_local_search_progress)
            worker.search_finished.connect(
                lambda observations, estimated_count, context=search_context: (
                    self._on_local_search_finished(
                        observations, estimated_count, context
                    )
                )
            )
            worker.search_failed.connect(self._on_local_search_failed)
            worker.search_cancelled.connect(self._on_local_search_cancelled)
            worker.finished.connect(self._on_local_search_worker_done)
            worker.start()

        except Exception as e:
            self.enhanced_progress.hide_progress()
            # The buttons were disabled above; if the worker never started, its
            # finished signal won't fire to re-enable them via
            # _on_local_search_worker_done, so restore them here.
            if (
                self.local_search_worker is None
                or not self.local_search_worker.isRunning()
            ):
                self.local_search_button.setEnabled(True)
                self.search_button.setEnabled(True)
                self.show_url_button.setEnabled(True)
                self.cancel_search_button.setEnabled(False)
                self.cancel_search_button.setText("Cancel Search")
                self.local_search_worker = None
            QMessageBox.critical(self, "Error", f"Local search failed: {e!s}")
            self.show_placeholder(f"Local search failed: {e!s}")
            logging.error(f"Local search failed: {e!s}")

    def _on_local_search_progress(self, message: str) -> None:
        if message.startswith("Executing DuckDB query"):
            self.status_bar.showMessage("Performing local search...")
        self.enhanced_progress.db_tracker.update_progress(message=message)

    def _on_local_search_finished(
        self,
        observations: list[dict[str, Any]],
        estimated_count: int,
        context: dict[str, Any],
    ) -> None:
        if (
            self.local_search_worker is not None
            and self.local_search_worker.is_cancelled()
        ):
            self._on_local_search_cancelled()
            return
        self.enhanced_progress.db_tracker.estimated_total = estimated_count
        self.enhanced_progress.db_tracker.finish_operation(
            len(observations), "Local search completed"
        )
        self.status_bar.showMessage("Local search completed.")

        if observations:
            self.plot_observations(
                observations,
                context["lat"],
                context["lon"],
                context["radius"],
                context["organism"],
                context["date_from"],
                context["date_to"],
                context["view"],
                source="Local",
            )
        else:
            self.show_placeholder("No local observations found.")
            self.update_status_bar(0)

    def _on_local_search_failed(self, error: str) -> None:
        self.enhanced_progress.hide_progress()
        QMessageBox.critical(self, "Error", f"Local search failed: {error}")
        self.show_placeholder(f"Local search failed: {error}")
        logging.error("Local search failed: %s", error)

    def _on_local_search_cancelled(self) -> None:
        self.enhanced_progress.finish_progress("Local search cancelled")
        self.status_bar.showMessage("Local search cancelled.")

    def _on_local_search_worker_done(self) -> None:
        self.local_search_button.setEnabled(self.local_database_available)
        self.search_button.setEnabled(True)
        self.show_url_button.setEnabled(True)
        self.cancel_search_button.setEnabled(False)
        self.cancel_search_button.setText("Cancel Search")
        self.local_search_worker = None

    def cancel_search(self) -> None:
        """Cancel whichever local or live search is currently running."""
        # Treat a worker as cancellable until its queued completion signal has
        # been handled. This closes the small gap between the thread finishing
        # and a large result being plotted on the main thread.
        cancelling_api = self.api_search_worker is not None
        cancelling_local = self.local_search_worker is not None
        if not cancelling_api and not cancelling_local:
            self.cancel_search_button.setEnabled(False)
            self.status_bar.showMessage("No search is currently running.")
            return

        self.cancel_search_button.setText("Cancelling...")
        self.cancel_search_button.setEnabled(False)
        if cancelling_local:
            self.local_search_worker.cancel()
            message = "Cancelling local search..."
        else:
            self.api_search_worker.cancel()
            message = (
                "Cancelling live search... The current network request may take "
                "a few seconds to stop."
            )
        self.status_bar.showMessage(message)
        self.enhanced_progress.update_progress(message=message)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Stop background workers before the window is destroyed."""
        for worker in (
            self.local_search_worker,
            self.api_search_worker,
            self.database_update_worker,
        ):
            if worker is not None and worker.isRunning():
                worker.cancel()
                # Workers perform blocking calls without a thread event
                # loop, so quit() cannot interrupt them. Wait briefly and use
                # terminate only as a last resort during application shutdown.
                if not worker.wait(5000):
                    worker.terminate()
                    worker.wait()
        super().closeEvent(a0)

    def search_observations(self) -> None:
        """Start a non-blocking search using the iNaturalist API."""
        if not self.canvas:
            QMessageBox.critical(
                self, "Error", "Cannot search: Plot initialization failed."
            )
            return
        if self.api_search_worker is not None and self.api_search_worker.isRunning():
            self.status_bar.showMessage(
                "A live iNaturalist search is already running..."
            )
            return

        try:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Invalid latitude or longitude. Please enter valid numbers.",
                )
                return

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip()
            exclude = self.exclude_input.text().strip()
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()

            # Build API parameters
            params = {
                "lat": lat,
                "lng": lon,
                "radius": radius,
                "d1": date_from,
                "d2": date_to,
                "per_page": 500,
            }

            search_context = {
                "lat": lat,
                "lon": lon,
                "radius": radius,
                "organism": organism,
                "date_from": date_from,
                "date_to": date_to,
                "view": view,
            }
            worker = ApiSearchWorker(
                params,
                organism,
                exclude,
                self.taxon_cache,
                self.http_cache_file,
                self.http_cache_max_bytes,
            )
            self.api_search_worker = worker
            worker.progress.connect(self._on_api_search_progress)
            worker.api_call_completed.connect(self._on_api_call_completed)
            worker.taxon_resolved.connect(self._on_api_taxon_resolved)
            worker.search_finished.connect(
                lambda observations, warnings, context=search_context: (
                    self._on_api_search_finished(observations, warnings, context)
                )
            )
            worker.search_failed.connect(self._on_api_search_failed)
            worker.search_cancelled.connect(self._on_api_search_cancelled)
            worker.finished.connect(self._on_api_search_worker_done)

            self.enhanced_progress.start_progress(
                0, "Starting live iNaturalist search..."
            )
            self.status_bar.showMessage("Starting live iNaturalist search...")
            self.local_search_button.setEnabled(False)
            self.search_button.setEnabled(False)
            self.search_button.setText("Searching live iNat data...")
            self.show_url_button.setEnabled(False)
            self.cancel_search_button.setText("Cancel Search")
            self.cancel_search_button.setEnabled(True)
            worker.start()

        except Exception as e:
            self.enhanced_progress.hide_progress()
            self.local_search_button.setEnabled(self.local_database_available)
            self.search_button.setEnabled(True)
            self.search_button.setText(LIVE_GRAPH_ACTION_TEXT)
            self.show_url_button.setEnabled(True)
            self.cancel_search_button.setEnabled(False)
            self.cancel_search_button.setText("Cancel Search")
            self.api_search_worker = None
            QMessageBox.critical(self, "Error", f"API search failed: {e!s}")
            self.show_placeholder(f"API search failed: {e!s}")
            logging.error(f"API search failed: {e!s}")

    def _on_api_search_progress(self, message: str, percentage: int | None) -> None:
        self.status_bar.showMessage(message)
        if percentage is None:
            self.enhanced_progress.update_progress(message=message)
        else:
            self.enhanced_progress.update_percentage(percentage, message)

    def _on_api_call_completed(self) -> None:
        self.api_call_count += 1
        self.update_api_call_count()

    def _on_api_taxon_resolved(self, query: str, taxon_id: int) -> None:
        self.taxon_cache[query] = taxon_id
        self.save_taxon_cache()

    def _on_api_search_finished(
        self,
        observations: list[dict[str, Any]],
        warnings: list[str],
        context: dict[str, Any],
    ) -> None:
        if self.api_search_worker is not None and self.api_search_worker.is_cancelled():
            self._on_api_search_cancelled()
            return
        self.enhanced_progress.finish_progress(
            f"Live search completed: {len(observations)} observations"
        )
        self.status_bar.showMessage("Live iNaturalist search completed.")
        if observations:
            self.plot_observations(
                observations,
                context["lat"],
                context["lon"],
                context["radius"],
                context["organism"],
                context["date_from"],
                context["date_to"],
                context["view"],
                source="API",
            )
        else:
            self.show_placeholder("No API observations found.")
            self.update_status_bar(0)
        if warnings:
            QMessageBox.warning(self, "Search Warning", "\n\n".join(warnings))

    def _on_api_search_failed(self, error: str) -> None:
        self.enhanced_progress.finish_progress("Live iNaturalist search failed")
        self.status_bar.showMessage("Live iNaturalist search failed.")
        QMessageBox.critical(self, "Error", f"API search failed: {error}")
        self.show_placeholder(f"API search failed: {error}")
        logging.error("API search failed: %s", error)

    def _on_api_search_cancelled(self) -> None:
        self.enhanced_progress.finish_progress("Live iNaturalist search cancelled")
        self.status_bar.showMessage("Live iNaturalist search cancelled.")

    def _on_api_search_worker_done(self) -> None:
        self.local_search_button.setEnabled(self.local_database_available)
        self.search_button.setEnabled(True)
        self.search_button.setText(LIVE_GRAPH_ACTION_TEXT)
        self.show_url_button.setEnabled(True)
        self.cancel_search_button.setEnabled(False)
        self.cancel_search_button.setText("Cancel Search")
        self.api_search_worker = None

    def export_graph(self) -> None:
        """Export the current graph as an image with metadata."""
        if not self.canvas:
            QMessageBox.critical(
                self, "Error", "No graph to export: Plot initialization failed."
            )
            return

        try:
            # Get current parameters
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Invalid latitude or longitude. Using default values for export.",
                )
                lat = float(self.default_lat)
                lon = float(self.default_lon)

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip() or "All Organisms"
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()
            source = (
                "Local" if self.info_bar.text().startswith("Source: Local") else "API"
            )
            observation_count = len(self.history_list.property("data") or [])
            descendant_count = 0
            if organism != "All Organisms":
                cache_key = f"{organism}_descendants"
                if cache_key in self.taxon_cache:
                    descendant_count = len(self.taxon_cache[cache_key])

            # Open file dialog
            file_dialog = QFileDialog(
                self, "Export Graph", "", "JPG (*.jpg);;PNG (*.png)"
            )
            file_dialog.setDefaultSuffix("jpg")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if not file_dialog.exec():
                return

            filename = file_dialog.selectedFiles()[0]
            # Ensure .jpg extension
            file_path = Path(filename)
            if file_path.suffix.lower() not in [".jpg", ".jpeg"]:
                filename = str(file_path.with_suffix(".jpg"))

            # Create a new figure for export to avoid modifying the UI graph
            export_fig, export_ax = plt.subplots(figsize=(10, 6))

            # Recompute plot data (replicate plot_observations logic)
            observations = self.history_list.property("data") or []
            if not observations:
                plt.close(export_fig)
                QMessageBox.warning(self, "Error", "No data to export.")
                return

            dates = []
            for obs in observations:
                date_str = obs.get("eventDate" if source == "Local" else "observed_on")
                if date_str:
                    try:
                        date = pd.to_datetime(date_str, utc=True).tz_localize(None)
                        dates.append(date)
                    except Exception:
                        continue

            if not dates:
                plt.close(export_fig)
                QMessageBox.warning(self, "Error", "No valid dates to export.")
                return

            df = pd.DataFrame(dates, columns=["date"])  # type: ignore[call-overload]

            if view == "daily":
                df["group"] = df["date"].dt.dayofyear
                bins = 366
                labels = [f"Day {i}" for i in range(1, 367)]
                tick_positions = range(0, bins, 30)
            elif view == "weekly":
                df["group"] = calendar_aligned_week_numbers(df["date"])
                bins = 52
                labels = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                tick_positions = np.linspace(0, 51, 12, endpoint=True)
            else:  # monthly
                df["group"] = df["date"].dt.month
                bins = 12
                labels = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                tick_positions = range(bins)

            counts = df["group"].value_counts().sort_index()
            data = np.zeros(bins)
            for idx, count in counts.items():
                index = int(idx) - 1  # type: ignore[arg-type]
                if 0 <= index < bins:
                    data[index] = count

            # Plot on new figure
            export_ax.bar(range(bins), data, color=self.color_input.text())
            export_ax.set_xticks(tick_positions)
            export_ax.set_xticklabels(labels, rotation=45)
            export_ax.set_xlabel("Time of Year")
            export_ax.set_ylabel("Observation Count")
            export_ax.set_title(f"Seasonal Observations for {organism} ({source})")

            # Apply theme to export graph
            current_theme = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
            if current_theme == "dark":
                export_fig.set_facecolor(DARK_WINDOW_BACKGROUND)
                export_ax.set_facecolor(DARK_GRAPH_BACKGROUND)
                export_ax.tick_params(colors="white")
                export_ax.xaxis.label.set_color("white")
                export_ax.yaxis.label.set_color("white")
                export_ax.title.set_color("white")
                for spine in export_ax.spines.values():
                    spine.set_color("white")
            else:
                export_fig.set_facecolor("white")
                export_ax.set_facecolor("white")
                export_ax.tick_params(colors="black")
                export_ax.xaxis.label.set_color("black")
                export_ax.yaxis.label.set_color("black")
                export_ax.title.set_color("black")
                for spine in export_ax.spines.values():
                    spine.set_color("black")

            # Add metadata text
            metadata = (
                f"Source: {source}\n"
                f"Location: ({lat}, {lon})\n"
                f"Radius: {radius} km\n"
                f"Organism: {organism}\n"
                f"Dates: {date_from} to {date_to}\n"
                f"Observations: {observation_count}\n"
                f"Descendant Taxa: {descendant_count}"
                if descendant_count > 0
                else ""
            )
            export_fig.text(
                0.02,
                0.02,
                metadata,
                fontsize=8,
                transform=export_fig.transFigure,
                verticalalignment="bottom",
            )

            # Render and save
            export_fig.tight_layout(rect=(0, 0.1, 1, 0.95))  # Adjust for metadata
            export_fig.canvas.draw()
            export_fig.savefig(filename, format="jpg", dpi=300)
            plt.close(export_fig)

            QMessageBox.information(self, "Success", f"Graph exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export graph: {e!s}")
            logging.error(f"Export graph failed: {e!s}")

    def export_data(self) -> None:
        """Export observation data as CSV."""
        observations = self.history_list.property("data") or []
        if not observations:
            QMessageBox.warning(self, "Error", "No data to export.")
            return

        try:
            file_dialog = QFileDialog(self, "Export Data", "", "CSV (*.csv)")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if not file_dialog.exec():
                return

            filename = file_dialog.selectedFiles()[0]
            source = (
                "Local" if self.info_bar.text().startswith("Source: Local") else "API"
            )
            data = []
            for obs in observations:
                data.append(
                    {
                        "id": obs.get("id"),
                        "latitude": obs.get(
                            "decimalLatitude" if source == "Local" else "geojson", {}
                        ).get("coordinates", [None, None])[1],
                        "longitude": obs.get(
                            "decimalLongitude" if source == "Local" else "geojson", {}
                        ).get("coordinates", [None, None])[0],
                        "date": obs.get(
                            "eventDate" if source == "Local" else "observed_on"
                        ),
                        "taxonID": obs.get(
                            "taxonID" if source == "Local" else "taxon_id"
                        ),
                    }
                )

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {e!s}")
            logging.error(f"Export data failed: {e!s}")

    def load_history_item(self) -> None:
        """Load and replot a history item."""
        observations = self.history_list.property("data") or []
        if not observations:
            return

        try:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                lat = float(self.default_lat)
                lon = float(self.default_lon)

            radius = float(self.radius_input.text())
            organism = self.organism_input.text().strip() or "All Organisms"
            date_from = self.date_from.text()
            date_to = self.date_to.text()
            view = self.view_combo.currentText().lower()
            _item = self.history_list.currentItem()
            source = (
                "Local"
                if _item is not None and _item.text().endswith("(Local)")
                else "API"
            )

            self.plot_observations(
                observations,
                lat,
                lon,
                radius,
                organism,
                date_from,
                date_to,
                view,
                source,
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load history item: {e!s}")
            logging.error(f"Load history item failed: {e!s}")

    def show_placeholder(self, message: str | None = None) -> None:
        """Show a placeholder message on the graph."""
        if self.canvas:
            # Determine text color based on background luminance for visibility
            current_theme = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
            custom_graph_bg = self.settings.value("graph_bg_color")
            graph_bg_color = custom_graph_bg or (
                DARK_GRAPH_BACKGROUND if current_theme == "dark" else LIGHT_BACKGROUND
            )
            text_color = self.get_contrasting_text_color(graph_bg_color)

            self.ax.clear()
            detail_message = None
            if message:
                main_message = message
            else:
                # Load current settings from UI widgets to display them
                lat = self.lat_input.text()
                lon = self.lon_input.text()

                radius = self.radius_input.text()
                date_from = self.date_from.text()
                date_to = self.date_to.text()
                graph_color = self.color_input.text() or "#1f77b4"
                graph_bg_color = self.bg_color_input.text() or "Theme Default"
                window_bg_color = self.settings.value(
                    "window_bg_color", "Theme Default"
                )

                settings_text = (
                    f"Current Settings:\n\n"
                    f"  - Latitude: {lat}\n"
                    f"  - Longitude: {lon}\n"
                    f"  - Radius: {radius} km\n"
                    f"  - Date Range: {date_from} to {date_to}\n"
                    f"  - Graph Color: {graph_color}\n"
                    f"  - Graph Background Color: {graph_bg_color}\n"
                    f"  - Window Background Color: {window_bg_color}\n\n"
                )

                if self.local_database_available:
                    search_instructions = (
                        f"2. Click '{LOCAL_GRAPH_ACTION_TEXT}' for a fast local "
                        f"database lookup or '{LIVE_GRAPH_ACTION_TEXT}' to retrieve "
                        "current observations from iNaturalist.\n\n"
                    )
                else:
                    search_instructions = (
                        f"2. Click '{LIVE_GRAPH_ACTION_TEXT}' to retrieve current "
                        "observations from iNaturalist. The optional local database "
                        f"is not installed, so '{LOCAL_GRAPH_ACTION_TEXT}' is "
                        "unavailable.\n\n"
                    )

                main_message = (
                    "Welcome to iNaturalist Seasonal Visualizer!\n\n"
                    "To get started:\n"
                    "1. Enter an organism name (e.g., Russula brevipes, "
                    "Agaricales, etc).\n"
                    + search_instructions
                    + "The graph will display seasonal observation patterns.\n\n"
                )
                detail_message = settings_text

                if self.total_observations:
                    try:
                        obs_count = f"{self.total_observations:,}"
                        taxa_count = f"{self.unique_taxa:,}"
                    except (ValueError, TypeError):  # Handle "Error" or "N/A" strings
                        obs_count = self.total_observations
                        taxa_count = self.unique_taxa

                    database_stats_text = (
                        f"Local Database Stats:\n"
                        f"  - Total Observations: {obs_count}\n"
                        f"  - Unique Taxa: {taxa_count}\n"
                        f"  - Most Recent Observation: "
                        f"{getattr(self, 'most_recent_date', 'N/A')}\n\n"
                    )
                    detail_message += database_stats_text

            main_font_size = self.graph_font_size * self.scale_factor
            if detail_message is None:
                self.ax.text(
                    0.5,
                    0.5,
                    main_message,
                    color=text_color,
                    fontsize=main_font_size,
                    ha="center",
                    va="center",
                    wrap=True,
                )
            else:
                self.ax.text(
                    0.5,
                    0.51,
                    main_message.rstrip(),
                    color=text_color,
                    fontsize=main_font_size,
                    ha="center",
                    va="bottom",
                    wrap=True,
                )
                self.ax.text(
                    0.5,
                    0.49,
                    detail_message.rstrip(),
                    color=text_color,
                    fontsize=max(
                        PLACEHOLDER_DETAILS_MIN_FONT_SIZE,
                        main_font_size * PLACEHOLDER_DETAILS_FONT_SCALE,
                    ),
                    ha="center",
                    va="top",
                    wrap=True,
                )
            self.ax.set_axis_off()
            self.canvas.draw()
            self.info_bar.setText("")
            self.update_status_bar(0)

    def update_status_bar(self, observation_count: int = 0) -> None:
        """Update the status bar with current info."""
        search_mode = (
            "Local DB available" if self.local_database_available else "API-only mode"
        )
        self.status_bar.showMessage(
            f"Ready | {search_mode} | Observations: {observation_count} | "
            f"API Calls: {self.api_call_count}"
        )

    def get_contrasting_text_color(
        self,
        bg_color_hex: str,
        light_fallback: str = "white",
        dark_fallback: str = "black",
    ) -> str:
        """Determine whether black or white text has better contrast against a hex background color."""
        try:
            q_color = QColor(bg_color_hex)
            # Using standard luminance formula
            luminance = (
                0.299 * q_color.red() + 0.587 * q_color.green() + 0.114 * q_color.blue()
            )
            return dark_fallback if luminance > 128 else light_fallback
        except Exception:
            # Fallback if the color string is invalid
            current_theme = normalize_theme(self.settings.value("theme", DEFAULT_THEME))
            return dark_fallback if current_theme == "light" else light_fallback


def main() -> None:
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="iNaturalist Seasonal Visualizer")
    parser.add_argument("--lat", type=float, help="Latitude (default from settings)")
    parser.add_argument("--lon", type=float, help="Longitude (default from settings)")
    parser.add_argument(
        "--radius", type=float, help="Radius in km (default from settings)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Manual UI scaling factor (e.g., 2.0 for 200%% scaling)",
    )
    parser.add_argument(
        "--http-cache-max-mb",
        type=int,
        help=(
            "Maximum API response cache size in MB "
            f"(default: {DEFAULT_HTTP_CACHE_MAX_MB}; environment: {HTTP_CACHE_MAX_MB_ENV})"
        ),
    )
    args = parser.parse_args()

    working_dir = ensure_application_data_dir()

    # Configure logging. Always write to the log file; also mirror to the
    # console (basicConfig with `filename` installs a file-only handler, so
    # without this nothing shows up on stdout/stderr, even with --debug).
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            create_log_file_handler(working_dir / "inat_visualizer.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Silence noisy third-party DEBUG chatter (matplotlib font scoring, PIL,
    # HTTP internals) so --debug stays focused on this app's own messages.
    for noisy_logger in (
        "matplotlib",
        "PIL",
        "urllib3",
        "requests",
        "requests_cache",
        "requests_ratelimiter",
        "pyrate_limiter",
        "pyinaturalist",
        "asyncio",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.info(
        "Starting iNaturalist Seasonal Visualizer v%s (debug=%s, data_dir=%s)",
        __version__,
        args.debug,
        working_dir,
    )

    if args.debug:
        print(f"DEBUG: Logging level set to {log_level}")

    # Check environment
    if args.debug:
        print("DEBUG: Checking environment...")
    env_error = check_environment()
    if env_error:
        print(env_error)
        sys.exit(1)
    if args.debug:
        print("DEBUG: Environment check passed.")

    # --- High DPI stability setup (Wayland + Qt6) ---
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
    )

    # Create QApplication first
    if args.debug:
        print("DEBUG: Creating QApplication...")
    app = QApplication(sys.argv)
    app.setApplicationName("iNaturalist Seasonal Visualizer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AlanRockefeller")
    if args.debug:
        print("DEBUG: QApplication created.")

    if args.smoke_test:
        logging.info(
            "Packaged application smoke test passed for version %s", __version__
        )
        return

    # Show splash screen
    splash_image_path = resource_path("splash_screen.jpg")
    if os.path.exists(splash_image_path):
        if args.debug:
            print(f"DEBUG: Splash screen found at {splash_image_path}. Initializing...")
        splash = CustomSplashScreen(splash_image_path)
        splash.update_status("Starting iNaturalist Seasonal Visualizer...")
        QApplication.processEvents()
    else:
        if args.debug:
            print(f"DEBUG: Splash screen not found at {splash_image_path}. Skipping.")
        splash = None
        logging.warning(f"Splash screen image not found at {splash_image_path}")

    # Initialize main window with splash screen updates
    if splash:
        splash.update_status("Loading application components...")
        QApplication.processEvents()

    if args.debug:
        print("DEBUG: Initializing main window (INatSeasonalVisualizer)...")
    window = INatSeasonalVisualizer(
        lat=args.lat,
        lon=args.lon,
        radius=args.radius,
        scale_factor=args.scale_factor,
        splash_screen=splash,
        working_dir=working_dir,
        http_cache_max_mb=args.http_cache_max_mb,
    )
    if args.debug:
        print("DEBUG: Main window initialized.")

    if splash:
        splash.update_status("Application ready!")
        QApplication.processEvents()
        if args.debug:
            print("DEBUG: Closing splash screen...")
        splash.close()

    if args.debug:
        print("DEBUG: Showing main window maximized...")
    window.showMaximized()
    QTimer.singleShot(0, window.start_database_update_check)

    if args.debug:
        print("DEBUG: Starting event loop (app.exec)...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

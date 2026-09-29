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
import hashlib
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
from urllib.parse import urlencode, urlparse

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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from PIL import Image, UnidentifiedImageError
from PyQt6.QtCore import (
    QDate,
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
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplashScreen,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
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
ABUNDANCE_ACTION_TEXT = "Map abundance (local data)"
SEASONAL_MODE_TEXT = "Seasonal graph"
ABUNDANCE_MODE_TEXT = "Abundance map"
ABUNDANCE_SEASON_YEAR = 2024
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
                privacy_safe_path(path),
            )
        if size_after > max_size_bytes:
            logging.warning(
                "HTTP cache remains above its %.1f MB budget (size=%.1f MB, path=%s).",
                max_size_bytes / (1024 * 1024),
                size_after / (1024 * 1024),
                privacy_safe_path(path),
            )
    except Exception as error:
        logging.warning(
            "Could not maintain HTTP cache %s: %s",
            privacy_safe_path(path),
            privacy_safe_message(error),
        )


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
            "Could not inspect legacy HTTP cache %s: %s",
            privacy_safe_path(legacy_path),
            privacy_safe_message(error),
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
        logging.warning(
            "Could not open legacy HTTP cache %s: %s",
            privacy_safe_path(legacy_path),
            privacy_safe_message(error),
        )
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
        logging.warning(
            "Could not save database statistics cache %s: %s",
            privacy_safe_path(path),
            privacy_safe_message(error),
        )
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
WEB_MERCATOR_MAX_LATITUDE = 85.05112878
EARTH_RADIUS_KM = 6371.0
MAP_RADIUS_MIN_KM = 1.0
MAP_RADIUS_MAX_KM = 1000.0
_LOG_PRIVACY_KEY = os.urandom(16)


def private_text_identifier(value: str) -> str:
    """Return a session-local identifier without exposing user-provided text."""
    digest = hashlib.blake2b(
        str(value).encode("utf-8", errors="replace"),
        key=_LOG_PRIVACY_KEY,
        digest_size=6,
    ).hexdigest()
    return f"text-{digest}"


def privacy_safe_path(path: str | Path) -> str:
    """Format a path for logs without exposing the user's home directory name."""
    if str(path) == ":memory:":
        return ":memory:"
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        relative = candidate.resolve(strict=False).relative_to(
            Path.home().resolve(strict=False)
        )
    except (OSError, ValueError):
        return str(candidate)
    return str(Path("~") / relative)


def privacy_safe_message(value: Any) -> str:
    """Remove the current home-directory prefix from exception text."""
    return str(value).replace(str(Path.home()), "~")


def place_search_error_diagnostic(error: BaseException) -> str:
    """Describe a place-search failure without logging query or exception text."""
    chain: list[str] = []
    http_status = None
    response_host = None
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        if isinstance(current, HTTPError):
            response = current.response
            if response is not None:
                http_status = response.status_code
                response_host = urlparse(response.url or "").hostname
        current = current.__cause__ or current.__context__

    details = [f"error_chain={' -> '.join(chain)}"]
    if http_status is not None:
        details.append(f"http_status={http_status}")
    if response_host:
        details.append(f"response_host={response_host}")
    return ", ".join(details)


def normalize_longitude(lon: float) -> float:
    """Return a longitude in the conventional [-180, 180) range."""
    return (float(lon) + 180.0) % 360.0 - 180.0


def clamp_latitude_limits(south: float, north: float) -> tuple[float, float]:
    """Shift latitude limits inside the Web Mercator drawable range."""
    if south < -WEB_MERCATOR_MAX_LATITUDE:
        north += -WEB_MERCATOR_MAX_LATITUDE - south
        south = -WEB_MERCATOR_MAX_LATITUDE
    if north > WEB_MERCATOR_MAX_LATITUDE:
        south -= north - WEB_MERCATOR_MAX_LATITUDE
        north = WEB_MERCATOR_MAX_LATITUDE
    return south, north


def longitude_near_reference(lon: float, reference: float) -> float:
    """Unwrap ``lon`` to the world copy nearest ``reference``."""
    return float(reference) + ((float(lon) - float(reference) + 180.0) % 360.0) - 180.0


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def geodesic_destination(
    lat: float, lon: float, distance_km: float, bearing_degrees: float
) -> tuple[float, float]:
    """Return the point reached along a great-circle bearing."""
    angular_distance = float(distance_km) / EARTH_RADIUS_KM
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    bearing = math.radians(float(bearing_degrees))

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    normalized_lon = normalize_longitude(math.degrees(lon2))
    return math.degrees(lat2), longitude_near_reference(normalized_lon, lon)


def geodesic_circle_coordinates(
    lat: float, lon: float, radius_km: float, point_count: int = 181
) -> tuple[list[float], list[float]]:
    """Return an antimeridian-safe great-circle boundary for a map overlay."""
    point_count = max(4, int(point_count))
    nearer_pole_lat = 90.0 if lat >= 0.0 else -90.0
    encloses_pole = radius_km >= geodesic_distance_km(
        lat, lon, nearer_pole_lat, lon
    )
    if encloses_pole:
        boundary = []
        for bearing in np.linspace(0.0, 360.0, point_count, endpoint=False):
            point_lat, point_lon = geodesic_destination(
                lat, lon, radius_km, float(bearing)
            )
            boundary.append(
                (
                    longitude_near_reference(point_lon, lon),
                    max(
                        -WEB_MERCATOR_MAX_LATITUDE,
                        min(WEB_MERCATOR_MAX_LATITUDE, point_lat),
                    ),
                )
            )
        boundary.sort(key=lambda point: point[0])
        west = float(lon) - 180.0
        east = float(lon) + 180.0
        boundary_lons = [west, *(point[0] for point in boundary), east]
        boundary_lats = [
            boundary[0][1],
            *(point[1] for point in boundary),
            boundary[-1][1],
        ]
        cap_lat = math.copysign(WEB_MERCATOR_MAX_LATITUDE, nearer_pole_lat)
        cap_lons = np.linspace(east, west, point_count).tolist()
        return (
            [*boundary_lons, *cap_lons, west],
            [*boundary_lats, *([cap_lat] * point_count), boundary_lats[0]],
        )

    lats: list[float] = []
    lons: list[float] = []
    for bearing in np.linspace(0.0, 360.0, point_count):
        point_lat, point_lon = geodesic_destination(lat, lon, radius_km, float(bearing))
        lats.append(point_lat)
        lons.append(point_lon)
    return lons, lats


def radius_view_limits(
    lat: float, lon: float, radius_km: float, padding: float = 0.25
) -> tuple[float, float, float, float]:
    """Return padded longitude/latitude limits containing a radius circle."""
    circle_lons, circle_lats = geodesic_circle_coordinates(lat, lon, radius_km)
    west, east = min(circle_lons), max(circle_lons)
    south = max(-WEB_MERCATOR_MAX_LATITUDE, min(circle_lats))
    north = min(WEB_MERCATOR_MAX_LATITUDE, max(circle_lats))

    lat_margin = max((north - south) * padding, 0.005)
    nearer_pole_lat = 90.0 if lat >= 0.0 else -90.0
    if radius_km >= geodesic_distance_km(lat, lon, nearer_pole_lat, lon):
        return (
            float(lon) - 180.0,
            float(lon) + 180.0,
            max(-WEB_MERCATOR_MAX_LATITUDE, south - lat_margin),
            min(WEB_MERCATOR_MAX_LATITUDE, north + lat_margin),
        )

    lon_margin = max((east - west) * padding, 0.005)
    return (
        west - lon_margin,
        east + lon_margin,
        max(-WEB_MERCATOR_MAX_LATITUDE, south - lat_margin),
        min(WEB_MERCATOR_MAX_LATITUDE, north + lat_margin),
    )


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
    """Serialize a compact, privacy-safe place-search response summary."""
    if not isinstance(response, dict):
        return json.dumps(
            {"response_type": type(response).__name__}, separators=(",", ":")
        )

    raw_results = response.get("results")
    results = raw_results if isinstance(raw_results, list) else []
    result_types: dict[str, int] = {}
    records_with_bounds = 0
    for result in results:
        if not isinstance(result, dict):
            result_type = type(result).__name__
        else:
            result_type = str(result.get("type") or "unknown")
            record = result.get("record")
            if isinstance(record, dict) and record.get("bounding_box_geojson"):
                records_with_bounds += 1
        result_types[result_type] = result_types.get(result_type, 0) + 1

    summary = {
        "total_results": response.get("total_results"),
        "returned_results": len(results),
        "result_types": result_types,
        "usable_places": len(normalize_place_search_results(response)),
        "records_with_bounds": records_with_bounds,
    }
    return json.dumps(summary, separators=(",", ":"), sort_keys=True)


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
    earth_radius_km = EARTH_RADIUS_KM
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


# OpenStreetMap sometimes answers a blocked client with this "403 Access
# blocked" placeholder image but an HTTP 200 status. It must never be cached
# or drawn as map content.
OSM_BLOCKED_TILE_MD5 = frozenset({"c069a15b2cc2d6b6f527ad09eb93c61a"})


def is_blocked_tile_image(image_bytes: bytes) -> bool:
    return hashlib.md5(image_bytes).hexdigest() in OSM_BLOCKED_TILE_MD5


def mercator_rows_to_equirectangular(
    composite: "Image.Image", y_min_tile: int, zoom: int, south: float, north: float
) -> "Image.Image":
    """Resample a Web-Mercator tile mosaic so rows are evenly spaced in latitude.

    The maps plot longitude/latitude axes, so drawing Mercator tiles straight
    into a latitude extent misplaces features by hundreds of kilometres once a
    view spans a continent. Remapping rows keeps tiles aligned with overlays
    such as the search circle and the abundance heatmap.
    """
    pixels = np.asarray(composite)
    height = pixels.shape[0]
    if height == 0:
        return composite
    row_lats = north - (np.arange(height) + 0.5) * (north - south) / height
    row_lats = np.clip(row_lats, -WEB_MERCATOR_MAX_LATITUDE, WEB_MERCATOR_MAX_LATITUDE)
    tile_y = (
        (1.0 - np.arcsinh(np.tan(np.radians(row_lats))) / np.pi) / 2.0 * 2.0**zoom
    )
    source_rows = np.clip(
        ((tile_y - y_min_tile) * 256.0).astype(int), 0, height - 1
    )
    return Image.fromarray(pixels[source_rows])


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
                image_bytes = path.read_bytes()
                if is_blocked_tile_image(image_bytes):
                    path.unlink(missing_ok=True)
                    return None
                # Update mtime for LRU
                path.touch()
                with Image.open(BytesIO(image_bytes)) as img:
                    return img.convert("RGB").copy()
            except Exception as e:
                logging.warning(
                    "Corrupt tile in disk cache %s: %s",
                    privacy_safe_path(path),
                    privacy_safe_message(e),
                )
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
            logging.error(
                "Failed to write tile to disk %s: %s",
                privacy_safe_path(path),
                privacy_safe_message(e),
            )
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
        started_at = time.monotonic()
        query_id = private_text_identifier(query)
        request_params = {
            "q": query,
            "sources": "places",
            "per_page": PLACE_SEARCH_RESULT_LIMIT,
        }
        debug_logging = logging.getLogger().isEnabledFor(logging.DEBUG)
        if debug_logging:
            logging.debug(
                "Place search request: query_id=%s, chars=%d, qualified=%s, limit=%d",
                query_id,
                len(query),
                "," in query,
                PLACE_SEARCH_RESULT_LIMIT,
            )
        response = pyinaturalist.search(session=request_session, **request_params)
        if debug_logging:
            logging.debug(
                "Place search completed: query_id=%s, duration_ms=%d, summary=%s",
                query_id,
                round((time.monotonic() - started_at) * 1000),
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
                    "Place search fallback: query_id=%s returned no usable results; "
                    "resolving qualifier_id=%s and retrying name_id=%s",
                    private_text_identifier(self.query),
                    private_text_identifier(qualifier),
                    private_text_identifier(name),
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
                        "Place search qualifier_id=%s resolved; kept %d of %d "
                        "fallback matches",
                        private_text_identifier(qualifier),
                        len(scoped_places),
                        len(fallback_places),
                    )
                    places = scoped_places
                else:
                    logging.debug(
                        "Place search qualifier_id=%s could not narrow %d fallback "
                        "matches; returning the unqualified matches",
                        private_text_identifier(qualifier),
                        len(fallback_places),
                    )
                    places = fallback_places
            if not self.isInterruptionRequested():
                self.results_ready.emit(places)
        except Exception as error:
            if not self.isInterruptionRequested():
                logging.error(
                    "Place search failed: query_id=%s, %s",
                    private_text_identifier(self.query),
                    place_search_error_diagnostic(error),
                )
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
            try:
                self._process_job(session, job_id, base_zoom, x0, x1, y0, y1)
            except Exception:
                logging.exception("Unexpected map tile job failure; continuing")

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
        started_at = time.monotonic()
        ram_hits = 0
        disk_hits = 0
        network_fetches = 0
        skipped_tiles = 0
        failed_tiles = 0

        def log_view_summary(
            status: str, selected_zoom: int, requested_tiles: int, loaded_tiles: int
        ) -> None:
            logging.debug(
                "Map tile view: job=%d, status=%s, zoom=%d->%d, tiles=%d, "
                "loaded=%d, ram=%d, disk=%d, network=%d, skipped=%d, failed=%d, "
                "duration_ms=%d",
                job_id,
                status,
                base_zoom,
                selected_zoom,
                requested_tiles,
                loaded_tiles,
                ram_hits,
                disk_hits,
                network_fetches,
                skipped_tiles,
                failed_tiles,
                round((time.monotonic() - started_at) * 1000),
            )

        zoom = base_zoom
        total_tiles = 0
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
                            log_view_summary("superseded", zoom, total_tiles, 0)
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
                        log_view_summary("superseded", zoom, total_tiles, len(tiles))
                        return

                wrapped_x = x % int(2.0**zoom)
                tile_key = (zoom, wrapped_x, y)
                img = None

                if tile_key in self.ram_cache:
                    img = self.ram_cache[tile_key]
                    self.ram_cache.move_to_end(tile_key)
                    ram_hits += 1

                if img is None:
                    img = self.disk_cache.get_tile(zoom, wrapped_x, y)
                    if img is not None:
                        self.ram_cache[tile_key] = img
                        if len(self.ram_cache) > MAX_CACHE_SIZE:
                            self.ram_cache.popitem(last=False)
                        disk_hits += 1

                if img is None:
                    if time.time() < self._network_suspended_until:
                        with QMutexLocker(self.mutex):
                            self._skipped_tiles.add(tile_key)
                        skipped_any = True
                        skipped_tiles += 1
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
                                log_view_summary(
                                    "superseded", zoom, total_tiles, len(tiles)
                                )
                                return

                    self.last_network_request_time = time.time()
                    url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_x}/{y}.png"

                    try:
                        resp = session.get(url, timeout=3.0)
                        status_code = resp.status_code
                        if status_code == 200 and is_blocked_tile_image(resp.content):
                            status_code = 403
                        if status_code == 200:
                            network_fetches += 1
                            img_bytes = resp.content
                            try:
                                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            except (UnidentifiedImageError, OSError, ValueError) as error:
                                failed_tiles += 1
                                logging.warning(
                                    "Map tile decode failed: zoom=%d, error_type=%s",
                                    zoom,
                                    type(error).__name__,
                                )
                                continue

                            if self._notified_error:
                                self._notified_error = False
                                self.network_recovered.emit()
                                trigger_retry = True

                            self.ram_cache[tile_key] = img
                            if len(self.ram_cache) > MAX_CACHE_SIZE:
                                self.ram_cache.popitem(last=False)
                            self.disk_cache.put_tile(zoom, wrapped_x, y, img_bytes)

                            with QMutexLocker(self.mutex):
                                self._skipped_tiles.discard(tile_key)
                        else:
                            failed_tiles += 1
                            if status_code in (
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
                                    if status_code in (403, 429, 418):
                                        msg = f"Map service throttled requests (HTTP {status_code})."
                                    elif status_code == 408:
                                        msg = "Map service request timed out."
                                    else:
                                        msg = f"Map service error (HTTP {status_code})."
                                    self.network_error.emit(f"{msg} Backing off.")
                                self._network_suspended_until = time.time() + 15.0
                                with QMutexLocker(self.mutex):
                                    self._skipped_tiles.add(tile_key)
                                skipped_any = True
                                skipped_tiles += 1
                                continue
                    except RequestException as e:
                        failed_tiles += 1
                        if not self._notified_error:
                            self._notified_error = True
                            self.network_error.emit(
                                f"Network error: {e!s}. Backing off."
                            )
                        self._network_suspended_until = time.time() + 15.0
                        with QMutexLocker(self.mutex):
                            self._skipped_tiles.add(tile_key)
                        skipped_any = True
                        skipped_tiles += 1
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
            composite = mercator_rows_to_equirectangular(
                composite, y_min, zoom, south, north
            )
            self.view_ready.emit(job_id, composite, (west, east, south, north))
        if skipped_any:
            self.tiles_skipped.emit()
        elif not tiles:
            self._notified_error = True
            self.network_error.emit(
                "Map tiles are unavailable for the selected view."
            )
        status = (
            "ready"
            if len(tiles) == total_tiles
            else "partial" if tiles else "unavailable"
        )
        log_view_summary(status, zoom, total_tiles, len(tiles))
        if trigger_retry:
            if self._retry_skipped_tiles(session):
                self.retry_complete.emit()

    def _retry_skipped_tiles(self, session: requests.Session) -> bool:
        started_at = time.monotonic()
        with QMutexLocker(self.mutex):
            skipped = sorted(self._skipped_tiles)
        recovered_any = False
        recovered_count = 0
        cached_count = 0
        attempted_count = 0
        failed_count = 0
        status = "complete"
        for tile_key in skipped:
            with QMutexLocker(self.mutex):
                if self._pending_job is not None:
                    logging.debug(
                        "Map tile retry: status=superseded, queued=%d, attempted=%d, "
                        "recovered=%d, cached=%d, duration_ms=%d",
                        len(skipped),
                        attempted_count,
                        recovered_count,
                        cached_count,
                        round((time.monotonic() - started_at) * 1000),
                    )
                    return recovered_any

            zoom, wrapped_x, y = tile_key
            if tile_key in self.ram_cache:
                with QMutexLocker(self.mutex):
                    self._skipped_tiles.discard(tile_key)
                cached_count += 1
                continue

            if time.time() < self._network_suspended_until:
                status = "deferred"
                break

            now = time.time()
            elapsed = now - self.last_network_request_time
            if elapsed < 0.15:
                time.sleep(0.15 - elapsed)
            self.last_network_request_time = time.time()

            url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_x}/{y}.png"
            try:
                attempted_count += 1
                resp = session.get(url, timeout=3.0)
                status_code = resp.status_code
                if status_code == 200 and is_blocked_tile_image(resp.content):
                    status_code = 403
                if status_code == 200:
                    img_bytes = resp.content
                    try:
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    except (UnidentifiedImageError, OSError, ValueError) as error:
                        failed_count += 1
                        status = "failed"
                        logging.warning(
                            "Map tile retry decode failed: zoom=%d, error_type=%s",
                            zoom,
                            type(error).__name__,
                        )
                        continue
                    self.ram_cache[tile_key] = img
                    if len(self.ram_cache) > MAX_CACHE_SIZE:
                        self.ram_cache.popitem(last=False)
                    self.disk_cache.put_tile(zoom, wrapped_x, y, img_bytes)
                    with QMutexLocker(self.mutex):
                        self._skipped_tiles.discard(tile_key)
                    recovered_any = True
                    recovered_count += 1
                else:
                    failed_count += 1
                    status = "failed"
                    if status_code in (403, 429, 418, 408, 500, 502, 503, 504):
                        if not self._notified_error:
                            self._notified_error = True
                            if status_code in (403, 429, 418):
                                msg = f"Map service throttled requests (HTTP {status_code})."
                            elif status_code == 408:
                                msg = "Map service request timed out."
                            else:
                                msg = f"Map service error (HTTP {status_code})."
                            self.network_error.emit(f"{msg} Backing off.")
                        self._network_suspended_until = time.time() + 15.0
                    break
            except RequestException as e:
                failed_count += 1
                status = "failed"
                if not self._notified_error:
                    self._notified_error = True
                    self.network_error.emit(f"Network error: {e!s}. Backing off.")
                self._network_suspended_until = time.time() + 15.0
                break

        with QMutexLocker(self.mutex):
            remaining_count = len(self._skipped_tiles)
            still_skipped = remaining_count > 0
        if still_skipped:
            self.tiles_skipped.emit()

        if still_skipped and status == "complete":
            status = "partial"
        logging.debug(
            "Map tile retry: status=%s, queued=%d, attempted=%d, recovered=%d, "
            "cached=%d, failed=%d, remaining=%d, duration_ms=%d",
            status,
            len(skipped),
            attempted_count,
            recovered_count,
            cached_count,
            failed_count,
            remaining_count,
            round((time.monotonic() - started_at) * 1000),
        )

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


REQUIRED_OBSERVATION_COLUMNS = (
    "eventdate",
    "decimallatitude",
    "decimallongitude",
    "taxonid",
)


def check_observation_columns(con, escaped_path: str) -> None:
    """Raise ValueError if observations.parquet lacks a column the queries use."""
    schema = con.execute(f"DESCRIBE SELECT * FROM '{escaped_path}'").fetchall()
    column_names = [row[0].lower() for row in schema]
    missing_columns = [
        col for col in REQUIRED_OBSERVATION_COLUMNS if col not in column_names
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns in observations.parquet: "
            + ", ".join(missing_columns)
            + ". The file must include eventDate, decimalLatitude, "
            "decimalLongitude, and taxonID."
        )


def register_taxon_filter(con, table_name: str, taxon_ids: list[int]) -> str:
    """Load taxon IDs into a temp table and return an ``IN (SELECT ...)`` operand.

    Higher taxa such as Agaricales expand to tens of thousands of descendant
    IDs. DuckDB evaluates a literal ``IN (1, 2, ...)`` list of that size very
    slowly (minutes over the full database), while a temp-table subquery is
    planned as a hash semi-join and stays under a second.
    """
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
        "SELECT DISTINCT unnest(?::BIGINT[]) AS id",
        [[int(taxon_id) for taxon_id in taxon_ids]],
    )
    return f"(SELECT id FROM {table_name})"


def taxon_filter_sql(
    con, taxon_ids: list[int], exclude_taxon_ids: list[int]
) -> str:
    """Return SQL predicates (each starting with AND) for include/exclude taxa."""
    filter_sql = ""
    if taxon_ids:
        include = register_taxon_filter(con, "include_taxa", taxon_ids)
        filter_sql += f" AND taxonID IN {include}"
    if exclude_taxon_ids:
        exclude = register_taxon_filter(con, "exclude_taxa", exclude_taxon_ids)
        filter_sql += f" AND taxonID NOT IN {exclude}"
    return filter_sql


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
        check_observation_columns(con, escaped_path)
        raise_if_search_cancelled(cancel_callback)

        base_where = """
            WHERE eventDate BETWEEN ? AND ?
            AND decimalLatitude BETWEEN ? AND ?
            AND decimalLongitude BETWEEN ? AND ?
        """
        filter_sql = taxon_filter_sql(con, taxon_ids, exclude_taxon_ids)

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


def season_filter_sql(
    season_start: tuple[int, int], season_end: tuple[int, int]
) -> tuple[str, list[int]]:
    """Return a month-day predicate and params; ranges past Dec 31 wrap to Jan."""
    start = season_start[0] * 100 + season_start[1]
    end = season_end[0] * 100 + season_end[1]
    month_day = "(month(CAST(eventDate AS DATE)) * 100 + day(CAST(eventDate AS DATE)))"
    if start <= end:
        return f"{month_day} BETWEEN ? AND ?", [start, end]
    return f"({month_day} >= ? OR {month_day} <= ?)", [start, end]


def run_local_abundance_query(
    parquet_path: str,
    date_from: str,
    date_to: str,
    season_start: tuple[int, int],
    season_end: tuple[int, int],
    cell_deg: float,
    taxon_ids: list[int],
    exclude_taxon_ids: list[int] | None = None,
    progress_callback=None,
    cancel_callback=None,
    connection_callback=None,
) -> list[dict[str, Any]]:
    """Count a taxon's observations per grid cell worldwide for a season.

    Each returned cell also carries the total number of observations of any
    organism in that cell and season, so callers can rank by raw count or by
    share of all observations (which corrects for how many observers live in
    an area) without re-running the query.
    """
    if not taxon_ids:
        raise ValueError("An organism is required to map where it is most abundant.")
    if cell_deg <= 0:
        raise ValueError("Grid cell size must be positive.")
    exclude_taxon_ids = exclude_taxon_ids or []
    escaped_path = parquet_path.replace("'", "''")
    con = None
    try:
        raise_if_search_cancelled(cancel_callback)
        con = duckdb.connect()
        if connection_callback:
            connection_callback(con)
        if progress_callback:
            progress_callback("Checking observations.parquet schema...")
        check_observation_columns(con, escaped_path)
        raise_if_search_cancelled(cancel_callback)

        include = register_taxon_filter(con, "include_taxa", taxon_ids)
        target_sql = f"taxonID IN {include}"
        if exclude_taxon_ids:
            exclude = register_taxon_filter(con, "exclude_taxa", exclude_taxon_ids)
            target_sql += f" AND taxonID NOT IN {exclude}"
        season_sql, season_params = season_filter_sql(season_start, season_end)
        raise_if_search_cancelled(cancel_callback)

        if progress_callback:
            progress_callback("Scanning all observations for the selected season...")
        query = f"""
            SELECT
                CAST(floor(CAST(decimalLatitude AS DOUBLE) / ?) AS BIGINT) AS lat_index,
                CAST(floor(CAST(decimalLongitude AS DOUBLE) / ?) AS BIGINT) AS lon_index,
                count(*) FILTER (WHERE {target_sql}) AS target_count,
                count(*) AS total_count,
                avg(CAST(decimalLatitude AS DOUBLE)) FILTER (WHERE {target_sql})
                    AS centroid_lat,
                avg(CAST(decimalLongitude AS DOUBLE)) FILTER (WHERE {target_sql})
                    AS centroid_lon
            FROM '{escaped_path}'
            WHERE CAST(eventDate AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            AND {season_sql}
            GROUP BY lat_index, lon_index
            HAVING target_count > 0
        """
        params = [cell_deg, cell_deg, date_from, date_to, *season_params]
        rows = con.execute(query, params).fetchall()
        raise_if_search_cancelled(cancel_callback)

        if progress_callback:
            progress_callback("Processing results...")
        cells = []
        for (
            lat_index,
            lon_index,
            target_count,
            total_count,
            centroid_lat,
            centroid_lon,
        ) in rows:
            cells.append(
                {
                    "cell_lat": lat_index * cell_deg,
                    "cell_lon": lon_index * cell_deg,
                    "cell_deg": cell_deg,
                    "count": int(target_count),
                    "total": int(total_count),
                    "share": target_count / total_count if total_count else 0.0,
                    "centroid_lat": float(centroid_lat),
                    "centroid_lon": float(centroid_lon),
                }
            )
        cells.sort(key=lambda cell: cell["count"], reverse=True)
        return cells
    finally:
        if connection_callback:
            connection_callback(None)
        if con is not None:
            con.close()


class LocalAbundanceWorker(QThread):
    """Background worker for worldwide seasonal abundance queries."""

    progress = pyqtSignal(str)
    search_finished = pyqtSignal(object)
    search_failed = pyqtSignal(str)
    search_cancelled = pyqtSignal()

    def __init__(
        self,
        parquet_path: str,
        date_from: str,
        date_to: str,
        season_start: tuple[int, int],
        season_end: tuple[int, int],
        cell_deg: float,
        taxon_ids: list[int],
        exclude_taxon_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.date_from = date_from
        self.date_to = date_to
        self.season_start = season_start
        self.season_end = season_end
        self.cell_deg = cell_deg
        self.taxon_ids = taxon_ids
        self.exclude_taxon_ids = exclude_taxon_ids or []
        self._cancel_requested = False
        self._connection = None
        self._connection_mutex = QMutex()

    def _set_connection(self, connection) -> None:
        with QMutexLocker(self._connection_mutex):
            self._connection = connection

    def cancel(self) -> None:
        """Cancel the query and interrupt an active DuckDB scan."""
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
            cells = run_local_abundance_query(
                self.parquet_path,
                self.date_from,
                self.date_to,
                self.season_start,
                self.season_end,
                self.cell_deg,
                self.taxon_ids,
                self.exclude_taxon_ids,
                progress_callback=self.progress.emit,
                cancel_callback=self.is_cancelled,
                connection_callback=self._set_connection,
            )
            raise_if_search_cancelled(self.is_cancelled)
            self.search_finished.emit(cells)
        except SearchCancelled:
            self.search_cancelled.emit()
        except Exception as e:
            if self.is_cancelled():
                self.search_cancelled.emit()
            else:
                logging.error("Abundance search worker failed: %s", e)
                self.search_failed.emit(str(e))


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
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
            }
        """)
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


class TileMapInteractionMixin:
    """OpenStreetMap tiles plus pan/zoom handling for a lon/lat Matplotlib axes.

    Hosts provide ``figure``, ``ax``, ``canvas`` and ``status_label``, call
    ``_init_tile_map_state`` before building widgets and ``_start_tile_map``
    once they exist, and override the hook methods for custom clicks/drags.
    """

    _map_network_hint = "you can still adjust the selected area."

    def _init_tile_map_state(self) -> None:
        self.job_counter = 0
        self.current_job_id = 0
        self._network_error_message: str | None = None
        self._drag_mode: str | None = None
        self._press_pixel: tuple[float, float] | None = None
        self._last_pan_pos: tuple[float, float] | None = None
        self._pan_start_limits: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None

    def _set_map_status_styles(self, dark: bool) -> None:
        secondary = "#c8c8c8" if dark else "#52606d"
        warning_bg = "#4a3c20" if dark else "#fff4d6"
        warning_text = "#ffe29a" if dark else "#6b4e00"
        self._map_status_normal_style = f"color: {secondary}; padding: 4px 6px;"
        self._map_status_warning_style = (
            f"background: {warning_bg}; color: {warning_text}; padding: 6px 8px; "
            "border-radius: 4px; font-weight: 600;"
        )

    def _init_map_axes(self, dark: bool) -> None:
        map_background = "#252a30" if dark else "#e7ebef"
        self.figure.set_facecolor(map_background)
        self.ax.set_facecolor(map_background)
        self.ax.set_axis_off()
        if getattr(self, "_map_attribution", None) is not None:
            return
        self._map_attribution = self.ax.text(
            0.99,
            0.015,
            "© OpenStreetMap contributors",
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=self.ax.transAxes,
            color="#333333",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.88,
                edgecolor="#b8b8b8",
                boxstyle="round,pad=0.25",
            ),
            zorder=20,
        )

    def _start_tile_map(self, working_dir: str | Path) -> None:
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200)
        self.update_timer.timeout.connect(self.request_tiles_for_current_view)

        self.retry_timer = QTimer(self)
        self.retry_timer.setSingleShot(True)
        self.retry_timer.timeout.connect(self.request_tiles_for_current_view)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)  # type: ignore[arg-type]
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)  # type: ignore[arg-type]
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)  # type: ignore[arg-type]
        self.canvas.mpl_connect("scroll_event", self.on_scroll)  # type: ignore[arg-type]

        self.worker = TileLoaderWorker(working_dir)
        self.worker.view_ready.connect(self.on_view_ready)
        self.worker.network_error.connect(self.on_network_error)
        self.worker.network_recovered.connect(self.on_network_recovered)
        self.worker.retry_complete.connect(self.request_tiles_for_current_view)
        self.worker.tiles_skipped.connect(self.on_tiles_skipped)
        self.worker.start()

    def _stop_tile_worker(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()

    # Hooks for hosts
    def update_overlays(self) -> None:
        """Refresh artists drawn above the tiles after the view changes."""

    def _on_map_click(self, lat: float, lon: float) -> None:
        """Handle a left click that did not turn into a pan."""

    def _begin_custom_drag(self, event: MouseEvent) -> str | None:
        """Return a drag-mode name to claim a left-button press, or None to pan."""
        return None

    def _custom_drag_move(self, drag_mode: str, event: MouseEvent) -> None:
        """Handle mouse motion for a drag claimed by ``_begin_custom_drag``."""

    def _end_custom_drag(self, drag_mode: str) -> None:
        """Handle release of a drag claimed by ``_begin_custom_drag``."""

    def _set_view_limits(
        self, west: float, east: float, south: float, north: float
    ) -> None:
        center_lon = (west + east) / 2.0
        center_lat = max(
            -WEB_MERCATOR_MAX_LATITUDE,
            min(WEB_MERCATOR_MAX_LATITUDE, (south + north) / 2.0),
        )
        lon_span = max(0.01, east - west)
        lat_span = max(0.01, north - south)

        width = max(1, self.canvas.width())
        height = max(1, self.canvas.height())
        mercator_aspect = 1.0 / max(0.15, math.cos(math.radians(center_lat)))
        desired_lon_lat_ratio = mercator_aspect * width / height
        if lon_span / lat_span < desired_lon_lat_ratio:
            lon_span = lat_span * desired_lon_lat_ratio
        else:
            lat_span = lon_span / desired_lon_lat_ratio

        lon_span = min(360.0, lon_span)
        lat_span = min(WEB_MERCATOR_MAX_LATITUDE * 2.0, lat_span)
        south = center_lat - lat_span / 2.0
        north = center_lat + lat_span / 2.0
        south, north = clamp_latitude_limits(south, north)

        self.ax.set_xlim(center_lon - lon_span / 2.0, center_lon + lon_span / 2.0)
        self.ax.set_ylim(south, north)
        self.ax.set_aspect(mercator_aspect, adjustable="box")

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
        if self._network_error_message is None:
            self.status_label.setText("Loading map…")
            self.status_label.setStyleSheet(self._map_status_normal_style)
            self.status_label.show()
        self.worker.request_view(
            self.current_job_id, zoom, float(x0), float(x1), float(y0), float(y1)
        )

    def on_network_error(self, message: str) -> None:
        self._network_error_message = message
        if "throttled" in message.lower():
            display_message = (
                "The map service is temporarily limiting requests. Retrying "
                f"automatically; {self._map_network_hint}"
            )
        elif "timed out" in message.lower():
            display_message = (
                "Map tiles are taking too long to load. Retrying automatically; "
                f"{self._map_network_hint}"
            )
        else:
            display_message = (
                "Map tiles are temporarily unavailable. Check your connection; "
                f"{self._map_network_hint}"
            )
        self.status_label.setText(display_message)
        self.status_label.setStyleSheet(self._map_status_warning_style)
        self.status_label.show()

    def on_network_recovered(self) -> None:
        self._network_error_message = None
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

        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        self.ax.imshow(
            np.array(composite), extent=extent, origin="upper", alpha=1.0, zorder=1
        )
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        center_lat = (y_limits[0] + y_limits[1]) / 2.0
        self.ax.set_aspect(
            1.0 / max(0.15, math.cos(math.radians(center_lat))),
            adjustable="box",
        )

        self.update_overlays()
        if self._network_error_message is None:
            self.status_label.hide()
        self.canvas.draw_idle()

    def zoom_map(self, scale: float) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        self._zoom_at(cx, cy, scale, 0.5, 0.5)

    def _zoom_at(
        self, mx: float, my: float, scale: float, relx: float, rely: float
    ) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_w = min(360.0, max(0.002, (x1 - x0) * scale))
        new_h = min(
            WEB_MERCATOR_MAX_LATITUDE * 2.0,
            max(0.002, (y1 - y0) * scale),
        )
        new_x0 = mx - relx * new_w
        new_x1 = new_x0 + new_w
        new_y0 = my - rely * new_h
        new_y1 = new_y0 + new_h
        new_y0, new_y1 = clamp_latitude_limits(new_y0, new_y1)
        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()
        self.update_timer.start()

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

        relx = (mx - x0) / w
        rely = (my - y0) / h
        self._zoom_at(float(mx), float(my), scale, relx, rely)


    def on_mouse_press(self, event: MouseEvent) -> None:
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or event.button not in (1, 3)
        ):
            return

        self._press_pixel = (float(event.x), float(event.y))
        self._last_pan_pos = (float(event.xdata), float(event.ydata))
        self._pan_start_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        custom_mode = self._begin_custom_drag(event) if event.button == 1 else None
        if custom_mode:
            self._drag_mode = custom_mode
            self.canvas.setCursor(Qt.CursorShape.SizeAllCursor)
        elif event.button == 3:
            self._drag_mode = "pan"
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._drag_mode = "pan_candidate"

    def on_mouse_release(self, event: MouseEvent) -> None:
        drag_mode = self._drag_mode
        self._drag_mode = None
        self._press_pixel = None
        self._last_pan_pos = None
        self._pan_start_limits = None
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        if drag_mode == "pan_candidate" and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                self._on_map_click(float(event.ydata), float(event.xdata))
        elif drag_mode == "pan":
            self.update_timer.start()
        elif drag_mode is not None:
            self._end_custom_drag(drag_mode)

    def on_mouse_move(self, event: MouseEvent) -> None:
        if (
            self._drag_mode is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        if self._drag_mode == "pan_candidate":
            if self._press_pixel is None:
                return
            movement = math.hypot(
                float(event.x) - self._press_pixel[0],
                float(event.y) - self._press_pixel[1],
            )
            if movement < QApplication.startDragDistance():
                return
            self._drag_mode = "pan"
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

        if self._drag_mode not in ("pan", "pan_candidate"):
            self._custom_drag_move(self._drag_mode, event)
            return

        if (
            self._drag_mode != "pan"
            or self._press_pixel is None
            or self._pan_start_limits is None
        ):
            return
        (x0, x1), (y0, y1) = self._pan_start_limits
        axes_box = self.ax.get_window_extent()
        if axes_box.width <= 0 or axes_box.height <= 0:
            return
        dx = (float(event.x) - self._press_pixel[0]) * (x1 - x0) / axes_box.width
        dy = (float(event.y) - self._press_pixel[1]) * (y1 - y0) / axes_box.height
        self.ax.set_xlim(x0 - dx, x1 - dx)
        new_y0 = y0 - dy
        new_y1 = y1 - dy
        new_y0, new_y1 = clamp_latitude_limits(new_y0, new_y1)
        self.ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()


class MapDialog(TileMapInteractionMixin, QDialog):
    """Professional map picker for the center and radius of a graph search."""

    def __init__(
        self,
        parent: "INatSeasonalVisualizer | None" = None,
        lat: float = 37.7749,
        lon: float = -122.4194,
        radius: float = 10,
    ) -> None:
        super().__init__(parent)
        self._main_window = parent
        self._original_selection = (float(lat), float(lon), float(radius))
        self.lat = max(
            -WEB_MERCATOR_MAX_LATITUDE,
            min(WEB_MERCATOR_MAX_LATITUDE, float(lat)),
        )
        self.lon = normalize_longitude(float(lon))
        self.radius = max(MAP_RADIUS_MIN_KM, min(MAP_RADIUS_MAX_KM, float(radius)))
        self.place_search_worker: PlaceSearchWorker | None = None
        self.place_search_results: list[dict[str, Any]] = []
        self._init_tile_map_state()
        self._initial_view_pending = True
        self._syncing_controls = False
        self._selection_source = "initial"
        self._result_logged = False

        self.setWindowTitle("Choose Search Area")
        self.setModal(True)

        _primary = QApplication.primaryScreen()
        if _primary is not None:
            screen = _primary.availableGeometry()
            self.resize(int(screen.width() * 0.82), int(screen.height() * 0.82))
        else:
            logging.warning("No primary screen detected; using fallback dialog size")
            self.resize(1100, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title_label = QLabel("Choose search area")
        title_label.setObjectName("mapDialogTitle")
        layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Find a place, then click the map to set the center and size the area "
            "with the radius handle."
        )
        subtitle_label.setObjectName("mapDialogSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        self.place_search_input = PlaceSearchLineEdit()
        self.place_search_input.setPlaceholderText(
            "Search for a city, country, park, or other place"
        )
        self.place_search_input.setClearButtonEnabled(True)
        self.place_search_input.search_requested.connect(self.start_place_search)
        self.place_search_button = QPushButton("Search")
        self.place_search_button.clicked.connect(self.start_place_search)
        search_layout.addWidget(self.place_search_input, stretch=1)
        search_layout.addWidget(self.place_search_button)
        layout.addLayout(search_layout)

        self.place_search_status_label = QLabel()
        self.place_search_status_label.setObjectName("placeSearchStatus")
        self.place_search_status_label.setWordWrap(True)
        self.place_search_status_label.hide()
        layout.addWidget(self.place_search_status_label)

        self.place_search_list = QListWidget()
        self.place_search_list.setMaximumHeight(150)
        self.place_search_list.itemClicked.connect(self.activate_place_search_item)
        self.place_search_list.itemActivated.connect(self.activate_place_search_item)
        self.place_search_list.hide()
        layout.addWidget(self.place_search_list)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        map_layout = QVBoxLayout()
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(6)

        map_toolbar = QHBoxLayout()
        map_toolbar.setSpacing(6)
        map_hint = QLabel("Drag to pan  •  Click to place center  •  Scroll to zoom")
        map_hint.setObjectName("mapHint")
        map_toolbar.addWidget(map_hint)
        map_toolbar.addStretch()

        self.fit_area_button = QPushButton("Fit selected area")
        self.fit_area_button.setToolTip("Show the complete selected radius")
        self.fit_area_button.clicked.connect(self.fit_selected_area_and_reload)
        map_toolbar.addWidget(self.fit_area_button)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.setFixedSize(34, 30)
        zoom_in_btn.clicked.connect(lambda: self.zoom_map(1.0 / 1.5))

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.setFixedSize(34, 30)
        zoom_out_btn.clicked.connect(lambda: self.zoom_map(1.5))
        map_toolbar.addWidget(zoom_in_btn)
        map_toolbar.addWidget(zoom_out_btn)
        map_layout.addLayout(map_toolbar)

        self.figure, self.ax = plt.subplots()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumSize(420, 320)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        map_layout.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel()
        self.status_label.setObjectName("mapStatus")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        map_layout.addWidget(self.status_label)
        body_layout.addLayout(map_layout, stretch=1)

        selection_panel = QFrame()
        selection_panel.setObjectName("selectionPanel")
        selection_panel.setMinimumWidth(245)
        selection_panel.setMaximumWidth(320)
        selection_layout = QVBoxLayout(selection_panel)
        selection_layout.setContentsMargins(16, 16, 16, 16)
        selection_layout.setSpacing(10)

        selection_title = QLabel("Selected area")
        selection_title.setObjectName("selectionTitle")
        selection_layout.addWidget(selection_title)

        selection_help = QLabel(
            "These values are used to build the observation graphs. You can edit "
            "them directly or use the map."
        )
        selection_help.setObjectName("selectionHelp")
        selection_help.setWordWrap(True)
        selection_layout.addWidget(selection_help)

        selection_form = QFormLayout()
        selection_form.setSpacing(8)
        self.latitude_spinbox = QDoubleSpinBox()
        self.latitude_spinbox.setDecimals(6)
        self.latitude_spinbox.setRange(
            -WEB_MERCATOR_MAX_LATITUDE, WEB_MERCATOR_MAX_LATITUDE
        )
        self.latitude_spinbox.setSingleStep(0.001)
        self.latitude_spinbox.setKeyboardTracking(False)

        self.longitude_spinbox = QDoubleSpinBox()
        self.longitude_spinbox.setDecimals(6)
        self.longitude_spinbox.setRange(-180.0, 180.0)
        self.longitude_spinbox.setSingleStep(0.001)
        self.longitude_spinbox.setKeyboardTracking(False)

        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setDecimals(1)
        self.radius_spinbox.setRange(MAP_RADIUS_MIN_KM, MAP_RADIUS_MAX_KM)
        self.radius_spinbox.setSingleStep(1.0)
        self.radius_spinbox.setSuffix(" km")
        self.radius_spinbox.setKeyboardTracking(False)
        self.radius_spinbox.setAccelerated(True)

        selection_form.addRow("Latitude", self.latitude_spinbox)
        selection_form.addRow("Longitude", self.longitude_spinbox)
        selection_form.addRow("Radius", self.radius_spinbox)
        selection_layout.addLayout(selection_form)

        handle_help = QLabel(
            "Tip: drag the white handle on the circle for quick visual sizing."
        )
        handle_help.setObjectName("selectionHelp")
        handle_help.setWordWrap(True)
        selection_layout.addWidget(handle_help)
        selection_layout.addStretch()

        body_layout.addWidget(selection_panel)
        layout.addLayout(body_layout, stretch=1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        use_area_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if use_area_button is not None:
            use_area_button.setText("Use selected area")
            use_area_button.setObjectName("useSelectedAreaButton")
            use_area_button.setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._apply_dialog_style(parent)
        self._initialize_map_artists(parent)

        self.latitude_spinbox.valueChanged.connect(self.update_center_from_fields)
        self.longitude_spinbox.valueChanged.connect(self.update_center_from_fields)
        self.radius_spinbox.valueChanged.connect(self.update_radius)
        self._sync_selection_controls()

        self._start_tile_map(parent.working_dir if parent else os.getcwd())

        logging.debug("Map picker opened: radius_km=%.1f", self.radius)

    def _apply_dialog_style(self, parent: "INatSeasonalVisualizer | None") -> None:
        mode = DEFAULT_THEME
        if parent and hasattr(parent, "settings"):
            mode = normalize_theme(parent.settings.value("theme", DEFAULT_THEME))
        dark = mode == "dark"
        panel_bg = "#383838" if dark else "#f4f6f8"
        panel_border = "#555555" if dark else "#d5dbe1"
        secondary = "#c8c8c8" if dark else "#52606d"
        self._set_map_status_styles(dark)
        self.setStyleSheet(self.styleSheet() + f"""
            QLabel#mapDialogTitle {{ font-size: 18pt; font-weight: 700; }}
            QLabel#mapDialogSubtitle, QLabel#mapHint, QLabel#selectionHelp,
            QLabel#placeSearchStatus {{ color: {secondary}; }}
            QFrame#selectionPanel {{
                background-color: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 7px;
            }}
            QFrame#selectionPanel QLabel {{ background: transparent; border: none; }}
            QFrame#selectionPanel QDoubleSpinBox {{
                background-color: {"#2e2e2e" if dark else "#ffffff"};
                border: 1px solid {panel_border};
                border-radius: 4px;
                padding: 4px 6px;
            }}
            QLabel#selectionTitle {{ font-size: 13pt; font-weight: 700; }}
            QPushButton#useSelectedAreaButton {{
                background-color: #2f80ed;
                color: white;
                border: 1px solid #2671d5;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 600;
            }}
            QPushButton#useSelectedAreaButton:hover {{ background-color: #3d8cf3; }}
            """)

    def _initialize_map_artists(self, parent: "INatSeasonalVisualizer | None") -> None:
        mode = DEFAULT_THEME
        if parent and hasattr(parent, "settings"):
            mode = normalize_theme(parent.settings.value("theme", DEFAULT_THEME))
        self._init_map_axes(mode == "dark")

        scale_factor = (
            parent.scale_factor if parent and hasattr(parent, "scale_factor") else 1.0
        )
        accent = "#2f80ed"
        (self.circle,) = self.ax.plot(
            [], [], color=accent, linewidth=2.2, alpha=0.95, zorder=7
        )
        self.radius_fill = self.ax.fill(
            [], [], facecolor=accent, edgecolor="none", alpha=0.18, zorder=6
        )[0]
        (self.marker,) = self.ax.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=10 * scale_factor,
            markerfacecolor=accent,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=10,
        )
        (self.radius_handle,) = self.ax.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=9 * scale_factor,
            markerfacecolor="white",
            markeredgecolor=accent,
            markeredgewidth=2,
            zorder=11,
        )

    def _sync_selection_controls(self) -> None:
        self._syncing_controls = True
        try:
            self.latitude_spinbox.setValue(self.lat)
            self.longitude_spinbox.setValue(normalize_longitude(self.lon))
            self.radius_spinbox.setValue(self.radius)
        finally:
            self._syncing_controls = False

    def _display_longitude(self) -> float:
        try:
            x0, x1 = self.ax.get_xlim()
            reference = (x0 + x1) / 2.0
        except Exception:
            reference = self.lon
        return longitude_near_reference(self.lon, reference)

    def fit_selected_area(
        self, extra_bounds: tuple[float, float, float, float] | None = None
    ) -> None:
        display_lon = self._display_longitude()
        west, east, south, north = radius_view_limits(
            self.lat, display_lon, self.radius
        )
        if extra_bounds is not None:
            extra_west, extra_east, extra_south, extra_north = extra_bounds
            extra_center = (extra_west + extra_east) / 2.0
            display_lon = longitude_near_reference(display_lon, extra_center)
            radius_bounds = radius_view_limits(self.lat, display_lon, self.radius)
            west = min(extra_west, radius_bounds[0])
            east = max(extra_east, radius_bounds[1])
            south = min(extra_south, radius_bounds[2])
            north = max(extra_north, radius_bounds[3])
        self._set_view_limits(west, east, south, north)
        self.update_overlays()
        self.canvas.draw_idle()

    def fit_selected_area_and_reload(self) -> None:
        self.fit_selected_area()
        self.request_tiles_for_current_view()

    def ensure_selection_visible(self) -> bool:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        display_lon = longitude_near_reference(self.lon, (x0 + x1) / 2.0)
        if not (x0 <= display_lon <= x1 and y0 <= self.lat <= y1):
            self.fit_selected_area()
            return True

        circle_lons, circle_lats = geodesic_circle_coordinates(
            self.lat, display_lon, self.radius
        )
        lon_margin = (x1 - x0) * 0.06
        lat_margin = (y1 - y0) * 0.06
        if (
            min(circle_lons) >= x0 + lon_margin
            and max(circle_lons) <= x1 - lon_margin
            and min(circle_lats) >= y0 + lat_margin
            and max(circle_lats) <= y1 - lat_margin
        ):
            return False

        circle_bounds = radius_view_limits(self.lat, display_lon, self.radius, 0.12)
        self._set_view_limits(
            min(x0, circle_bounds[0]),
            max(x1, circle_bounds[1]),
            min(y0, circle_bounds[2]),
            max(y1, circle_bounds[3]),
        )
        self.update_overlays()
        self.canvas.draw_idle()
        return True

    def _stop_workers(self) -> None:
        if self.place_search_worker and self.place_search_worker.isRunning():
            self.place_search_worker.cancel()
            self.place_search_worker.wait()
        self._stop_tile_worker()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self._log_map_result("cancelled")
        self._stop_workers()
        super().closeEvent(a0)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if self._initial_view_pending:
            self._initial_view_pending = False
            QTimer.singleShot(0, self._initialize_initial_view)

    def _initialize_initial_view(self) -> None:
        if not self.isVisible():
            self._initial_view_pending = True
            return
        self.fit_selected_area()
        self.request_tiles_for_current_view()

    def _log_map_result(self, status: str) -> None:
        if self._result_logged:
            return
        self._result_logged = True
        logging.debug(
            "Map picker %s: radius_km=%.1f, last_change=%s",
            status,
            self.radius,
            self._selection_source,
        )

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
            "Choose a result to move the selected area."
        )
        self.place_search_status_label.show()
        self.place_search_list.setCurrentRow(-1)
        self.place_search_list.show()
        self.place_search_list.setFocus()

    def on_place_search_failed(self, _message: str) -> None:
        logging.warning("Place search could not be completed.")
        self.place_search_status_label.setText(
            "Place search is unavailable. Check your network connection and try again."
        )
        self.place_search_status_label.show()
        self.place_search_list.hide()

    def on_place_search_finished(self) -> None:
        self.place_search_button.setEnabled(True)

    def activate_place_search_item(self, item) -> None:
        self.select_place_search_result(self.place_search_list.row(item))

    def select_place_search_result(self, row: int) -> None:
        if row < 0 or row >= len(self.place_search_results):
            return
        place = self.place_search_results[row]
        self.lat = max(
            -WEB_MERCATOR_MAX_LATITUDE,
            min(WEB_MERCATOR_MAX_LATITUDE, float(place["lat"])),
        )
        self.lon = normalize_longitude(float(place["lon"]))
        self._selection_source = "place_search"
        self._sync_selection_controls()
        self.fit_selected_area(place_result_view_limits(place))
        self.place_search_input.setText(place["display_name"])
        self.place_search_status_label.setText(
            f'Showing "{place["display_name"]}". Click the map to fine-tune the center.'
        )
        self.place_search_list.hide()
        logging.debug(
            "Map place selected: result_index=%d, admin_level=%r, has_bounds=%s",
            row,
            place.get("admin_level"),
            place.get("bounds") is not None,
        )
        self.request_tiles_for_current_view()

    def update_overlays(self) -> None:
        display_lon = self._display_longitude()
        self.marker.set_data([display_lon], [self.lat])
        circle_lons, circle_lats = geodesic_circle_coordinates(
            self.lat, display_lon, self.radius
        )
        self.circle.set_data(circle_lons, circle_lats)
        self.radius_fill.set_xy(np.column_stack((circle_lons, circle_lats)))
        handle_lat, handle_lon = geodesic_destination(
            self.lat, display_lon, self.radius, 90.0
        )
        self.radius_handle.set_data([handle_lon], [handle_lat])

    def _set_selected_center(
        self, lat: float, lon: float, *, source: str = "map_click"
    ) -> None:
        self.lat = max(
            -WEB_MERCATOR_MAX_LATITUDE,
            min(WEB_MERCATOR_MAX_LATITUDE, float(lat)),
        )
        self.lon = normalize_longitude(float(lon))
        self._selection_source = source
        self._sync_selection_controls()
        self.update_overlays()
        viewport_changed = self.ensure_selection_visible()
        self.canvas.draw_idle()
        if viewport_changed:
            self.update_timer.start()

    def update_center_from_fields(self, _value: float) -> None:
        if self._syncing_controls:
            return
        self._set_selected_center(
            self.latitude_spinbox.value(),
            self.longitude_spinbox.value(),
            source="coordinate_fields",
        )

    def update_radius(self, value: float) -> None:
        if self._syncing_controls:
            return
        self.radius = max(MAP_RADIUS_MIN_KM, min(MAP_RADIUS_MAX_KM, float(value)))
        self._selection_source = "radius_field"
        self.update_overlays()
        viewport_changed = self.ensure_selection_visible()
        self.canvas.draw_idle()
        if viewport_changed:
            self.update_timer.start()

    def accept(self) -> None:
        if self._selection_source == "initial":
            self.lat, self.lon, self.radius = self._original_selection
        self._log_map_result("accepted")
        self._stop_workers()
        super().accept()

    def reject(self) -> None:
        self._log_map_result("cancelled")
        self._stop_workers()
        super().reject()

    def _on_map_click(self, lat: float, lon: float) -> None:
        self._set_selected_center(lat, lon)

    def _begin_custom_drag(self, event: MouseEvent) -> str | None:
        return "radius" if self._event_near_radius_handle(event) else None

    def _custom_drag_move(self, drag_mode: str, event: MouseEvent) -> None:
        if drag_mode != "radius":
            return
        cursor_lon = longitude_near_reference(float(event.xdata), self.lon)
        new_radius = geodesic_distance_km(
            self.lat, self.lon, float(event.ydata), cursor_lon
        )
        self.radius = max(MAP_RADIUS_MIN_KM, min(MAP_RADIUS_MAX_KM, new_radius))
        self._selection_source = "radius_handle"
        self._sync_selection_controls()
        self.update_overlays()
        self.canvas.draw_idle()

    def _end_custom_drag(self, drag_mode: str) -> None:
        if drag_mode != "radius":
            return
        viewport_changed = self.ensure_selection_visible()
        self.canvas.draw_idle()
        if viewport_changed:
            self.update_timer.start()

    def _event_near_radius_handle(self, event: MouseEvent) -> bool:
        handle_x = np.asarray(self.radius_handle.get_xdata())
        handle_y = np.asarray(self.radius_handle.get_ydata())
        if not len(handle_x) or not len(handle_y):
            return False
        pixel_x, pixel_y = self.ax.transData.transform((handle_x[0], handle_y[0]))
        tolerance = 14.0 * self.figure.dpi / 72.0
        return (
            math.hypot(float(event.x) - pixel_x, float(event.y) - pixel_y) <= tolerance
        )

ABUNDANCE_TOP_CELLS = 25
ABUNDANCE_GRID_OPTIONS = (("0.25°", 0.25), ("0.5°", 0.5), ("1°", 1.0), ("2°", 2.0))
ABUNDANCE_DEFAULT_GRID_INDEX = 2


def abundance_cell_key(cell: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        round(float(cell["cell_lat"]), 6),
        round(float(cell["cell_lon"]), 6),
        float(cell["cell_deg"]),
    )


def rank_abundance_cells(
    cells: list[dict[str, Any]], metric: str, min_count: int, limit: int | None = None
) -> list[dict[str, Any]]:
    """Order cells by raw count, or by share among cells with enough records.

    Share mode needs a minimum count because a cell with one observation of
    the organism and two observations overall would otherwise outrank a
    genuine hotspot.
    """
    if metric == "share":
        ranked = [cell for cell in cells if cell["count"] >= min_count]
        ranked.sort(key=lambda cell: (cell["share"], cell["count"]), reverse=True)
    else:
        ranked = sorted(cells, key=lambda cell: cell["count"], reverse=True)
    return ranked[:limit] if limit is not None else ranked


def abundance_grid_array(
    cells: list[dict[str, Any]], cell_deg: float, metric: str
) -> np.ndarray:
    """Return a global lat x lon array (row 0 = south) of the chosen metric."""
    rows = int(round(180.0 / cell_deg))
    cols = int(round(360.0 / cell_deg))
    grid = np.full((rows, cols), np.nan)
    for cell in cells:
        row = int(round((cell["cell_lat"] + 90.0) / cell_deg))
        col = int(round((cell["cell_lon"] + 180.0) / cell_deg)) % cols
        if 0 <= row < rows:
            grid[row, col] = cell["share"] * 100.0 if metric == "share" else cell["count"]
    return grid


def abundance_cell_radius_km(cell: Mapping[str, Any]) -> float:
    """Radius of a circle centred on a grid cell that covers the whole cell."""
    cell_deg = float(cell["cell_deg"])
    south = max(-90.0, float(cell["cell_lat"]))
    north = min(90.0, south + cell_deg)
    west = float(cell["cell_lon"])
    center_lat = (south + north) / 2.0
    center_lon = west + cell_deg / 2.0
    corner_distance = max(
        geodesic_distance_km(center_lat, center_lon, lat, lon)
        for lat in (south, north)
        for lon in (west, west + cell_deg)
    )
    return max(MAP_RADIUS_MIN_KM, min(MAP_RADIUS_MAX_KM, math.ceil(corner_distance)))


def most_specific_standard_place(response: Any) -> str | None:
    """Pick the finest-grained administrative place from a places/nearby reply."""
    results = response.get("results") if isinstance(response, Mapping) else None
    standard = results.get("standard") if isinstance(results, Mapping) else None
    candidates = [
        place
        for place in standard or []
        if isinstance(place, Mapping)
        and place.get("admin_level") is not None
        and (place.get("display_name") or place.get("name"))
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda place: (
            place.get("admin_level"),
            -(place.get("bbox_area") or 0.0),
        ),
    )
    return str(best.get("display_name") or best.get("name"))


class PlaceNameWorker(QThread):
    """Look up readable names for abundance cells without blocking the UI."""

    name_ready = pyqtSignal(object, str)

    def __init__(
        self,
        requests_to_run: list[tuple[Any, float, float]],
        http_cache_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.requests_to_run = requests_to_run
        self.http_cache_file = (
            str(http_cache_file) if http_cache_file is not None else ":memory:"
        )

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        request_session = None
        try:
            request_session = pyinaturalist.ClientSession(
                cache_file=self.http_cache_file,
                max_retries=0,
                timeout=8,
            )
            for key, lat, lon in self.requests_to_run:
                if self.isInterruptionRequested():
                    return
                delta = 0.005
                response = pyinaturalist.get_places_nearby(
                    nelat=lat + delta,
                    nelng=lon + delta,
                    swlat=lat - delta,
                    swlng=lon - delta,
                    per_page=30,
                    session=request_session,
                )
                name = most_specific_standard_place(response)
                if name and not self.isInterruptionRequested():
                    self.name_ready.emit(key, name)
        except Exception as error:
            if not self.isInterruptionRequested():
                logging.warning(
                    "Abundance place-name lookup stopped: %s",
                    place_search_error_diagnostic(error),
                )
        finally:
            if request_session is not None:
                request_session.close()


class AbundanceMapView(TileMapInteractionMixin, QWidget):
    """World heatmap and ranked list showing where an organism is most observed."""

    graph_requested = pyqtSignal(float, float, float)  # lat, lon, radius_km

    _map_network_hint = "the ranked list is still available."

    def __init__(self, main_window: "INatSeasonalVisualizer | None" = None) -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self._init_tile_map_state()
        self.cells: list[dict[str, Any]] = []
        self.ranked_cells: list[dict[str, Any]] = []
        self.metric = "count"
        self.min_count = 10
        self.cell_deg = 1.0
        self.context: dict[str, Any] = {}
        self.place_names: dict[tuple[float, float, float], str] = {}
        self._name_workers: list[PlaceNameWorker] = []
        self._initial_view_pending = True
        self._heatmap_images: list[Any] = []
        self._colorbar = None
        self._colorbar_ax = None
        self._highlight = None
        self.map_visible = True
        if main_window is not None and hasattr(main_window, "settings"):
            self.map_visible = main_window.settings.value(
                "abundance_show_map", True, bool
            )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        map_layout = QVBoxLayout()
        map_layout.setSpacing(6)
        map_toolbar = QHBoxLayout()
        map_toolbar.setSpacing(6)
        self.map_title_label = QLabel(
            "Choose an organism and season, then click Map abundance."
        )
        self.map_title_label.setObjectName("abundanceTitle")
        self.map_title_label.setWordWrap(True)
        map_toolbar.addWidget(self.map_title_label, stretch=1)
        self.map_toggle_button = QPushButton()
        self.map_toggle_button.setCheckable(True)
        self.map_toggle_button.setChecked(not self.map_visible)
        self.map_toggle_button.toggled.connect(
            lambda hidden: self.set_map_visible(not hidden)
        )
        map_toolbar.addWidget(self.map_toggle_button)
        self.fit_results_button = QPushButton("Fit results")
        self.fit_results_button.setToolTip("Zoom to the top-ranked locations")
        self.fit_results_button.clicked.connect(self.fit_results_and_reload)
        map_toolbar.addWidget(self.fit_results_button)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.setFixedSize(34, 30)
        zoom_in_btn.clicked.connect(lambda: self.zoom_map(1.0 / 1.5))
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.setFixedSize(34, 30)
        zoom_out_btn.clicked.connect(lambda: self.zoom_map(1.5))
        map_toolbar.addWidget(zoom_in_btn)
        map_toolbar.addWidget(zoom_out_btn)
        map_layout.addLayout(map_toolbar)

        self.figure, self.ax = plt.subplots()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumSize(0, 0)
        map_layout.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        map_layout.addWidget(self.status_label)
        layout.addLayout(map_layout, stretch=3)

        panel = QFrame()
        panel.setObjectName("abundancePanel")
        panel.setMinimumWidth(260)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)
        self.summary_label = QLabel("Top locations")
        self.summary_label.setWordWrap(True)
        panel_layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Location", "Obs", "Share"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.itemDoubleClicked.connect(lambda _item: self.request_graph())
        panel_layout.addWidget(self.table, stretch=1)

        self.graph_here_button = QPushButton("Graph seasonality here")
        self.graph_here_button.setToolTip(
            "Use the selected location as the center of a seasonal graph"
        )
        self.graph_here_button.setEnabled(False)
        self.graph_here_button.clicked.connect(self.request_graph)
        panel_layout.addWidget(self.graph_here_button)
        layout.addWidget(panel, stretch=1)

        self.apply_theme(self._is_dark())
        self._start_tile_map(main_window.working_dir if main_window else os.getcwd())

    def _is_dark(self) -> bool:
        if self._main_window is not None and hasattr(self._main_window, "settings"):
            theme = self._main_window.settings.value("theme", DEFAULT_THEME)
            return normalize_theme(theme) == "dark"
        return DEFAULT_THEME == "dark"

    def apply_theme(self, dark: bool) -> None:
        self._set_map_status_styles(dark)
        if self._network_error_message is None:
            self.status_label.setStyleSheet(self._map_status_normal_style)
        self._init_map_axes(dark)
        self._apply_map_visibility()

    def set_map_visible(self, visible: bool) -> None:
        """Show the OpenStreetMap base map, or only the heatmap on black."""
        self.map_visible = visible
        if self._main_window is not None and hasattr(self._main_window, "settings"):
            self._main_window.settings.setValue("abundance_show_map", visible)
        self.map_toggle_button.blockSignals(True)
        self.map_toggle_button.setChecked(not visible)
        self.map_toggle_button.blockSignals(False)
        if not visible:
            self.retry_timer.stop()
            self.update_timer.stop()
            self.status_label.hide()
        self._apply_map_visibility()
        if visible:
            self.request_tiles_for_current_view()

    def _apply_map_visibility(self) -> None:
        visible = self.map_visible
        self.map_toggle_button.setText("Hide map" if visible else "Show map")
        self.map_toggle_button.setToolTip(
            "Show only the observation cells on a black background"
            if visible
            else "Show the OpenStreetMap base map under the observation cells"
        )
        if visible:
            self._init_map_axes(self._is_dark())
        else:
            self.figure.set_facecolor("black")
            self.ax.set_facecolor("black")
        if getattr(self, "_map_attribution", None) is not None:
            self._map_attribution.set_visible(visible)
        heatmap_ids = {id(image) for image in self._heatmap_images}
        for image in self.ax.images:
            if id(image) in heatmap_ids:
                image.set_alpha(self._heatmap_alpha())
            else:
                image.set_visible(visible)
        self._style_colorbar()
        if self._highlight is not None:
            self._highlight.set_edgecolor(self._highlight_color())
        self.canvas.draw_idle()

    def _heatmap_alpha(self) -> float:
        return 0.72 if self.map_visible else 1.0

    def _highlight_color(self) -> str:
        return "#2f80ed" if self.map_visible else "#ffffff"

    def _style_colorbar(self) -> None:
        if self._colorbar is None or self._colorbar_ax is None:
            return
        text_color = "#222222" if self.map_visible else "#dddddd"
        if self._colorbar.solids is not None:
            self._colorbar.solids.set_alpha(self._heatmap_alpha())
        self._colorbar.ax.xaxis.label.set_color(text_color)
        self._colorbar_ax.tick_params(colors=text_color)

    def request_tiles_for_current_view(self) -> None:
        if not self.map_visible:
            return
        super().request_tiles_for_current_view()

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if self._initial_view_pending:
            self._initial_view_pending = False
            QTimer.singleShot(0, self._initialize_initial_view)

    def _initialize_initial_view(self) -> None:
        if self.ranked_cells:
            self.fit_results()
        else:
            self._set_view_limits(-170.0, 190.0, -60.0, 75.0)
        self.request_tiles_for_current_view()

    def shutdown(self) -> None:
        for worker in self._name_workers:
            worker.cancel()
        for worker in self._name_workers:
            worker.wait(2000)
        self._stop_tile_worker()

    def set_results(
        self,
        cells: list[dict[str, Any]],
        cell_deg: float,
        metric: str,
        min_count: int,
        context: dict[str, Any],
    ) -> None:
        self.cells = cells
        self.cell_deg = cell_deg
        self.context = context
        self.metric = metric
        self.min_count = min_count
        self._rerank()
        self.fit_results()
        self.request_tiles_for_current_view()

    def set_ranking(self, metric: str, min_count: int) -> None:
        self.metric = metric
        self.min_count = min_count
        if self.cells:
            self._rerank()

    def _rerank(self) -> None:
        self.ranked_cells = rank_abundance_cells(
            self.cells, self.metric, self.min_count, ABUNDANCE_TOP_CELLS
        )
        self._draw_heatmap()
        self._populate_table()
        self._update_summary()
        self._request_place_names()
        self.canvas.draw_idle()

    def _update_summary(self) -> None:
        organism = self.context.get("organism", "")
        season = self.context.get("season_label", "")
        self.map_title_label.setText(
            f"Where {organism} is observed, {season}" if organism else ""
        )
        total = sum(cell["count"] for cell in self.cells)
        metric_text = (
            f"share of all observations (cells with ≥ {self.min_count} obs)"
            if self.metric == "share"
            else "observation count"
        )
        self.summary_label.setText(
            f"Top {len(self.ranked_cells)} of {len(self.cells):,} "
            f"{self.cell_deg:g}° cells by {metric_text}. "
            f"{total:,} observations in total."
        )

    def _draw_heatmap(self) -> None:
        # Remove the colorbar's inset axes while its image still belongs to
        # the map axes; Colorbar.remove() looks the image's axes up.
        self._colorbar = None
        if self._colorbar_ax is not None:
            self._colorbar_ax.remove()
            self._colorbar_ax = None
        for image in self._heatmap_images:
            image.remove()
        self._heatmap_images = []
        self._set_highlight(None)

        if self.metric == "share":
            visible = [c for c in self.cells if c["count"] >= self.min_count]
        else:
            visible = self.cells
        if not visible:
            return
        grid = abundance_grid_array(visible, self.cell_deg, self.metric)
        finite = grid[np.isfinite(grid)]
        if self.metric == "share":
            norm = Normalize(vmin=0.0, vmax=max(float(finite.max()), 1e-9))
            label = "% of all observations"
        else:
            norm = LogNorm(vmin=1.0, vmax=max(float(finite.max()), 2.0))
            label = "Observations"

        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        for offset in (-360.0, 0.0, 360.0):
            image = self.ax.imshow(
                np.ma.masked_invalid(grid),
                extent=(-180.0 + offset, 180.0 + offset, -90.0, 90.0),
                origin="lower",
                cmap="YlOrRd",
                norm=norm,
                alpha=self._heatmap_alpha(),
                interpolation="nearest",
                zorder=3,
            )
            self._heatmap_images.append(image)
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)

        self._colorbar_ax = self.ax.inset_axes((0.02, 0.05, 0.28, 0.025))
        self._colorbar = self.figure.colorbar(
            self._heatmap_images[1], cax=self._colorbar_ax, orientation="horizontal"
        )
        self._colorbar.set_label(label, fontsize=8)
        self._colorbar_ax.tick_params(labelsize=7)
        self._colorbar_ax.patch.set_alpha(0.0)
        self._colorbar_ax.set_zorder(21)
        self._style_colorbar()

    def on_view_ready(
        self, job_id: int, composite, extent: tuple[float, float, float, float]
    ) -> None:
        # The base implementation removes every image on the axes, which would
        # include the heatmap, so keep the heatmap images out of that sweep.
        if job_id < self.current_job_id or not self.map_visible:
            return
        heatmap_ids = {id(image) for image in self._heatmap_images}
        for image in list(self.ax.images):
            if id(image) not in heatmap_ids:
                image.remove()
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        self.ax.imshow(
            np.array(composite), extent=extent, origin="upper", alpha=1.0, zorder=1
        )
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        center_lat = (y_limits[0] + y_limits[1]) / 2.0
        self.ax.set_aspect(
            1.0 / max(0.15, math.cos(math.radians(center_lat))),
            adjustable="box",
        )
        if self._network_error_message is None:
            self.status_label.hide()
        self.canvas.draw_idle()

    def _location_label(self, cell: Mapping[str, Any]) -> str:
        name = self.place_names.get(abundance_cell_key(cell))
        coordinates = f"{cell['centroid_lat']:.2f}, {cell['centroid_lon']:.2f}"
        return f"{name} ({coordinates})" if name else coordinates

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(self.ranked_cells))
        for row, cell in enumerate(self.ranked_cells):
            values = (
                str(row + 1),
                self._location_label(cell),
                f"{cell['count']:,}",
                f"{cell['share'] * 100:.1f}%",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                item.setToolTip(
                    f"{cell['count']:,} of {cell['total']:,} observations in this "
                    f"{cell['cell_deg']:g}° cell during the season"
                )
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.graph_here_button.setEnabled(False)

    def _request_place_names(self) -> None:
        pending = [
            (
                abundance_cell_key(cell),
                float(cell["centroid_lat"]),
                float(cell["centroid_lon"]),
            )
            for cell in self.ranked_cells
            if abundance_cell_key(cell) not in self.place_names
        ]
        for worker in self._name_workers:
            worker.cancel()
        if not pending:
            return
        cache_file = getattr(self._main_window, "http_cache_file", None)
        worker = PlaceNameWorker(pending, cache_file)
        worker.name_ready.connect(self._on_place_name)
        worker.finished.connect(lambda w=worker: self._forget_name_worker(w))
        self._name_workers.append(worker)
        worker.start()

    def _forget_name_worker(self, worker: PlaceNameWorker) -> None:
        if worker in self._name_workers:
            self._name_workers.remove(worker)

    def _on_place_name(self, key: tuple[float, float, float], name: str) -> None:
        self.place_names[key] = name
        for row, cell in enumerate(self.ranked_cells):
            if abundance_cell_key(cell) == key:
                item = self.table.item(row, 1)
                if item is not None:
                    item.setText(self._location_label(cell))

    def selected_cell(self) -> dict[str, Any] | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return self.ranked_cells[row] if 0 <= row < len(self.ranked_cells) else None

    def _on_row_selected(self) -> None:
        cell = self.selected_cell()
        self.graph_here_button.setEnabled(cell is not None)
        self._set_highlight(cell)
        if cell is None:
            self.canvas.draw_idle()
            return
        span = max(self.cell_deg * 5.0, 2.0)
        center_lat = cell["cell_lat"] + self.cell_deg / 2.0
        center_lon = cell["cell_lon"] + self.cell_deg / 2.0
        self._set_view_limits(
            center_lon - span, center_lon + span, center_lat - span, center_lat + span
        )
        self.canvas.draw_idle()
        self.update_timer.start()

    def _set_highlight(self, cell: Mapping[str, Any] | None) -> None:
        if self._highlight is not None:
            self._highlight.remove()
            self._highlight = None
        if cell is None:
            return
        self._highlight = Rectangle(
            (cell["cell_lon"], cell["cell_lat"]),
            cell["cell_deg"],
            cell["cell_deg"],
            fill=False,
            edgecolor=self._highlight_color(),
            linewidth=2.5,
            zorder=8,
        )
        self.ax.add_patch(self._highlight)

    def fit_results(self) -> None:
        if not self.ranked_cells:
            return
        lats = [c["cell_lat"] for c in self.ranked_cells]
        lons = [c["cell_lon"] for c in self.ranked_cells]
        pad = max(self.cell_deg * 2.0, 1.0)
        self._set_view_limits(
            min(lons) - pad,
            max(lons) + self.cell_deg + pad,
            min(lats) - pad,
            max(lats) + self.cell_deg + pad,
        )
        self.canvas.draw_idle()

    def fit_results_and_reload(self) -> None:
        self.fit_results()
        self.request_tiles_for_current_view()

    def _on_map_click(self, lat: float, lon: float) -> None:
        lon = normalize_longitude(lon)
        for row, cell in enumerate(self.ranked_cells):
            if (
                cell["cell_lat"] <= lat < cell["cell_lat"] + cell["cell_deg"]
                and cell["cell_lon"] <= lon < cell["cell_lon"] + cell["cell_deg"]
            ):
                self.table.selectRow(row)
                return

    def request_graph(self) -> None:
        cell = self.selected_cell()
        if cell is None:
            return
        self.graph_requested.emit(
            round(cell["cell_lat"] + cell["cell_deg"] / 2.0, 4),
            round(normalize_longitude(cell["cell_lon"] + cell["cell_deg"] / 2.0), 4),
            float(abundance_cell_radius_km(cell)),
        )


MONTH_ABBREVIATIONS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
# Feb 29 stays selectable; observations on it only exist in leap years.
DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
SEASON_PRESETS: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...] = (
    ("Spring (Mar – May)", (3, 1), (5, 31)),
    ("Summer (Jun – Aug)", (6, 1), (8, 31)),
    ("Fall (Sep – Nov)", (9, 1), (11, 30)),
    ("Winter (Dec – Feb)", (12, 1), (2, 29)),
) + tuple(
    (MONTH_NAMES[month - 1], (month, 1), (month, DAYS_IN_MONTH[month - 1]))
    for month in range(1, 13)
)
SEASON_CUSTOM_TEXT = "Custom range"
SEASON_AROUND_TODAY_TEXT = "Around today (± 2 weeks)"


def season_day_of_year(month_day: tuple[int, int]) -> int:
    """Day number of a month/day in a leap year (Jan 1 = 1, Dec 31 = 366)."""
    month, day = month_day
    return sum(DAYS_IN_MONTH[: month - 1]) + day


def season_length_days(start: tuple[int, int], end: tuple[int, int]) -> int:
    return (season_day_of_year(end) - season_day_of_year(start)) % 366 + 1


def format_month_day(month_day: tuple[int, int]) -> str:
    return f"{MONTH_ABBREVIATIONS[month_day[0] - 1]} {month_day[1]}"


def around_date_season(
    today: datetime, days: int = 14
) -> tuple[tuple[int, int], tuple[int, int]]:
    start = today - pd.Timedelta(days=days)
    end = today + pd.Timedelta(days=days)
    return (start.month, start.day), (end.month, end.day)


class SeasonRangeWidget(QWidget):
    """Pick a month/day range, with presets, that may wrap past Dec 31."""

    changed = pyqtSignal()

    def __init__(
        self,
        start: tuple[int, int] = (10, 1),
        end: tuple[int, int] = (11, 30),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._updating = False
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setColumnStretch(1, 1)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem(SEASON_CUSTOM_TEXT)
        self.preset_combo.addItem(SEASON_AROUND_TODAY_TEXT)
        self.preset_combo.insertSeparator(self.preset_combo.count())
        for index, (label, _start, _end) in enumerate(SEASON_PRESETS):
            if index == 4:
                self.preset_combo.insertSeparator(self.preset_combo.count())
            self.preset_combo.addItem(label)
        self.preset_combo.setToolTip("Quickly choose a common time of year")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.start_month, self.start_day = self._month_day_pair()
        self.end_month, self.end_day = self._month_day_pair()
        self.summary_label = QLabel()
        self.summary_label.setObjectName("seasonSummary")

        layout.addWidget(QLabel("Season:"), 0, 0)
        layout.addWidget(self.preset_combo, 0, 1, 1, 2)
        layout.addWidget(QLabel("From"), 1, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.start_month, 1, 1)
        layout.addWidget(self.start_day, 1, 2)
        layout.addWidget(QLabel("To"), 2, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.end_month, 2, 1)
        layout.addWidget(self.end_day, 2, 2)
        layout.addWidget(self.summary_label, 3, 1, 1, 2)
        self.set_range(start, end)

    def _month_day_pair(self) -> tuple[QComboBox, QComboBox]:
        month_combo = QComboBox()
        month_combo.addItems(MONTH_ABBREVIATIONS)
        month_combo.setMinimumContentsLength(3)
        month_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        day_combo = QComboBox()
        day_combo.setMinimumContentsLength(2)
        day_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        day_combo.setMaxVisibleItems(16)
        month_combo.currentIndexChanged.connect(
            lambda _index, m=month_combo, d=day_combo: self._on_month_changed(m, d)
        )
        day_combo.currentIndexChanged.connect(self._on_fields_changed)
        return month_combo, day_combo

    @staticmethod
    def _fill_days(month_combo: QComboBox, day_combo: QComboBox, day: int) -> None:
        month = month_combo.currentIndex() + 1
        count = DAYS_IN_MONTH[month - 1]
        day_combo.blockSignals(True)
        day_combo.clear()
        day_combo.addItems([str(value) for value in range(1, count + 1)])
        day_combo.setCurrentIndex(min(max(day, 1), count) - 1)
        day_combo.blockSignals(False)

    def _on_month_changed(self, month_combo: QComboBox, day_combo: QComboBox) -> None:
        self._fill_days(month_combo, day_combo, day_combo.currentIndex() + 1)
        self._on_fields_changed()

    def _set_pair(
        self, month_combo: QComboBox, day_combo: QComboBox, month_day: tuple[int, int]
    ) -> None:
        month, day = month_day
        month_combo.blockSignals(True)
        month_combo.setCurrentIndex(month - 1)
        month_combo.blockSignals(False)
        self._fill_days(month_combo, day_combo, day)

    def set_range(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self._updating = True
        try:
            self._set_pair(self.start_month, self.start_day, start)
            self._set_pair(self.end_month, self.end_day, end)
        finally:
            self._updating = False
        self._on_fields_changed()

    def start(self) -> tuple[int, int]:
        return (self.start_month.currentIndex() + 1, self.start_day.currentIndex() + 1)

    def end(self) -> tuple[int, int]:
        return (self.end_month.currentIndex() + 1, self.end_day.currentIndex() + 1)

    def label(self) -> str:
        return f"{format_month_day(self.start())} – {format_month_day(self.end())}"

    def _apply_preset(self, text: str) -> None:
        if self._updating:
            return
        if text == SEASON_AROUND_TODAY_TEXT:
            start, end = around_date_season(datetime.now())
        else:
            match = next((p for p in SEASON_PRESETS if p[0] == text), None)
            if match is None:
                return
            _label, start, end = match
        self.set_range(start, end)
        if text == SEASON_AROUND_TODAY_TEXT:
            self._select_preset(SEASON_AROUND_TODAY_TEXT)

    def _select_preset(self, text: str) -> None:
        self._updating = True
        try:
            self.preset_combo.setCurrentText(text)
        finally:
            self._updating = False

    def _on_fields_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        start, end = self.start(), self.end()
        preset = next(
            (label for label, p_start, p_end in SEASON_PRESETS
             if (p_start, p_end) == (start, end)),
            SEASON_CUSTOM_TEXT,
        )
        self._select_preset(preset)
        days = season_length_days(start, end)
        wraps = season_day_of_year(end) < season_day_of_year(start)
        self.summary_label.setText(
            f"{days} day{'s' if days != 1 else ''}"
            + (" · crosses Jan 1" if wraps else "")
        )
        self.changed.emit()


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
        self.abundance_worker = None
        self.abundance_view = None
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
                logging.info(
                    "Successfully downloaded %s to %s",
                    filename,
                    privacy_safe_path(file_path),
                )
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
                    privacy_safe_path(self.taxon_cache_file),
                    len(self.taxon_cache),
                    descendant_lists,
                )
                for key, value in self.taxon_cache.items():
                    if key.endswith("_descendants"):
                        logging.debug(
                            "Loaded %d descendant taxon IDs for %s",
                            len(value),
                            key.removesuffix("_descendants"),
                        )
            else:
                logging.info(
                    "No taxon cache found at %s",
                    privacy_safe_path(self.taxon_cache_file),
                )
        except Exception as e:
            logging.error("Failed to load taxon cache: %s", privacy_safe_message(e))
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
            logging.info(
                "Saved taxon cache to %s", privacy_safe_path(self.taxon_cache_file)
            )
        except Exception as e:
            logging.error("Failed to save taxon cache: %s", privacy_safe_message(e))

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
                            "Loaded %d descendant taxon IDs for %s from %s",
                            len(taxon_ids),
                            query,
                            privacy_safe_path(self.descendant_taxons_file),
                        )
                        return taxon_ids
            return None
        except Exception as e:
            logging.error(
                "Failed to load descendant taxons from %s: %s",
                privacy_safe_path(self.descendant_taxons_file),
                privacy_safe_message(e),
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

        self.map_button = QPushButton("Choose Location on Map…")
        self.map_button.setObjectName("mapPickerButton")
        self.map_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.map_button.clicked.connect(self.open_map_dialog)
        self.map_button.setToolTip("Choose the graph center and radius on a map")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([SEASONAL_MODE_TEXT, ABUNDANCE_MODE_TEXT])
        self.mode_combo.setToolTip(
            "Seasonal graph: when is an organism seen at one place?\n"
            "Abundance map: where is it seen most during part of the year?"
        )
        if not self.local_database_available:
            abundance_item = self.mode_combo.model().item(1)
            if abundance_item is not None:
                abundance_item.setEnabled(False)
                abundance_item.setToolTip(
                    "Download observations.parquet to map abundance worldwide."
                )

        self.season_input = SeasonRangeWidget()
        self.season_input.setToolTip(
            "Time of year to map, matched in every year between Date From and "
            "Date To. A range such as Nov 15 to Feb 10 crosses the new year."
        )
        self.grid_combo = QComboBox()
        for label, _cell_deg in ABUNDANCE_GRID_OPTIONS:
            self.grid_combo.addItem(label)
        self.grid_combo.setCurrentIndex(ABUNDANCE_DEFAULT_GRID_INDEX)
        self.grid_combo.setToolTip("Size of each map cell in degrees")
        self.rank_combo = QComboBox()
        self.rank_combo.addItems(["Observation count", "Share of all observations"])
        self.rank_combo.setToolTip(
            "Count favours places with many iNaturalist users. Share divides by "
            "all observations in the cell during the season to correct for that."
        )
        self.min_obs_input = QSpinBox()
        self.min_obs_input.setRange(1, 100000)
        self.min_obs_input.setValue(10)
        self.min_obs_input.setToolTip(
            "When ranking by share, ignore cells with fewer observations of the "
            "organism than this"
        )
        self.rank_combo.currentIndexChanged.connect(self._on_abundance_ranking_changed)
        self.min_obs_input.valueChanged.connect(self._on_abundance_ranking_changed)

        self.sidebar_layout.addRow("Mode:", self.mode_combo)
        self.sidebar_layout.addRow("Latitude:", self.lat_input)
        self.sidebar_layout.addRow("Longitude:", self.lon_input)
        self.sidebar_layout.addRow("Radius (km):", self.radius_input)
        self.sidebar_layout.addRow("", self.map_button)
        self.sidebar_layout.addRow("Organism:", self.organism_input)
        self.sidebar_layout.addRow("Exclude:", self.exclude_input)
        self.sidebar_layout.addRow("Date From:", self.date_from)
        self.sidebar_layout.addRow("Date To:", self.date_to)
        # Full-width row: the month/day pickers need more room than a field.
        self.sidebar_layout.addRow(self.season_input)
        self.sidebar_layout.addRow("Grid:", self.grid_combo)
        self.sidebar_layout.addRow("Rank By:", self.rank_combo)
        self.sidebar_layout.addRow("Min Obs:", self.min_obs_input)
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

        self.abundance_button = QPushButton(ABUNDANCE_ACTION_TEXT)
        self.abundance_button.clicked.connect(self.abundance_search)
        self.abundance_button.setToolTip(
            "Scan the local database for where the organism is observed most "
            "during the chosen season"
        )
        self.sidebar_layout.addWidget(self.abundance_button)

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
            # The abundance map page is created on first use so its tile
            # worker does not start (or touch the network) during startup.
            self.view_stack = QStackedWidget()
            self.view_stack.addWidget(self.canvas)
            self.top_layout.addWidget(self.view_stack, 3)

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

        self.mode_combo.currentIndexChanged.connect(self._apply_search_mode)
        self._apply_search_mode()

    def is_abundance_mode(self) -> bool:
        return self.mode_combo.currentText() == ABUNDANCE_MODE_TEXT

    def _apply_search_mode(self, _index: int | None = None) -> None:
        """Show the sidebar rows and page that belong to the selected mode."""
        abundance = self.is_abundance_mode()
        for widget in (
            self.lat_input,
            self.lon_input,
            self.radius_input,
            self.map_button,
            self.view_combo,
        ):
            self.sidebar_layout.setRowVisible(widget, not abundance)
        for widget in (
            self.season_input,
            self.grid_combo,
            self.rank_combo,
            self.min_obs_input,
        ):
            self.sidebar_layout.setRowVisible(widget, abundance)
        self.local_search_button.setVisible(not abundance)
        self.search_button.setVisible(not abundance)
        self.show_url_button.setVisible(not abundance)
        self.abundance_button.setVisible(abundance)
        self._update_min_obs_enabled()
        if not hasattr(self, "view_stack"):
            return
        if abundance:
            self.view_stack.setCurrentWidget(self._ensure_abundance_view())
        else:
            self.view_stack.setCurrentWidget(self.canvas)

    def _ensure_abundance_view(self) -> "AbundanceMapView":
        if self.abundance_view is None:
            self.abundance_view = AbundanceMapView(self)
            self.abundance_view.graph_requested.connect(self.graph_abundance_cell)
            self.view_stack.addWidget(self.abundance_view)
        return self.abundance_view

    def _abundance_metric(self) -> str:
        return "share" if self.rank_combo.currentIndex() == 1 else "count"

    def _update_min_obs_enabled(self) -> None:
        self.min_obs_input.setEnabled(self._abundance_metric() == "share")

    def _on_abundance_ranking_changed(self, _value: int | None = None) -> None:
        self._update_min_obs_enabled()
        if self.abundance_view is not None:
            self.abundance_view.set_ranking(
                self._abundance_metric(), self.min_obs_input.value()
            )

    def graph_abundance_cell(self, lat: float, lon: float, radius: float) -> None:
        """Switch to a seasonal graph centred on an abundance-map cell."""
        self.lat_input.setText(f"{lat:.4f}")
        self.lon_input.setText(f"{lon:.4f}")
        self.radius_input.setText(f"{radius:g}")
        self.mode_combo.setCurrentText(SEASONAL_MODE_TEXT)
        self.local_search()

    def run_preferred_search(self) -> None:
        """Run a local search when available, otherwise use the online API."""
        if self.is_abundance_mode():
            self.abundance_search()
        elif self.local_database_available:
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
            map_picker_button_stylesheet = """
                QPushButton#mapPickerButton {
                    background-color: #50555b;
                    color: #ffffff;
                    border: 2px solid #858d96;
                    border-radius: 6px;
                    padding: 7px 14px;
                    font-weight: 700;
                }
                QPushButton#mapPickerButton:hover {
                    background-color: #5d636a;
                    border-color: #a0a7af;
                }
                QPushButton#mapPickerButton:pressed {
                    background-color: #454a50;
                    border-color: #737b84;
                }
            """
            # UI dark mode
            self.setStyleSheet(
                font_stylesheet
                + f"background-color: {window_bg_color}; color: {window_text_color};"
            )
            self.central_widget.setStyleSheet(
                "QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, "
                "QLabel { background-color: #3e3e3e; color: #ffffff; }"
                + map_picker_button_stylesheet
            )
        else:
            map_picker_button_stylesheet = """
                QPushButton#mapPickerButton {
                    background-color: #e2e5e9;
                    color: #202428;
                    border: 2px solid #8a939d;
                    border-radius: 6px;
                    padding: 7px 14px;
                    font-weight: 700;
                }
                QPushButton#mapPickerButton:hover {
                    background-color: #d5d9de;
                    border-color: #6f7882;
                }
                QPushButton#mapPickerButton:pressed {
                    background-color: #c7ccd2;
                    border-color: #626b75;
                }
            """
            # UI light mode
            self.setStyleSheet(
                font_stylesheet
                + f"background-color: {window_bg_color}; color: {window_text_color};"
            )
            self.central_widget.setStyleSheet(
                "QLineEdit, QComboBox, QPushButton, QListWidget, QProgressBar, "
                "QLabel { background-color: #ffffff; color: #000000; }"
                + map_picker_button_stylesheet
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
        if self.abundance_view is not None:
            self.abundance_view.apply_theme(normalize_theme(mode) == "dark")

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
        except ValueError:
            lat = float(self.default_lat)
        try:
            lon = float(self.lon_input.text().strip())
        except ValueError:
            lon = float(self.default_lon)
        try:
            radius = float(self.radius_input.text())
        except ValueError:
            radius = float(self.default_radius)

        dialog = MapDialog(self, lat, lon, radius)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.lat_input.setText(f"{dialog.lat:.6f}")
            self.lon_input.setText(f"{dialog.lon:.6f}")
            self.radius_input.setText(f"{dialog.radius:.1f}")

    def load_settings(self) -> None:
        """Load saved settings."""
        self.lat_input.setText(self.settings.value("latitude", self.default_lat))
        self.lon_input.setText(self.settings.value("longitude", self.default_lon))
        self.radius_input.setText(
            self.settings.value("radius", str(self.default_radius))
        )
        self.color_input.setText(self.settings.value("graph_color", "#1f77b4"))
        self.bg_color_input.setText(self.settings.value("graph_bg_color", ""))
        saved_season = []
        for key in ("abundance_season_from", "abundance_season_to"):
            saved = QDate.fromString(
                f"{ABUNDANCE_SEASON_YEAR}-{self.settings.value(key, '')}", "yyyy-MM-dd"
            )
            if saved.isValid():
                saved_season.append((saved.month(), saved.day()))
        if len(saved_season) == 2:
            self.season_input.set_range(*saved_season)
        grid_index = self.settings.value(
            "abundance_grid_index", ABUNDANCE_DEFAULT_GRID_INDEX, int
        )
        if 0 <= grid_index < len(ABUNDANCE_GRID_OPTIONS):
            self.grid_combo.setCurrentIndex(grid_index)
        self.rank_combo.setCurrentIndex(
            1 if self.settings.value("abundance_rank_by", "count") == "share" else 0
        )
        self.min_obs_input.setValue(self.settings.value("abundance_min_obs", 10, int))

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
            self.settings.setValue(
                "abundance_season_from",
                "{:02d}-{:02d}".format(*self.season_input.start()),
            )
            self.settings.setValue(
                "abundance_season_to", "{:02d}-{:02d}".format(*self.season_input.end())
            )
            self.settings.setValue(
                "abundance_grid_index", self.grid_combo.currentIndex()
            )
            self.settings.setValue("abundance_rank_by", self._abundance_metric())
            self.settings.setValue("abundance_min_obs", self.min_obs_input.value())
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        except ValueError as e:
            QMessageBox.warning(self, "Settings", f"Invalid input: {e!s}")

    def update_api_call_count(self) -> None:
        """Update the API call count display."""
        self.api_call_label.setText(f"API Calls: {self.api_call_count}")

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km using Haversine formula."""
        return geodesic_distance_km(lat1, lon1, lat2, lon2)

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

            taxon_ids, exclude_taxon_ids = self._resolve_local_taxon_filters(
                organism, exclude
            )

            # Start database progress tracking with estimated progress
            self.enhanced_progress.db_tracker.start_operation("local database search")
            self.status_bar.showMessage("Performing local search...")
            self.local_search_button.setEnabled(False)
            self.search_button.setEnabled(False)
            self.show_url_button.setEnabled(False)
            self.mode_combo.setEnabled(False)
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
                self.mode_combo.setEnabled(True)
                self.cancel_search_button.setEnabled(False)
                self.cancel_search_button.setText("Cancel Search")
                self.local_search_worker = None
            QMessageBox.critical(self, "Error", f"Local search failed: {e!s}")
            self.show_placeholder(f"Local search failed: {e!s}")
            logging.error(f"Local search failed: {e!s}")

    def _resolve_local_taxon_filters(
        self, organism: str, exclude: str
    ) -> tuple[list[int], list[int]]:
        """Expand the organism and exclude fields to descendant taxon IDs."""
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

        exclude_taxon_ids = []
        if exclude:
            exclude_taxon_id = self.get_taxon_id(exclude)
            if exclude_taxon_id:
                exclude_taxon_ids = self.get_descendant_taxon_ids(
                    exclude, exclude_taxon_id
                )
        return taxon_ids, exclude_taxon_ids

    def _set_search_running(self, running: bool) -> None:
        """Toggle the search buttons while a background search is active."""
        self.local_search_button.setEnabled(
            not running and self.local_database_available
        )
        self.abundance_button.setEnabled(not running and self.local_database_available)
        self.search_button.setEnabled(not running)
        self.show_url_button.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.cancel_search_button.setText("Cancel Search")
        self.cancel_search_button.setEnabled(running)

    def season_label(self) -> str:
        return self.season_input.label()

    def abundance_search(self) -> None:
        """Map where the organism is observed most during the chosen season."""
        if self.abundance_worker is not None and self.abundance_worker.isRunning():
            self.status_bar.showMessage("Abundance search is already running...")
            return
        if not self.local_database_available or not os.path.exists(
            self.observations_file
        ):
            QMessageBox.warning(
                self,
                "Local Database Required",
                "Mapping abundance scans every observation worldwide, so it needs "
                "the local observations.parquet database.",
            )
            return

        organism = self.organism_input.text().strip()
        if not organism:
            QMessageBox.warning(
                self, "Organism Required", "Enter an organism to map its abundance."
            )
            return

        try:
            exclude = self.exclude_input.text().strip()
            date_from = self.date_from.text().strip()
            date_to = self.date_to.text().strip()
            season_start = self.season_input.start()
            season_end = self.season_input.end()
            cell_deg = ABUNDANCE_GRID_OPTIONS[self.grid_combo.currentIndex()][1]

            taxon_ids, exclude_taxon_ids = self._resolve_local_taxon_filters(
                organism, exclude
            )
            if not taxon_ids:
                return

            self._ensure_abundance_view()
            self.enhanced_progress.db_tracker.start_operation("abundance map search")
            self.status_bar.showMessage("Mapping abundance from local data...")
            self._set_search_running(True)

            worker = LocalAbundanceWorker(
                self.observations_file,
                date_from,
                date_to,
                season_start,
                season_end,
                cell_deg,
                taxon_ids,
                exclude_taxon_ids,
            )
            self.abundance_worker = worker
            context = {
                "organism": organism,
                "date_from": date_from,
                "date_to": date_to,
                "season_label": self.season_label(),
                "cell_deg": cell_deg,
            }
            worker.progress.connect(self._on_local_search_progress)
            worker.search_finished.connect(
                lambda cells, context=context: self._on_abundance_finished(
                    cells, context
                )
            )
            worker.search_failed.connect(self._on_abundance_failed)
            worker.search_cancelled.connect(self._on_abundance_cancelled)
            worker.finished.connect(self._on_abundance_worker_done)
            worker.start()
        except Exception as e:
            self.enhanced_progress.hide_progress()
            if self.abundance_worker is None or not self.abundance_worker.isRunning():
                self._set_search_running(False)
                self.abundance_worker = None
            QMessageBox.critical(self, "Error", f"Abundance search failed: {e!s}")
            logging.error("Abundance search failed: %s", e)

    def _on_abundance_finished(
        self, cells: list[dict[str, Any]], context: dict[str, Any]
    ) -> None:
        if self.abundance_worker is not None and self.abundance_worker.is_cancelled():
            self._on_abundance_cancelled()
            return
        total = sum(cell["count"] for cell in cells)
        self.enhanced_progress.db_tracker.estimated_total = total
        self.enhanced_progress.db_tracker.finish_operation(
            total, "Abundance search completed"
        )
        view = self._ensure_abundance_view()
        view.set_results(
            cells,
            context["cell_deg"],
            self._abundance_metric(),
            self.min_obs_input.value(),
            context,
        )
        if not cells:
            self.status_bar.showMessage(
                f"No observations of {context['organism']} found in that season."
            )
        else:
            self.status_bar.showMessage(
                f"Found {total:,} observations in {len(cells):,} map cells."
            )
        history_item = QListWidgetItem(
            f"{context['organism']} | {context['season_label']} abundance (Local)"
        )
        history_item.setData(
            Qt.ItemDataRole.UserRole, {"cells": cells, "context": context}
        )
        self.history_list.addItem(history_item)
        self.info_bar.setText(
            f"Source: Local abundance map, Organism: {context['organism']}, "
            f"Season: {context['season_label']}, "
            f"Dates: {context['date_from']} to {context['date_to']}, "
            f"Grid: {context['cell_deg']:g}°, Cells: {len(cells):,}, "
            f"Observations: {total:,}"
        )

    def _on_abundance_failed(self, error: str) -> None:
        self.enhanced_progress.hide_progress()
        QMessageBox.critical(self, "Error", f"Abundance search failed: {error}")
        logging.error("Abundance search failed: %s", error)

    def _on_abundance_cancelled(self) -> None:
        self.enhanced_progress.finish_progress("Abundance search cancelled")
        self.status_bar.showMessage("Abundance search cancelled.")

    def _on_abundance_worker_done(self) -> None:
        self._set_search_running(False)
        self.abundance_worker = None

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
        self.mode_combo.setEnabled(True)
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
        cancelling_abundance = self.abundance_worker is not None
        if not cancelling_api and not cancelling_local and not cancelling_abundance:
            self.cancel_search_button.setEnabled(False)
            self.status_bar.showMessage("No search is currently running.")
            return

        self.cancel_search_button.setText("Cancelling...")
        self.cancel_search_button.setEnabled(False)
        if cancelling_abundance:
            self.abundance_worker.cancel()
            message = "Cancelling abundance search..."
        elif cancelling_local:
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
            self.abundance_worker,
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
        if self.abundance_view is not None:
            self.abundance_view.shutdown()
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
            self.mode_combo.setEnabled(False)
            self.cancel_search_button.setText("Cancel Search")
            self.cancel_search_button.setEnabled(True)
            worker.start()

        except Exception as e:
            self.enhanced_progress.hide_progress()
            self.local_search_button.setEnabled(self.local_database_available)
            self.search_button.setEnabled(True)
            self.search_button.setText(LIVE_GRAPH_ACTION_TEXT)
            self.show_url_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
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
        self.mode_combo.setEnabled(True)
        self.cancel_search_button.setEnabled(False)
        self.cancel_search_button.setText("Cancel Search")
        self.api_search_worker = None

    def export_graph(self) -> None:
        """Export the current graph as an image with metadata."""
        if self.is_abundance_mode():
            self.export_abundance_map()
            return
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

    def export_abundance_data(self) -> None:
        """Export the abundance map cells, ranked as currently shown, as CSV."""
        view = self.abundance_view
        if view is None or not view.cells:
            QMessageBox.warning(self, "Error", "No abundance map data to export.")
            return
        file_dialog = QFileDialog(self, "Export Abundance Data", "", "CSV (*.csv)")
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if not file_dialog.exec():
            return
        filename = file_dialog.selectedFiles()[0]
        try:
            rows = []
            for rank, cell in enumerate(
                rank_abundance_cells(view.cells, view.metric, view.min_count), start=1
            ):
                rows.append(
                    {
                        "rank": rank,
                        "place": view.place_names.get(abundance_cell_key(cell), ""),
                        "cell_south_lat": cell["cell_lat"],
                        "cell_west_lon": cell["cell_lon"],
                        "cell_size_deg": cell["cell_deg"],
                        "centroid_lat": round(cell["centroid_lat"], 4),
                        "centroid_lon": round(cell["centroid_lon"], 4),
                        "observations": cell["count"],
                        "all_observations": cell["total"],
                        "share": round(cell["share"], 6),
                    }
                )
            pd.DataFrame(rows).to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {e!s}")
            logging.error("Export abundance data failed: %s", e)

    def export_abundance_map(self) -> None:
        """Save the abundance map as an image."""
        view = self.abundance_view
        if view is None or not view.cells:
            QMessageBox.warning(self, "Error", "No abundance map to export.")
            return
        file_dialog = QFileDialog(
            self, "Export Abundance Map", "", "PNG (*.png);;JPG (*.jpg)"
        )
        file_dialog.setDefaultSuffix("png")
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if not file_dialog.exec():
            return
        filename = file_dialog.selectedFiles()[0]
        try:
            view.figure.savefig(filename, dpi=200, facecolor=view.figure.get_facecolor())
            QMessageBox.information(self, "Success", f"Map exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export map: {e!s}")
            logging.error("Export abundance map failed: %s", e)

    def export_data(self) -> None:
        """Export observation data as CSV."""
        if self.is_abundance_mode():
            self.export_abundance_data()
            return
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
        current_item = self.history_list.currentItem()
        abundance = (
            current_item.data(Qt.ItemDataRole.UserRole)
            if current_item is not None
            else None
        )
        if isinstance(abundance, dict) and "cells" in abundance:
            self.mode_combo.setCurrentText(ABUNDANCE_MODE_TEXT)
            context = abundance["context"]
            self._ensure_abundance_view().set_results(
                abundance["cells"],
                context["cell_deg"],
                self._abundance_metric(),
                self.min_obs_input.value(),
                context,
            )
            return

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
        privacy_safe_path(working_dir),
    )

    if args.debug:
        logging.debug("Debug logging enabled (level=%s).", log_level)

    # Check environment
    if args.debug:
        logging.debug("Checking environment...")
    env_error = check_environment()
    if env_error:
        print(env_error)
        sys.exit(1)
    if args.debug:
        logging.debug("Environment check passed.")

    # --- High DPI stability setup (Wayland + Qt6) ---
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
    )

    # Create QApplication first
    if args.debug:
        logging.debug("Creating QApplication...")
    app = QApplication(sys.argv)
    app.setApplicationName("iNaturalist Seasonal Visualizer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AlanRockefeller")
    if args.debug:
        logging.debug("QApplication created.")

    if args.smoke_test:
        logging.info(
            "Packaged application smoke test passed for version %s", __version__
        )
        return

    # Show splash screen
    splash_image_path = resource_path("splash_screen.jpg")
    if os.path.exists(splash_image_path):
        if args.debug:
            logging.debug(
                "Splash screen found at %s; initializing.",
                privacy_safe_path(splash_image_path),
            )
        splash = CustomSplashScreen(splash_image_path)
        splash.update_status("Starting iNaturalist Seasonal Visualizer...")
        QApplication.processEvents()
    else:
        if args.debug:
            logging.debug(
                "Splash screen not found at %s; skipping.",
                privacy_safe_path(splash_image_path),
            )
        splash = None
        logging.warning(
            "Splash screen image not found at %s",
            privacy_safe_path(splash_image_path),
        )

    # Initialize main window with splash screen updates
    if splash:
        splash.update_status("Loading application components...")
        QApplication.processEvents()

    if args.debug:
        logging.debug("Initializing main window (INatSeasonalVisualizer)...")
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
        logging.debug("Main window initialized.")

    if splash:
        splash.update_status("Application ready!")
        QApplication.processEvents()
        if args.debug:
            logging.debug("Closing splash screen...")
        splash.close()

    if args.debug:
        logging.debug("Showing main window maximized...")
    window.showMaximized()
    QTimer.singleShot(0, window.start_database_update_check)

    if args.debug:
        logging.debug("Starting event loop (app.exec)...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

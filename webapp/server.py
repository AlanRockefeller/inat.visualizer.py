"""FastAPI backend for the web version of the iNaturalist Seasonal Visualizer.

Run locally from the repository root:

    INAT_DATA_DIR=. uvicorn webapp.server:app --port 8765

Taxon and place names are resolved in the browser against api.inaturalist.org,
so this server only ever receives numeric taxon IDs and never calls iNat itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inat_core import (  # noqa: E402
    MAP_RADIUS_MAX_KM,
    MAP_RADIUS_MIN_KM,
    abundance_cell_radius_km,
    descendant_taxon_ids,
    run_local_abundance_query,
    run_local_date_counts,
    run_local_observation_query,
    seasonal_bin_counts,
)
from inat_visualizer_version import __version__  # noqa: E402

logger = logging.getLogger("inat_visualizer.web")

DATA_DIR = Path(os.environ.get("INAT_DATA_DIR", ".")).resolve()
OBSERVATIONS_FILE = DATA_DIR / "observations.parquet"
TAXONOMY_FILE = DATA_DIR / "taxonomy.parquet"
MAX_CONCURRENT_QUERIES = int(os.environ.get("INAT_MAX_CONCURRENT_QUERIES", "2"))
QUERY_TIMEOUT_SECONDS = float(os.environ.get("INAT_QUERY_TIMEOUT", "90"))
DUCKDB_MEMORY_LIMIT = os.environ.get("INAT_DUCKDB_MEMORY", "4GB")
DUCKDB_THREADS = int(os.environ.get("INAT_DUCKDB_THREADS", "4"))
# In-memory DuckDB spills to ./.tmp by default, which is read-only in production.
DUCKDB_TEMP_DIR = os.environ.get(
    "INAT_DUCKDB_TEMP_DIR", os.path.join(tempfile.gettempdir(), "inat-visualizer-duckdb")
)
RESULT_CACHE_ENTRIES = int(os.environ.get("INAT_RESULT_CACHE_ENTRIES", "256"))
# Budget for cached results, measured as the size of their JSON. Broad
# abundance searches at 0.25 degrees return tens of thousands of cells, so an
# entry count alone would not bound memory.
RESULT_CACHE_MB = float(os.environ.get("INAT_RESULT_CACHE_MB", "64"))
MAX_CSV_ROWS = int(os.environ.get("INAT_MAX_CSV_ROWS", "500000"))
MAX_TAXON_FILTERS = 10
# How often a running query checks whether the browser has gone away.
DISCONNECT_POLL_SECONDS = 0.5

if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", DUCKDB_MEMORY_LIMIT, re.IGNORECASE):
    raise RuntimeError(f"INAT_DUCKDB_MEMORY must look like '4GB', not {DUCKDB_MEMORY_LIMIT!r}")

ABUNDANCE_CELL_SIZES = (0.25, 0.5, 1.0, 2.0)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MONTH_DAY_PATTERN = r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

app = FastAPI(
    title="iNaturalist Seasonal Visualizer",
    version=__version__,
    docs_url=None,
    redoc_url=None,
)


class SecurityHeadersMiddleware:
    """Add security headers to every response.

    They are set here rather than in nginx because an ``add_header`` inside an
    nginx ``location`` replaces every header inherited from the ``server``
    block (such as Strict-Transport-Security). This is a plain ASGI middleware
    rather than ``@app.middleware("http")``: Starlette's BaseHTTPMiddleware
    hides client disconnects from the handlers below it, which
    ``run_limited_query`` relies on to interrupt abandoned queries.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=list(message.get("headers") or []))
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
                message["headers"] = headers.raw
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityHeadersMiddleware)


class LruCache:
    """Small thread-safe LRU cache that empties when the database files change.

    ``max_bytes`` bounds the total of ``size_of(value)`` over the entries; a
    value larger than the whole budget is not cached at all.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int | None = None,
        size_of: Callable[[Any], int] | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._size_of = size_of
        self._items: OrderedDict[Any, Any] = OrderedDict()
        self._sizes: dict[Any, int] = {}
        self._total_bytes = 0
        self._signature: tuple | None = None
        self._lock = threading.Lock()

    def _check_signature(self) -> None:
        signature = database_signature()
        if signature != self._signature:
            self._items.clear()
            self._sizes.clear()
            self._total_bytes = 0
            self._signature = signature

    def _evict(self, key: Any) -> None:
        del self._items[key]
        self._total_bytes -= self._sizes.pop(key, 0)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: Any) -> Any:
        with self._lock:
            self._check_signature()
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: Any, value: Any, signature: tuple | None = None) -> None:
        """Store ``value``, unless the database changed after ``signature`` was taken.

        A weekly update can swap the Parquet files while a query is running.
        Callers take the signature before the query starts; a result computed
        against the old files is then dropped instead of being cached under
        the new files' signature.
        """
        with self._lock:
            self._check_signature()
            if signature is not None and signature != self._signature:
                return
            size = self._size_of(value) if self._size_of else 0
            if self.max_bytes is not None and size > self.max_bytes:
                return
            if key in self._items:
                self._evict(key)
            self._items[key] = value
            self._sizes[key] = size
            self._total_bytes += size
            while len(self._items) > self.max_entries or (
                self.max_bytes is not None and self._total_bytes > self.max_bytes
            ):
                self._evict(next(iter(self._items)))


def database_signature() -> tuple:
    """Identify the current database files, so a weekly update clears caches."""
    signature = []
    for path in (OBSERVATIONS_FILE, TAXONOMY_FILE):
        try:
            stat = path.stat()
            signature.append((stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            signature.append(None)
    return tuple(signature)


def json_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":")))


descendant_cache = LruCache(2048)
result_cache = LruCache(
    RESULT_CACHE_ENTRIES, max_bytes=int(RESULT_CACHE_MB * 1024 * 1024), size_of=json_size
)
query_semaphore: asyncio.Semaphore | None = None


def get_query_semaphore() -> asyncio.Semaphore:
    global query_semaphore
    if query_semaphore is None:
        query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    return query_semaphore


def require_database() -> None:
    missing = [p.name for p in (OBSERVATIONS_FILE, TAXONOMY_FILE) if not p.exists()]
    if missing:
        raise HTTPException(503, f"Database not installed on the server: {', '.join(missing)}")


def configure_connection(connection) -> None:
    temp_dir = DUCKDB_TEMP_DIR.replace("'", "''")
    connection.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET threads = {DUCKDB_THREADS}")
    connection.execute(f"SET temp_directory = '{temp_dir}'")


async def run_limited_query(
    function: Callable[..., Any], request: Request | None = None, **kwargs: Any
) -> Any:
    """Run a DuckDB query in a thread with concurrency, memory and time limits.

    Starlette keeps running a handler after the browser has given up, so the
    request is watched for a client disconnect both while it waits for a slot
    and while the query runs. On timeout or disconnect the DuckDB connection
    is interrupted, and the concurrency slot is only released once the thread
    has really stopped.
    """
    state: dict[str, Any] = {"connection": None, "cancelled": False}
    lock = threading.Lock()

    def connection_callback(connection) -> None:
        if connection is not None:
            configure_connection(connection)
        with lock:
            state["connection"] = connection

    async def stop(task: asyncio.Future) -> None:
        state["cancelled"] = True
        with lock:
            if state["connection"] is not None:
                state["connection"].interrupt()
        await asyncio.gather(task, return_exceptions=True)

    async def disconnected() -> bool:
        return request is not None and await request.is_disconnected()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + QUERY_TIMEOUT_SECONDS
    semaphore = get_query_semaphore()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise HTTPException(503, "The server is busy. Please try again in a moment.")
        try:
            await asyncio.wait_for(
                semaphore.acquire(), min(remaining, DISCONNECT_POLL_SECONDS)
            )
            break
        except asyncio.TimeoutError:
            if await disconnected():
                raise HTTPException(499, "The client closed the request") from None

    try:
        if await disconnected():
            raise HTTPException(499, "The client closed the request")
        task = asyncio.ensure_future(
            asyncio.to_thread(
                function,
                cancel_callback=lambda: state["cancelled"],
                connection_callback=connection_callback,
                **kwargs,
            )
        )
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    await stop(task)
                    raise HTTPException(
                        504,
                        "The query took too long. Try a smaller radius, date range or taxon.",
                    )
                done, _pending = await asyncio.wait(
                    {task}, timeout=min(remaining, DISCONNECT_POLL_SECONDS)
                )
                if done:
                    return task.result()
                if await disconnected():
                    await stop(task)
                    raise HTTPException(499, "The client closed the request")
        except asyncio.CancelledError:
            await stop(task)
            raise
        except duckdb.OutOfMemoryException:
            raise HTTPException(
                503, "The server ran out of memory for this query. Try a narrower search."
            ) from None
    finally:
        semaphore.release()


async def expand_taxa(
    taxon_ids: list[int], request: Request | None = None
) -> list[int]:
    """Expand each taxon to itself plus all of its descendants."""
    expanded: set[int] = set()
    for taxon_id in taxon_ids:
        descendants = descendant_cache.get(taxon_id)
        if descendants is None:
            signature = database_signature()
            descendants = await run_limited_query(
                descendant_taxon_ids,
                request=request,
                taxonomy_path=str(TAXONOMY_FILE),
                taxon_id=taxon_id,
            )
            descendant_cache.put(taxon_id, descendants, signature)
        expanded.update(descendants)
    return sorted(expanded)


def validate_date_range(date_from: str, date_to: str) -> None:
    try:
        start = dt.date.fromisoformat(date_from)
        end = dt.date.fromisoformat(date_to)
    except ValueError as error:
        raise HTTPException(422, f"Invalid date: {error}") from None
    if start > end:
        raise HTTPException(422, "date_from must be on or before date_to")


def parse_month_day(value: str) -> tuple[int, int]:
    month, day = (int(part) for part in value.split("-"))
    return month, day


def validate_taxon_list(name: str, values: list[int]) -> list[int]:
    if len(values) > MAX_TAXON_FILTERS:
        raise HTTPException(422, f"At most {MAX_TAXON_FILTERS} {name} values are allowed")
    return values


@app.get("/api/stats")
async def stats(request: Request) -> dict[str, Any]:
    require_database()
    cached = result_cache.get("stats")
    if cached is None:
        signature = database_signature()
        cached = await run_limited_query(compute_stats, request=request)
        result_cache.put("stats", cached, signature)
    return cached


def compute_stats(cancel_callback=None, connection_callback=None) -> dict[str, Any]:
    escaped = str(OBSERVATIONS_FILE).replace("'", "''")
    con = duckdb.connect()
    try:
        if connection_callback:
            connection_callback(con)
        # count(*) comes from the Parquet footer; max(eventDate) reads one
        # column of the whole file, which is why the result is cached.
        total, newest = con.execute(
            f"SELECT count(*), max(eventDate) FROM '{escaped}'"
        ).fetchone()
    finally:
        if connection_callback:
            connection_callback(None)
        con.close()
    return {
        "version": __version__,
        "total_observations": int(total or 0),
        "newest_date": newest.isoformat() if newest else None,
        "updated": dt.datetime.fromtimestamp(
            OBSERVATIONS_FILE.stat().st_mtime, dt.timezone.utc
        ).date().isoformat(),
        "radius_min_km": MAP_RADIUS_MIN_KM,
        "radius_max_km": MAP_RADIUS_MAX_KM,
        "cell_sizes": ABUNDANCE_CELL_SIZES,
    }


def observations_csv(observations: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    buffer.write("id,decimalLatitude,decimalLongitude,eventDate,taxonID\n")
    for row in observations:
        buffer.write(
            f"{row['id']},{row['decimalLatitude']},{row['decimalLongitude']},"
            f"{row['eventDate']},{row['taxonID']}\n"
        )
    return buffer.getvalue()


@app.get("/api/seasonal")
async def seasonal(
    request: Request,
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
    radius: float = Query(ge=MAP_RADIUS_MIN_KM, le=MAP_RADIUS_MAX_KM),
    date_from: str = Query(pattern=DATE_PATTERN),
    date_to: str = Query(pattern=DATE_PATTERN),
    taxon_id: list[int] = Query(default=[]),
    exclude_taxon_id: list[int] = Query(default=[]),
    format: Literal["json", "csv"] = "json",
) -> Response:
    """Observation counts by day, week and month of year within a radius."""
    require_database()
    validate_date_range(date_from, date_to)
    validate_taxon_list("taxon_id", taxon_id)
    validate_taxon_list("exclude_taxon_id", exclude_taxon_id)
    # Round to about a metre so the cache key and the query use the same values.
    lat, lon, radius = round(lat, 5), round(lon, 5), round(radius, 3)
    key = (
        "seasonal",
        lat,
        lon,
        radius,
        date_from,
        date_to,
        tuple(sorted(taxon_id)),
        tuple(sorted(exclude_taxon_id)),
    )
    result = result_cache.get(key)
    taxon_ids = exclude_ids = None
    if result is None:
        signature = database_signature()
        taxon_ids = await expand_taxa(taxon_id, request)
        exclude_ids = await expand_taxa(exclude_taxon_id, request)
        frame = await run_limited_query(
            run_local_date_counts,
            request=request,
            parquet_path=str(OBSERVATIONS_FILE),
            date_from=date_from,
            date_to=date_to,
            lat=lat,
            lon=lon,
            radius=radius,
            taxon_ids=taxon_ids,
            exclude_taxon_ids=exclude_ids,
        )
        dates, weights = frame["date"], frame["count"]
        result = {
            "total": int(weights.sum()),
            "days_with_observations": int(len(frame)),
            "first_date": dates.min().date().isoformat() if len(frame) else None,
            "last_date": dates.max().date().isoformat() if len(frame) else None,
            "taxa_included": len(taxon_ids),
            "taxa_excluded": len(exclude_ids),
            "bins": {
                view: seasonal_bin_counts(dates, view, weights)
                for view in ("daily", "weekly", "monthly")
            },
        }
        result_cache.put(key, result, signature)

    if format == "json":
        return JSONResponse(result)

    if result["total"] > MAX_CSV_ROWS:
        raise HTTPException(
            413,
            f"{result['total']:,} observations is more than the {MAX_CSV_ROWS:,} "
            "row CSV limit. Narrow the search to download the data.",
        )
    if taxon_ids is None or exclude_ids is None:
        taxon_ids = await expand_taxa(taxon_id, request)
        exclude_ids = await expand_taxa(exclude_taxon_id, request)
    observations, _count = await run_limited_query(
        run_local_observation_query,
        request=request,
        parquet_path=str(OBSERVATIONS_FILE),
        date_from=date_from,
        date_to=date_to,
        lat=lat,
        lon=lon,
        radius=radius,
        taxon_ids=taxon_ids,
        exclude_taxon_ids=exclude_ids,
        estimate_count=False,
        limit=MAX_CSV_ROWS + 1,
    )
    if len(observations) > MAX_CSV_ROWS:
        # The database may have changed since the cached total was computed.
        raise HTTPException(
            413,
            f"More than {MAX_CSV_ROWS:,} observations match, which is over the CSV "
            "limit. Narrow the search to download the data.",
        )
    body = await asyncio.to_thread(observations_csv, observations)
    return Response(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="observations.csv"'},
    )


@app.get("/api/abundance")
async def abundance(
    request: Request,
    taxon_id: list[int] = Query(min_length=1),
    exclude_taxon_id: list[int] = Query(default=[]),
    season_start: str = Query(pattern=MONTH_DAY_PATTERN),
    season_end: str = Query(pattern=MONTH_DAY_PATTERN),
    date_from: str = Query(pattern=DATE_PATTERN),
    date_to: str = Query(pattern=DATE_PATTERN),
    cell_deg: float = 1.0,
) -> JSONResponse:
    """Worldwide per-grid-cell counts of a taxon during a time of year."""
    require_database()
    validate_date_range(date_from, date_to)
    validate_taxon_list("taxon_id", taxon_id)
    validate_taxon_list("exclude_taxon_id", exclude_taxon_id)
    if cell_deg not in ABUNDANCE_CELL_SIZES:
        raise HTTPException(422, f"cell_deg must be one of {ABUNDANCE_CELL_SIZES}")
    key = (
        "abundance",
        tuple(sorted(taxon_id)),
        tuple(sorted(exclude_taxon_id)),
        season_start,
        season_end,
        date_from,
        date_to,
        cell_deg,
    )
    result = result_cache.get(key)
    if result is None:
        signature = database_signature()
        taxon_ids = await expand_taxa(taxon_id, request)
        exclude_ids = await expand_taxa(exclude_taxon_id, request)
        cells = await run_limited_query(
            run_local_abundance_query,
            request=request,
            parquet_path=str(OBSERVATIONS_FILE),
            date_from=date_from,
            date_to=date_to,
            season_start=parse_month_day(season_start),
            season_end=parse_month_day(season_end),
            cell_deg=cell_deg,
            taxon_ids=taxon_ids,
            exclude_taxon_ids=exclude_ids,
        )
        for cell in cells:
            cell["radius_km"] = abundance_cell_radius_km(cell)
            cell["centroid_lat"] = round(cell["centroid_lat"], 4)
            cell["centroid_lon"] = round(cell["centroid_lon"], 4)
            cell["share"] = round(cell["share"], 6)
        result = {
            "cell_deg": cell_deg,
            "total": sum(cell["count"] for cell in cells),
            "taxa_included": len(taxon_ids),
            "cells": cells,
        }
        result_cache.put(key, result, signature)
    return JSONResponse(result)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

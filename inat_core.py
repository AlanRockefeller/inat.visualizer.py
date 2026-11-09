"""Qt-free core queries shared by the desktop visualizer and the web app.

Everything here depends only on DuckDB, pandas and the standard library, so the
web server can import it without PyQt6 or matplotlib.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import duckdb
import pandas as pd

EARTH_RADIUS_KM = 6371.0
MAP_RADIUS_MIN_KM = 1.0
MAP_RADIUS_MAX_KM = 1000.0
REQUIRED_OBSERVATION_COLUMNS = (
    "eventdate",
    "decimallatitude",
    "decimallongitude",
    "taxonid",
)
# Queries that return individual rows also need the observation id.
REQUIRED_ROW_COLUMNS = REQUIRED_OBSERVATION_COLUMNS + ("id",)
# Longitude in [-180, 180): the data keeps +180 as a legal value, and a cell
# or box starting at +180 would otherwise be counted apart from -180.
NORMALIZED_LONGITUDE_SQL = (
    "(CASE WHEN decimalLongitude >= 180 THEN decimalLongitude - 360 "
    "ELSE decimalLongitude END)"
)

# Great-circle distance in km from (?, ?) = (lat, lon) placeholders, in the
# order: lat, lat, lon. DuckDB-native math so large scans never call Python.
HAVERSINE_KM_SQL = """2 * 6371 * asin(
                sqrt(
                    pow(sin(radians(decimalLatitude - ?) / 2), 2)
                    + cos(radians(?)) * cos(radians(decimalLatitude))
                    * pow(sin(radians(decimalLongitude - ?) / 2), 2)
                )
            )"""


def normalize_longitude(lon: float) -> float:
    """Return a longitude in the conventional [-180, 180) range."""
    return (float(lon) + 180.0) % 360.0 - 180.0


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


def calculate_local_search_bounds(
    lat: float, lon: float, radius: float
) -> tuple[float, float, float, float]:
    """Calculate a latitude/longitude bounding box around a radius in km.

    The longitude bounds are not wrapped: a search near the antimeridian
    returns values beyond +/-180 so callers can see that the box crosses it.
    ``longitude_filter_sql`` turns them into a predicate that handles the wrap.
    """
    lat_rad = math.radians(lat)
    lat_diff = radius / EARTH_RADIUS_KM
    lat_min_rad = lat_rad - lat_diff
    lat_max_rad = lat_rad + lat_diff
    if lat_min_rad > -math.pi / 2 and lat_max_rad < math.pi / 2:
        lon_diff = math.asin(min(1.0, math.sin(lat_diff) / math.cos(lat_rad)))
        lon_min = lon - math.degrees(lon_diff)
        lon_max = lon + math.degrees(lon_diff)
    else:
        # The circle contains a pole, so every longitude is inside it.
        lon_min, lon_max = -180.0, 180.0
    lat_min = max(-90.0, math.degrees(lat_min_rad))
    lat_max = min(90.0, math.degrees(lat_max_rad))
    return lat_min, lat_max, lon_min, lon_max


def longitude_filter_sql(lon_min: float, lon_max: float) -> tuple[str, list[float]]:
    """Return a SQL predicate and parameters for a longitude range.

    A range that crosses the antimeridian (for example 170 to 190, or -190 to
    -170) becomes ``(lon >= west OR lon <= east)`` once both edges are wrapped
    into [-180, 180). A range at least 360 degrees wide matches everything.
    """
    if lon_max - lon_min >= 360.0:
        return "TRUE", []
    west = normalize_longitude(lon_min)
    east = normalize_longitude(lon_max)
    if west <= east:
        return "decimalLongitude BETWEEN ? AND ?", [west, east]
    return "(decimalLongitude >= ? OR decimalLongitude <= ?)", [west, east]


def radius_where_sql(
    date_from: str, date_to: str, lat: float, lon: float, radius: float
) -> tuple[str, list[Any]]:
    """Return the WHERE clause and parameters shared by the radius searches.

    The date and bounding-box predicates let DuckDB skip Parquet row groups
    from their min/max statistics. Callers append the exact great-circle test.
    eventDate is cast to DATE so a timestamp column still includes the whole
    final day; on a DATE column the cast is a no-op.
    """
    lat_min, lat_max, lon_min, lon_max = calculate_local_search_bounds(lat, lon, radius)
    lon_sql, lon_params = longitude_filter_sql(lon_min, lon_max)
    where = f"""
            WHERE CAST(eventDate AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            AND decimalLatitude BETWEEN ? AND ?
            AND {lon_sql}
    """
    return where, [date_from, date_to, lat_min, lat_max, *lon_params]


def leap_aligned_day_of_year(dates: pd.Series) -> pd.Series:
    """Number days 1-366 as in a leap year, so each calendar date has one bin.

    Plain day-of-year puts March 1 at day 60 in common years but day 61 in
    leap years, smearing every later date across two bins. Shifting common-year
    dates after February by one gives Feb 29 its own bin (60) and keeps every
    other calendar date in a single bin whatever the year.
    """
    shift = ((~dates.dt.is_leap_year) & (dates.dt.month > 2)).astype("int64")
    return dates.dt.day_of_year.astype("int64") + shift


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


class SearchCancelled(Exception):
    """Raised inside a worker when the user cancels a search."""


def raise_if_search_cancelled(cancel_callback=None) -> None:
    if cancel_callback and cancel_callback():
        raise SearchCancelled


def check_observation_columns(
    con, escaped_path: str, required: tuple[str, ...] = REQUIRED_OBSERVATION_COLUMNS
) -> None:
    """Raise ValueError if observations.parquet lacks a column the query uses."""
    schema = con.execute(f"DESCRIBE SELECT * FROM '{escaped_path}'").fetchall()
    column_names = [row[0].lower() for row in schema]
    missing_columns = [col for col in required if col not in column_names]
    if missing_columns:
        raise ValueError(
            "Missing required columns in observations.parquet: "
            + ", ".join(missing_columns)
            + ". The file must include eventDate, decimalLatitude, "
            "decimalLongitude, taxonID"
            + (", and id." if "id" in required else ".")
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
    estimate_count: bool = True,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Query observations.parquet for local observations.

    Returns the matching rows and a count. With ``estimate_count`` the count is
    a preliminary bounding-box count used for progress messages, which costs a
    second scan of the file; without it the count is simply the number of rows.
    ``limit`` caps the number of rows fetched.
    """
    taxon_ids = taxon_ids or []
    exclude_taxon_ids = exclude_taxon_ids or []
    escaped_path = parquet_path.replace("'", "''")
    base_where, where_params = radius_where_sql(date_from, date_to, lat, lon, radius)
    con = None
    try:
        raise_if_search_cancelled(cancel_callback)
        con = duckdb.connect()
        if connection_callback:
            connection_callback(con)
        raise_if_search_cancelled(cancel_callback)
        if progress_callback:
            progress_callback("Checking observations.parquet schema...")
        check_observation_columns(con, escaped_path, REQUIRED_ROW_COLUMNS)
        raise_if_search_cancelled(cancel_callback)

        filter_sql = taxon_filter_sql(con, taxon_ids, exclude_taxon_ids)

        estimated_count = None
        if estimate_count:
            if progress_callback:
                progress_callback("Estimating result count...")
            count_query = (
                f"SELECT COUNT(*) FROM '{escaped_path}' {base_where} {filter_sql}"
            )
            _count_row = con.execute(count_query, where_params).fetchone()
            raise_if_search_cancelled(cancel_callback)
            estimated_count = _count_row[0] if _count_row else 0
            if progress_callback:
                progress_callback(
                    f"Executing DuckDB query (estimated {estimated_count:,} candidates)..."
                )
        elif progress_callback:
            progress_callback("Executing DuckDB query...")

        # Use DuckDB-native math instead of a Python UDF so large scans do not
        # call back into Python once per row.
        query = f"""
            SELECT id, decimalLatitude, decimalLongitude, eventDate, taxonID
            FROM '{escaped_path}'
            {base_where}
            {filter_sql}
            AND {HAVERSINE_KM_SQL} <= ?
        """
        query_params = where_params + [lat, lat, lon, radius]
        if limit is not None:
            query += " LIMIT ?"
            query_params.append(int(limit))
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
        if estimated_count is None:
            estimated_count = len(observations)
        return observations, estimated_count
    finally:
        if connection_callback:
            connection_callback(None)
        if con is not None:
            con.close()



def run_local_date_counts(
    parquet_path: str,
    date_from: str,
    date_to: str,
    lat: float,
    lon: float,
    radius: float,
    taxon_ids: list[int] | None = None,
    exclude_taxon_ids: list[int] | None = None,
    cancel_callback=None,
    connection_callback=None,
) -> pd.DataFrame:
    """Count observations per date inside a radius, aggregated in DuckDB.

    Same filters as ``run_local_observation_query``, but returns at most one
    row per calendar date (columns ``date`` and ``count``) instead of one per
    observation, so huge searches stay small in memory.
    """
    escaped_path = parquet_path.replace("'", "''")
    base_where, where_params = radius_where_sql(date_from, date_to, lat, lon, radius)
    con = None
    try:
        raise_if_search_cancelled(cancel_callback)
        con = duckdb.connect()
        if connection_callback:
            connection_callback(con)
        check_observation_columns(con, escaped_path)
        filter_sql = taxon_filter_sql(con, taxon_ids or [], exclude_taxon_ids or [])
        raise_if_search_cancelled(cancel_callback)
        query = f"""
            SELECT CAST(eventDate AS DATE) AS date, count(*) AS count
            FROM '{escaped_path}'
            {base_where}
            {filter_sql}
            AND {HAVERSINE_KM_SQL} <= ?
            GROUP BY 1
            ORDER BY 1
        """
        params = where_params + [lat, lat, lon, radius]
        frame = con.execute(query, params).fetch_df()
        raise_if_search_cancelled(cancel_callback)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["count"] = frame["count"].astype("int64")
        return frame
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
        lon_sql = NORMALIZED_LONGITUDE_SQL
        query = f"""
            SELECT
                CAST(floor(CAST(decimalLatitude AS DOUBLE) / ?) AS BIGINT) AS lat_index,
                CAST(floor(CAST({lon_sql} AS DOUBLE) / ?) AS BIGINT) AS lon_index,
                count(*) FILTER (WHERE {target_sql}) AS target_count,
                count(*) AS total_count,
                avg(CAST(decimalLatitude AS DOUBLE)) FILTER (WHERE {target_sql})
                    AS centroid_lat,
                avg(CAST({lon_sql} AS DOUBLE)) FILTER (WHERE {target_sql})
                    AS centroid_lon
            FROM '{escaped_path}'
            WHERE CAST(eventDate AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            AND {season_sql}
            AND decimalLatitude IS NOT NULL
            AND decimalLongitude IS NOT NULL
            GROUP BY lat_index, lon_index
            HAVING target_count > 0
        """
        params = [cell_deg, cell_deg, date_from, date_to, *season_params]
        rows = con.execute(query, params).fetchall()
        raise_if_search_cancelled(cancel_callback)

        if progress_callback:
            progress_callback("Processing results...")
        cells = []
        for index, (
            lat_index,
            lon_index,
            target_count,
            total_count,
            centroid_lat,
            centroid_lon,
        ) in enumerate(rows):
            if index % 1000 == 0:
                raise_if_search_cancelled(cancel_callback)
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
        raise_if_search_cancelled(cancel_callback)
        cells.sort(key=lambda cell: cell["count"], reverse=True)
        return cells
    finally:
        if connection_callback:
            connection_callback(None)
        if con is not None:
            con.close()


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


def descendant_taxon_ids(
    taxonomy_path: str,
    taxon_id: int,
    cancel_callback=None,
    connection_callback=None,
) -> list[int]:
    """Return ``taxon_id`` and every descendant ID listed in taxonomy.parquet.

    The recursion is seeded with the requested ID itself rather than its
    taxonomy row, so children are still found when the root row is missing.
    ``UNION`` rather than ``UNION ALL`` feeds only new IDs into the next
    round, so a parent cycle in a damaged taxonomy ends the recursion instead
    of running forever. The callbacks let the web server apply its connection
    settings and interrupt the query like any other.
    """
    escaped_path = str(taxonomy_path).replace("'", "''")
    con = duckdb.connect()
    try:
        if connection_callback:
            connection_callback(con)
        raise_if_search_cancelled(cancel_callback)
        con.execute(f"CREATE VIEW taxonomy AS SELECT * FROM '{escaped_path}'")
        results = con.execute(
            """
            WITH RECURSIVE descendants AS (
                SELECT CAST(? AS BIGINT) AS id
                UNION
                SELECT t.id FROM taxonomy t
                JOIN descendants d ON t.parent_id = d.id
            )
            SELECT id FROM descendants
            """,
            [taxon_id],
        ).fetchall()
        raise_if_search_cancelled(cancel_callback)
    finally:
        if connection_callback:
            connection_callback(None)
        con.close()
    return [int(row[0]) for row in results]


SEASONAL_BIN_COUNTS = {"daily": 366, "weekly": 52, "monthly": 12}


def seasonal_bin_counts(
    dates: pd.Series, view: str, weights: pd.Series | None = None
) -> list[int]:
    """Count dates per calendar day, calendar week, or month, as the graph bins them.

    Daily bins follow a leap-year calendar (see ``leap_aligned_day_of_year``)
    so that bin 61 is always March 1. ``weights`` gives a count per date, for
    pre-aggregated per-date rows.
    """
    if view not in SEASONAL_BIN_COUNTS:
        raise ValueError(f"Unknown view: {view}")
    bins = SEASONAL_BIN_COUNTS[view]
    counts = [0] * bins
    if dates.empty:
        return counts
    if view == "daily":
        groups = leap_aligned_day_of_year(dates)
    elif view == "weekly":
        groups = calendar_aligned_week_numbers(dates)
    else:
        groups = dates.dt.month
    if weights is None:
        weights = pd.Series(1, index=dates.index)
    totals = weights.groupby(groups.to_numpy()).sum()
    for group, count in totals.items():
        index = int(group) - 1
        if 0 <= index < bins:
            counts[index] = int(count)
    return counts

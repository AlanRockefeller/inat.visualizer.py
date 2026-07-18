#!/usr/bin/env python3
"""Rebuild the visualizer's local Parquet databases from iNaturalist DWCA files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import duckdb


OBSERVATIONS_URL = (
    "https://static.inaturalist.org/observations/gbif-observations-dwca.zip"
)
TAXONOMY_URL = "https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip"
OBSERVATIONS_ARCHIVE_NAME = "gbif-observations-dwca.zip"
TAXONOMY_ARCHIVE_NAME = "inaturalist-taxonomy.dwca.zip"
EXTRACTION_RESERVE_BYTES = 5 * 1024**3
COPY_BUFFER_BYTES = 16 * 1024**2


class UpdateError(RuntimeError):
    """Raised when an input cannot safely produce a replacement database."""


def status(message: str) -> None:
    """Print an immediately visible status message."""
    print(message, flush=True)


def human_size(size: int | None) -> str:
    """Return a compact binary size for progress messages."""
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def sql_literal(value: str | Path) -> str:
    """Quote a string for a DuckDB SQL statement."""
    return "'" + str(value).replace("'", "''") + "'"


def _content_range_total(header: str | None) -> int | None:
    if not header or "/" not in header:
        return None
    total = header.rsplit("/", 1)[1]
    return int(total) if total.isdigit() else None


def download(url: str, destination: Path) -> None:
    """Download *url* with a resumable .part file, then rename it atomically."""
    part = destination.with_name(destination.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "iNat-Seasonal-Visualizer-database-updater/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        status(f"Resuming {destination.name} at {human_size(existing)}...")
    else:
        status(f"Downloading {destination.name}...")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        total = _content_range_total(error.headers.get("Content-Range"))
        if error.code == 416 and total == existing:
            os.replace(part, destination)
            return
        raise UpdateError(f"Download failed for {url}: {error}") from error
    except OSError as error:
        raise UpdateError(f"Download failed for {url}: {error}") from error

    with response:
        resumed = response.status == 206
        if resumed:
            mode = "ab"
            downloaded = existing
            total = _content_range_total(response.headers.get("Content-Range"))
        else:
            mode = "wb"
            downloaded = 0
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None

        last_report = 0.0
        with part.open(mode) as output:
            while chunk := response.read(COPY_BUFFER_BYTES):
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2:
                    if total:
                        percent = 100 * downloaded / total
                        status(
                            f"  {destination.name}: {percent:5.1f}% "
                            f"({human_size(downloaded)} / {human_size(total)})"
                        )
                    else:
                        status(f"  {destination.name}: {human_size(downloaded)}")
                    last_report = now

    os.replace(part, destination)
    status(f"Downloaded {destination.name} ({human_size(destination.stat().st_size)}).")


def ensure_archive(path: Path, url: str) -> None:
    """Download an archive only when it is not already available locally."""
    if path.exists():
        status(f"Using existing archive {path} ({human_size(path.stat().st_size)}).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    download(url, path)


def find_member(archive: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    """Find one archive member by basename without trusting its directory path."""
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name == basename
    ]
    if len(matches) != 1:
        raise UpdateError(
            f"Expected exactly one {basename!r} in {archive.filename}, "
            f"found {len(matches)}."
        )
    return matches[0]


def check_extraction_space(destination: Path, member: zipfile.ZipInfo) -> None:
    """Fail before extraction if the raw member would nearly fill the filesystem."""
    free = shutil.disk_usage(destination).free
    required = member.file_size + EXTRACTION_RESERVE_BYTES
    if free < required:
        raise UpdateError(
            f"Not enough free space to extract {member.filename}: "
            f"need at least {human_size(required)}, have {human_size(free)}."
        )


def extract_member(
    archive_path: Path,
    basename: str,
    destination: Path,
    report: Callable[[str], None] = status,
) -> None:
    """Extract one required member and report progress; CRC is checked by zipfile."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise UpdateError(
            f"{archive_path} is not a complete ZIP archive. If it is still "
            "downloading, wait for that download to finish and run this again."
        ) from error

    with archive:
        member = find_member(archive, basename)
        check_extraction_space(destination.parent, member)
        report(
            f"Extracting {basename} ({human_size(member.file_size)}) from "
            f"{archive_path.name}..."
        )
        copied = 0
        last_report = 0.0
        try:
            with archive.open(member) as source, destination.open("wb") as output:
                while chunk := source.read(COPY_BUFFER_BYTES):
                    output.write(chunk)
                    copied += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 2:
                        percent = 100 * copied / member.file_size
                        report(
                            f"  {basename}: {percent:5.1f}% "
                            f"({human_size(copied)} / {human_size(member.file_size)})"
                        )
                        last_report = now
        except (OSError, EOFError, zipfile.BadZipFile) as error:
            raise UpdateError(f"Could not extract {basename}: {error}") from error
    report(f"Extracted {basename}.")


def require_csv_columns(csv_path: Path, required: set[str]) -> None:
    """Check the DWCA header before beginning an expensive DuckDB conversion."""
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            columns = next(csv.reader(source))
    except (OSError, UnicodeError, StopIteration, csv.Error) as error:
        raise UpdateError(f"Could not read the header of {csv_path}: {error}") from error
    missing = sorted(required.difference(columns))
    if missing:
        raise UpdateError(
            f"{csv_path.name} is missing required columns: {', '.join(missing)}"
        )


def configure_duckdb(temp_directory: Path) -> duckdb.DuckDBPyConnection:
    """Create a conversion connection that may spill beside the staged outputs."""
    duckdb_temp = temp_directory / "duckdb-temp"
    duckdb_temp.mkdir()
    connection = duckdb.connect(
        database=":memory:", config={"temp_directory": str(duckdb_temp)}
    )
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("PRAGMA enable_progress_bar")
    connection.execute("PRAGMA progress_bar_time = 2000")
    return connection


def build_observations_parquet(
    connection: duckdb.DuckDBPyConnection, csv_path: Path, output_path: Path
) -> None:
    """Keep only the fields used by visualizer.py and reject unusable records."""
    require_csv_columns(
        csv_path,
        {"id", "decimalLatitude", "decimalLongitude", "taxonID", "taxonRank", "eventDate"},
    )
    source = sql_literal(csv_path)
    output = sql_literal(output_path)
    status("Building observations.parquet with DuckDB (this is the long step)...")
    connection.execute(
        f"""
        COPY (
            WITH parsed AS (
                SELECT
                    TRY_CAST(id AS BIGINT) AS id,
                    TRY_CAST(
                        ROUND(TRY_CAST(decimalLatitude AS DOUBLE), 3)
                        AS DECIMAL(8, 3)
                    ) AS decimalLatitude,
                    TRY_CAST(
                        ROUND(TRY_CAST(decimalLongitude AS DOUBLE), 3)
                        AS DECIMAL(8, 3)
                    ) AS decimalLongitude,
                    TRY_CAST(taxonID AS BIGINT) AS taxonID,
                    taxonRank,
                    TRY_CAST(eventDate AS DATE) AS eventDate
                FROM read_csv(
                    {source},
                    header = true,
                    all_varchar = true,
                    delim = ',',
                    quote = '"',
                    escape = '"',
                    parallel = true
                )
            )
            SELECT *
            FROM parsed
            WHERE id IS NOT NULL
              AND decimalLatitude BETWEEN -90 AND 90
              AND decimalLongitude BETWEEN -180 AND 180
              AND taxonID IS NOT NULL
              AND eventDate IS NOT NULL
        ) TO {output} (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 1000000
        )
        """
    )


def build_taxonomy_parquet(
    connection: duckdb.DuckDBPyConnection, csv_path: Path, output_path: Path
) -> None:
    """Build the id/parent_id hierarchy consumed by the recursive app query."""
    require_csv_columns(csv_path, {"taxonID", "parentNameUsageID"})
    source = sql_literal(csv_path)
    output = sql_literal(output_path)
    status("Building taxonomy.parquet...")
    connection.execute(
        f"""
        COPY (
            WITH parsed AS (
                SELECT
                    TRY_CAST(
                        REGEXP_EXTRACT(taxonID, '([0-9]+)/*$', 1) AS INTEGER
                    ) AS id,
                    TRY_CAST(
                        REGEXP_EXTRACT(parentNameUsageID, '([0-9]+)/*$', 1)
                        AS INTEGER
                    ) AS parent_id
                FROM read_csv(
                    {source},
                    header = true,
                    all_varchar = true,
                    delim = ',',
                    quote = '"',
                    escape = '"',
                    parallel = true
                )
            )
            SELECT id, parent_id
            FROM parsed
            WHERE id IS NOT NULL
        ) TO {output} (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 1000000
        )
        """
    )


def parquet_schema(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> list[tuple[str, str]]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(path)})"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def validate_observations(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> dict[str, object]:
    expected = [
        ("id", "BIGINT"),
        ("decimalLatitude", "DECIMAL(8,3)"),
        ("decimalLongitude", "DECIMAL(8,3)"),
        ("taxonID", "BIGINT"),
        ("taxonRank", "VARCHAR"),
        ("eventDate", "DATE"),
    ]
    schema = parquet_schema(connection, path)
    normalized = [(name, data_type.replace(" ", "")) for name, data_type in schema]
    if normalized != expected:
        raise UpdateError(f"Unexpected observations.parquet schema: {schema}")
    row = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT taxonID), MIN(eventDate), MAX(eventDate),
               COUNT(*) FILTER (
                   WHERE id IS NULL OR decimalLatitude IS NULL
                      OR decimalLongitude IS NULL OR taxonID IS NULL
                      OR eventDate IS NULL
               )
        FROM read_parquet({sql_literal(path)})
        """
    ).fetchone()
    if not row or row[0] == 0 or row[4] != 0:
        raise UpdateError(f"Observation validation failed: {row}")
    return {
        "rows": row[0],
        "taxa": row[1],
        "first_date": row[2],
        "last_date": row[3],
    }


def validate_taxonomy(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> dict[str, int]:
    expected = [("id", "INTEGER"), ("parent_id", "INTEGER")]
    schema = parquet_schema(connection, path)
    if schema != expected:
        raise UpdateError(f"Unexpected taxonomy.parquet schema: {schema}")
    row = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT id),
               COUNT(*) FILTER (WHERE id = parent_id)
        FROM read_parquet({sql_literal(path)})
        """
    ).fetchone()
    if not row or row[0] == 0 or row[0] != row[1] or row[2] != 0:
        raise UpdateError(f"Taxonomy validation failed: {row}")
    return {"rows": row[0]}


def stage_sanitized_cache(cache_path: Path, staging_directory: Path) -> Path | None:
    """Remove descendant lists that were calculated from the old taxonomy."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as source:
            cache = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        status(f"Warning: leaving unreadable taxon cache unchanged: {error}")
        return None
    if not isinstance(cache, dict):
        status("Warning: leaving non-object taxon cache unchanged.")
        return None
    stale_keys = [key for key in cache if key.endswith("_descendants")]
    if not stale_keys:
        return None
    for key in stale_keys:
        del cache[key]
    staged = staging_directory / "taxon_cache.json.new"
    with staged.open("w", encoding="utf-8") as output:
        json.dump(cache, output, indent=2)
        output.write("\n")
    status(f"Prepared taxon cache with {len(stale_keys):,} stale descendant entries removed.")
    return staged


def update_databases(
    observation_archive: Path,
    taxonomy_archive: Path,
    output_directory: Path,
    *,
    keep_archives: bool = False,
) -> tuple[dict[str, object], dict[str, int]]:
    """Build, validate, install, and clean up both application databases."""
    output_directory.mkdir(parents=True, exist_ok=True)
    observation_archive = observation_archive.resolve()
    taxonomy_archive = taxonomy_archive.resolve()
    output_directory = output_directory.resolve()

    with tempfile.TemporaryDirectory(
        prefix=".inat-db-update-", dir=output_directory
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_observations = temporary / "observations.csv"
        raw_taxa = temporary / "taxa.csv"
        staged_observations = temporary / "observations.parquet.new"
        staged_taxonomy = temporary / "taxonomy.parquet.new"

        connection = configure_duckdb(temporary)
        try:
            extract_member(observation_archive, "observations.csv", raw_observations)
            build_observations_parquet(connection, raw_observations, staged_observations)
            raw_observations.unlink()
            observation_stats = validate_observations(connection, staged_observations)
            status(
                "Validated observations: "
                f"{observation_stats['rows']:,} rows, "
                f"{observation_stats['taxa']:,} taxa, "
                f"{observation_stats['first_date']} through "
                f"{observation_stats['last_date']}."
            )

            extract_member(taxonomy_archive, "taxa.csv", raw_taxa)
            build_taxonomy_parquet(connection, raw_taxa, staged_taxonomy)
            raw_taxa.unlink()
            taxonomy_stats = validate_taxonomy(connection, staged_taxonomy)
            status(f"Validated taxonomy: {taxonomy_stats['rows']:,} taxa.")
        finally:
            connection.close()

        staged_cache = stage_sanitized_cache(
            output_directory / "taxon_cache.json", temporary
        )

        # Each replace is atomic. Clear old descendant results first so even an
        # unlikely later replacement failure cannot pair a new hierarchy with
        # cached results calculated from the old one.
        if staged_cache is not None:
            os.replace(staged_cache, output_directory / "taxon_cache.json")
        os.replace(staged_taxonomy, output_directory / "taxonomy.parquet")
        os.replace(staged_observations, output_directory / "observations.parquet")

    if not keep_archives:
        for archive in {observation_archive, taxonomy_archive}:
            try:
                archive.unlink()
                status(f"Removed source archive {archive}.")
            except FileNotFoundError:
                pass

    status("Database update complete; only the Parquet databases are retained.")
    return observation_stats, taxonomy_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/reuse iNaturalist DWCA files, rebuild observations.parquet "
            "and taxonomy.parquet, and remove the raw inputs after success."
        )
    )
    parser.add_argument(
        "--observations-archive",
        type=Path,
        help=f"existing observation DWCA (default: ./{OBSERVATIONS_ARCHIVE_NAME})",
    )
    parser.add_argument(
        "--taxonomy-archive",
        type=Path,
        help=f"existing taxonomy DWCA (default: ./{TAXONOMY_ARCHIVE_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing the app's Parquet files (default: script directory)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep the two DWCA ZIP files after a successful update",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory = args.output_dir.resolve()
    observation_archive = (
        args.observations_archive.resolve()
        if args.observations_archive
        else output_directory / OBSERVATIONS_ARCHIVE_NAME
    )
    taxonomy_archive = (
        args.taxonomy_archive.resolve()
        if args.taxonomy_archive
        else output_directory / TAXONOMY_ARCHIVE_NAME
    )

    try:
        ensure_archive(observation_archive, OBSERVATIONS_URL)
        ensure_archive(taxonomy_archive, TAXONOMY_URL)
        update_databases(
            observation_archive,
            taxonomy_archive,
            output_directory,
            keep_archives=args.keep_archives,
        )
    except (UpdateError, duckdb.Error, OSError) as error:
        print(f"Database update failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

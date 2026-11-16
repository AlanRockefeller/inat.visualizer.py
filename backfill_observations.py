#!/usr/bin/env python3
"""Fetch the research-grade iNaturalist observations missing from the GBIF export.

iNaturalist's GBIF export (what ``update_database.py`` imports) only carries
observations whose observer chose CC0, CC-BY or CC-BY-NC. This tool pages
through the API for the rest -- unlicensed ("all rights reserved") and the
CC-BY-SA / -ND families -- keeping only the columns the visualizer counts with,
so the seasonal graphs and maps can include every research-grade observation.

Everything is kept in a store directory (default ``./backfill``):

    state.json        cursors and timestamps, saved after every flush
    parts/*.parquet   fetched pages, appended in order (later rows win)
    extra.parquet     compacted store: latest row per id, removals dropped

The first run walks every matching observation in id order, about 51 million
rows at 200 per request, at most one request a second and 10,000 a day (iNat's
API guidance) -- roughly four weeks. It can be stopped and restarted at any
time. Once both walks are finished, each run instead asks only for
observations updated since the previous sync, whatever their license, and
removes those that are no longer research grade or that moved to a license the
export carries (the export then counts them). Observations deleted on
iNaturalist are not seen: the API only lists deletions to their owner.

    python backfill_observations.py                 # run until today's budget is spent
    python backfill_observations.py --max-hours 2   # or stop after two hours
    python backfill_observations.py --status

``update_database.py`` merges ``extra.parquet`` into observations.parquet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

API_V2 = "https://api.inaturalist.org/v2"
USER_AGENT = "iNat-Seasonal-Visualizer-backfill/1 (+https://visualizer.dikarya.us/)"
PER_PAGE = 200
FIELDS = "id,observed_on,location,quality_grade,license_code,taxon.id,taxon.rank"
# Licenses the GBIF export carries; observations under these belong to it.
EXPORT_LICENSES = {"cc0", "cc-by", "cc-by-nc"}
# The two groups missing from the export, as API filters.
STREAMS = {
    "unlicensed": {"licensed": "false"},
    "sharealike_noderivs": {"license": "cc-by-sa,cc-by-nd,cc-by-nc-sa,cc-by-nc-nd"},
}
# The sync asks for every updated observation, whatever its license, so a
# stored observation that changed license still reaches the store.
SYNC_STREAMS = {"changes": {}}
DEFAULT_DAILY_REQUESTS = 10_000
MIN_REQUEST_INTERVAL = 1.0
FLUSH_EVERY_PAGES = 25
COLUMNS = (
    "id BIGINT, decimalLatitude DECIMAL(8, 3), decimalLongitude DECIMAL(8, 3), "
    "taxonID BIGINT, taxonRank VARCHAR, eventDate DATE, removed BOOLEAN, seq BIGINT"
)


class BackfillError(Exception):
    """The API refused or kept failing; the store is left consistent."""


class BudgetSpent(BackfillError):
    """Today's request budget ran out; run again tomorrow to continue."""


def sql_literal(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def status(message: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class ApiClient:
    """Rate-limited JSON GETs with retries, and a per-day request budget."""

    def __init__(
        self,
        state: dict[str, Any],
        daily_limit: int = DEFAULT_DAILY_REQUESTS,
        min_interval: float = MIN_REQUEST_INTERVAL,
        opener: Callable[[urllib.request.Request], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state = state
        self.daily_limit = daily_limit
        self.min_interval = min_interval
        self.opener = opener or (lambda request: urllib.request.urlopen(request, timeout=120))
        self.sleep = sleep
        self.clock = clock
        self.last_request = -float("inf")

    def budget_left(self) -> int:
        today = dt.date.today().isoformat()
        if self.state.get("budget_day") != today:
            self.state["budget_day"] = today
            self.state["requests_today"] = 0
        return self.daily_limit - self.state["requests_today"]

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        delay = 5.0
        for attempt in range(8):
            # Retries count against the budget too.
            if self.budget_left() <= 0:
                raise BudgetSpent("today's request budget is spent")
            wait = self.min_interval - (self.clock() - self.last_request)
            if wait > 0:
                self.sleep(wait)
            self.last_request = self.clock()
            self.state["requests_today"] += 1
            try:
                with self.opener(urllib.request.Request(full, headers=headers)) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                if error.code in (429, 500, 502, 503, 504):
                    retry_after = error.headers.get("Retry-After") if error.headers else None
                    pause = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    status(f"API answered {error.code}; waiting {pause:.0f}s (attempt {attempt + 1}).")
                    self.sleep(pause)
                    delay = min(delay * 2, 600)
                    continue
                raise BackfillError(f"API request failed ({error.code}): {full}") from None
            except (OSError, ValueError) as error:
                status(f"API request failed ({error}); waiting {delay:.0f}s.")
                self.sleep(delay)
                delay = min(delay * 2, 600)
        raise BackfillError(f"API kept failing: {full}")


def parse_observation(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one API result into a store row, or None when it cannot be counted.

    Rows that are not research grade become removals, so a later update can
    take back an observation that lost its research-grade status. So do rows
    now under a license the export carries: the export counts those itself.
    """
    observation_id = raw.get("id")
    if not isinstance(observation_id, int):
        return None
    if raw.get("quality_grade") != "research":
        return {"id": observation_id, "removed": True}
    if str(raw.get("license_code") or "").lower() in EXPORT_LICENSES:
        return {"id": observation_id, "removed": True}
    try:
        lat_text, lon_text = (raw.get("location") or "").split(",")
        lat, lon = round(float(lat_text), 3), round(float(lon_text), 3)
        observed = dt.date.fromisoformat(raw.get("observed_on") or "")
    except ValueError:
        return {"id": observation_id, "removed": True}
    taxon = raw.get("taxon") or {}
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not isinstance(taxon.get("id"), int):
        return {"id": observation_id, "removed": True}
    return {
        "id": observation_id,
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "taxonID": taxon["id"],
        "taxonRank": taxon.get("rank"),
        "eventDate": observed,
        "removed": False,
    }


class Store:
    """The backfill directory: state, appended parts and the compacted file."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.parts = directory / "parts"
        self.state_path = directory / "state.json"
        self.extra_path = directory / "extra.parquet"
        self.parts.mkdir(parents=True, exist_ok=True)
        # One process at a time: a compaction deletes the parts it folded in,
        # and a fetcher must not write a part (or move a cursor) meanwhile.
        self.lock = open(directory / ".lock", "w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise BackfillError(f"{directory} is in use by another backfill or update") from None
        self.state = self.load_state()
        # A crash between writing a part and saving the state leaves rows on
        # disk with higher sequence numbers than the state; never reuse them.
        self.state["seq"] = max(self.state["seq"], self.committed_seq())
        self.pending: list[dict[str, Any]] = []

    def close(self) -> None:
        self.lock.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def sources(self) -> list[str]:
        sources = [str(p) for p in sorted(self.parts.glob("*.parquet"))]
        if self.extra_path.exists():
            sources.insert(0, str(self.extra_path))
        return sources

    def committed_seq(self) -> int:
        sources = self.sources()
        if not sources:
            return 0
        con = duckdb.connect()
        try:
            return int(con.execute("SELECT max(seq) FROM read_parquet(?)", [sources]).fetchone()[0] or 0)
        finally:
            con.close()

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "version": 1,
            "seq": 0,
            "started_at": None,
            "streams": {name: {"id_above": 0, "done": False} for name in STREAMS},
            "last_sync": None,
            "sync": None,
        }

    def save_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        os.replace(temporary, self.state_path)

    def add(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.state["seq"] += 1
            self.pending.append({**row, "seq": self.state["seq"]})

    def flush(self) -> None:
        """Write pending rows as a new part, then save the state that covers them.

        The part is complete on disk before the cursor moves past it, so a crash
        can at worst refetch a page (later rows win), never lose one.
        """
        if self.pending:
            names = [int(p.stem) for p in self.parts.glob("*.parquet")]
            target = self.parts / f"{max(names, default=0) + 1:08d}.parquet"
            temporary = target.with_suffix(".tmp")
            con = duckdb.connect()
            try:
                con.execute(f"CREATE TABLE rows ({COLUMNS})")
                con.executemany(
                    "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            r["id"], r.get("decimalLatitude"), r.get("decimalLongitude"),
                            r.get("taxonID"), r.get("taxonRank"), r.get("eventDate"),
                            r["removed"], r["seq"],
                        )
                        for r in self.pending
                    ],
                )
                con.execute(f"COPY rows TO {sql_literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            finally:
                con.close()
            os.replace(temporary, target)
            self.pending = []
        self.save_state()

    def compact(self) -> int:
        """Fold the parts into extra.parquet (latest row per id, removals dropped)."""
        sources = self.sources()
        if not sources:
            return 0
        temporary = self.extra_path.with_suffix(".parquet.tmp")
        con = duckdb.connect()
        try:
            con.execute(
                f"""
                COPY (
                    SELECT * FROM (
                        SELECT id, decimalLatitude, decimalLongitude, taxonID, taxonRank,
                               eventDate, removed, seq
                        FROM read_parquet(?)
                        QUALIFY row_number() OVER (PARTITION BY id ORDER BY seq DESC) = 1
                    ) WHERE NOT removed
                ) TO {sql_literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                [sources],
            )
            # A removal has cancelled every earlier row once folded in, and
            # later parts carry higher sequence numbers, so none are kept.
            rows = con.execute(
                f"SELECT count(*) FROM read_parquet({sql_literal(temporary)})"
            ).fetchone()[0]
        finally:
            con.close()
        os.replace(temporary, self.extra_path)
        for part in sources:
            if part != str(self.extra_path):
                Path(part).unlink()
        return rows


def fetch_stream(
    client: ApiClient,
    store: Store,
    name: str,
    cursor: dict[str, Any],
    extra_params: dict[str, Any],
    should_stop: Callable[[], bool],
) -> bool:
    """Page through one filter in id order from ``cursor``; True when finished."""
    pages = 0
    while not should_stop():
        params = {
            **(STREAMS | SYNC_STREAMS)[name],
            **extra_params,
            "order_by": "id",
            "order": "asc",
            "id_above": cursor["id_above"],
            "per_page": PER_PAGE,
            "fields": FIELDS,
        }
        try:
            body = client.get(f"{API_V2}/observations", params)
        except BudgetSpent:
            break  # the cursor still points at the page that was not fetched
        results = body.get("results") or []
        rows = [row for row in map(parse_observation, results) if row]
        store.add(rows)
        if results:
            cursor["id_above"] = max(r["id"] for r in results if isinstance(r.get("id"), int))
        cursor["remaining"] = max(0, int(body.get("total_results") or 0) - len(results))
        pages += 1
        if len(results) < PER_PAGE:
            store.flush()
            return True
        if pages % FLUSH_EVERY_PAGES == 0:
            store.flush()
            status(f"{name}: up to id {cursor['id_above']:,}, about {cursor['remaining']:,} to go.")
    store.flush()
    return False


def run(
    store: Store,
    client: ApiClient,
    *,
    deadline: float | None = None,
) -> str:
    """Advance the backfill or sync as far as the budget and deadline allow.

    Returns "backfilling", "synced" or "budget" (stopped early, resume later).
    """
    state = store.state
    stopping = {"flag": False}

    def should_stop() -> bool:
        return stopping["flag"] or (deadline is not None and time.monotonic() >= deadline)

    previous = signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    try:
        if state["started_at"] is None:
            state["started_at"] = utc_now()
        # Phase 1: the initial walk of every missing observation.
        for name, cursor in state["streams"].items():
            if cursor["done"]:
                continue
            status(f"Backfilling {name} from id {cursor['id_above']:,}...")
            cursor["done"] = fetch_stream(
                client, store, name, cursor, {"quality_grade": "research"}, should_stop
            )
            if not cursor["done"]:
                return "budget"
        # Phase 2: changes since the last sync (or since the backfill began, to
        # catch edits made during the weeks it took).
        if state["sync"] is not None and set(state["sync"]["streams"]) != set(SYNC_STREAMS):
            state["sync"] = None  # begun by an older version with other streams
        if state["sync"] is None:
            since = state["last_sync"] or state["started_at"]
            state["sync"] = {
                "since": since,
                "started": utc_now(),
                "streams": {name: {"id_above": 0, "done": False} for name in SYNC_STREAMS},
            }
            store.save_state()
        sync = state["sync"]
        for name, cursor in sync["streams"].items():
            if cursor["done"]:
                continue
            status(f"Syncing changes since {sync['since']}...")
            # No quality or license filter: an update may have taken research
            # grade away, or moved the observation to an export license.
            cursor["done"] = fetch_stream(
                client, store, name, cursor, {"updated_since": sync["since"]}, should_stop
            )
            if not cursor["done"]:
                return "budget"
        state["last_sync"] = sync["started"]
        state["sync"] = None
        store.flush()
        return "synced"
    finally:
        signal.signal(signal.SIGINT, previous)
        store.flush()


def describe(store: Store) -> str:
    state = store.state
    lines = []
    for name, cursor in state["streams"].items():
        left = cursor.get("remaining")
        lines.append(
            f"  {name}: {'done' if cursor['done'] else 'in progress'}, "
            f"id above {cursor['id_above']:,}"
            + (f", about {left:,} left" if left is not None and not cursor["done"] else "")
        )
    lines.append(f"  started: {state['started_at']}, last sync: {state['last_sync']}")
    lines.append(f"  requests today: {state.get('requests_today', 0)} ({state.get('budget_day')})")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, default=Path(__file__).resolve().parent / "backfill",
                        help="backfill directory (default: ./backfill)")
    parser.add_argument("--max-hours", type=float, help="stop after this many hours")
    parser.add_argument("--daily-requests", type=int, default=DEFAULT_DAILY_REQUESTS,
                        help=f"API requests per day (default {DEFAULT_DAILY_REQUESTS:,})")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    parser.add_argument("--compact", action="store_true",
                        help="fold fetched parts into extra.parquet and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        store = Store(args.store)
    except BackfillError as error:
        print(error, file=sys.stderr)
        return 1
    with store:
        if args.status:
            print(describe(store))
            return 0
        if args.compact:
            status(f"extra.parquet now holds {store.compact():,} observations.")
            return 0
        client = ApiClient(store.state, daily_limit=args.daily_requests)
        deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
        try:
            outcome = run(store, client, deadline=deadline)
        except BackfillError as error:
            print(f"Backfill stopped: {error}", file=sys.stderr)
            return 1
    status({"synced": "Up to date.", "budget": "Stopped; run again to continue.",
            "backfilling": "Stopped; run again to continue."}[outcome])
    print(describe(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path

import duckdb
import pandas as pd

import backfill_observations as backfill
import update_database


def observation(oid, grade="research", lat=37.12345, lon=-122.5, taxon=10, day="2024-05-01"):
    return {
        "id": oid,
        "quality_grade": grade,
        "location": f"{lat},{lon}",
        "observed_on": day,
        "taxon": {"id": taxon, "rank": "species"},
    }


class FakeApi:
    """Serves /v2/observations pages from per-filter lists, in id order."""

    def __init__(self, streams, per_page=2):
        self.streams = streams  # {"unlicensed"|"sharealike_noderivs": [obs...]}
        self.per_page = per_page
        self.calls = []
        self.failures = []  # HTTP codes to raise before answering

    def __call__(self, request):
        url = urllib.parse.urlparse(request.full_url)
        params = dict(urllib.parse.parse_qsl(url.query))
        self.calls.append(params)
        if self.failures:
            code = self.failures.pop(0)
            raise urllib.error.HTTPError(request.full_url, code, "busy", {}, None)
        name = "unlicensed" if params.get("licensed") == "false" else "sharealike_noderivs"
        rows = [
            o for o in self.streams.get(name, [])
            if o["id"] > int(params["id_above"])
            and (params.get("quality_grade") is None or o["quality_grade"] == params["quality_grade"])
        ]
        page = rows[: self.per_page]
        body = json.dumps({"total_results": len(rows), "results": page}).encode()
        return io.BytesIO(body)


def client(store, api, daily=1000):
    return backfill.ApiClient(
        store.state, daily_limit=daily, min_interval=0, opener=api, sleep=lambda _s: None
    )


def read(path):
    return duckdb.sql(f"SELECT * FROM '{path}' ORDER BY id").df()


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        backfill.PER_PAGE = 2
        self.addCleanup(setattr, backfill, "PER_PAGE", 200)

    def test_parse_observation(self):
        row = backfill.parse_observation(observation(5))
        self.assertEqual((row["decimalLatitude"], row["decimalLongitude"]), (37.123, -122.5))
        self.assertFalse(row["removed"])
        self.assertTrue(backfill.parse_observation(observation(6, grade="needs_id"))["removed"])
        self.assertTrue(backfill.parse_observation({**observation(7), "location": None})["removed"])
        self.assertTrue(backfill.parse_observation({**observation(8), "observed_on": None})["removed"])
        self.assertIsNone(backfill.parse_observation({"id": "x"}))

    def test_backfill_pages_both_streams_and_resumes_after_budget(self):
        api = FakeApi({
            "unlicensed": [observation(i) for i in (1, 3, 5, 7, 9)],
            "sharealike_noderivs": [observation(i) for i in (2, 4)],
        })
        store = backfill.Store(self.dir / "bf")
        self.assertEqual(backfill.run(store, client(store, api, daily=2)), "budget")
        self.assertEqual(store.state["streams"]["unlicensed"]["id_above"], 7)  # two pages of two

        # A fresh process picks up from the saved cursor.
        store = backfill.Store(self.dir / "bf")
        store.state["requests_today"] = 0
        self.assertEqual(backfill.run(store, client(store, api)), "synced")
        self.assertEqual(store.compact(), 7)
        self.assertEqual(list(read(store.extra_path)["id"]), [1, 2, 3, 4, 5, 7, 9])
        self.assertIsNotNone(store.state["last_sync"])
        self.assertEqual(list(store.parts.glob("*.parquet")), [])

    def test_sync_updates_and_removes_without_quality_filter(self):
        api = FakeApi({"unlicensed": [observation(1), observation(2)]})
        store = backfill.Store(self.dir / "bf")
        backfill.run(store, client(store, api))
        store.compact()
        # Observation 1 lost research grade, 2 moved, 3 is new.
        api.streams["unlicensed"] = [
            observation(1, grade="needs_id"), observation(2, lat=10.0), observation(3)
        ]
        api.calls.clear()
        self.assertEqual(backfill.run(store, client(store, api)), "synced")
        sync_calls = [c for c in api.calls if "updated_since" in c]
        self.assertTrue(sync_calls)
        self.assertTrue(all("quality_grade" not in c for c in sync_calls))
        self.assertEqual(store.compact(), 2)
        frame = read(store.extra_path)
        live = frame[~frame["removed"]]
        self.assertEqual(list(live["id"]), [2, 3])
        self.assertEqual(float(live.iloc[0]["decimalLatitude"]), 10.0)
        self.assertTrue(bool(frame[frame["id"] == 1]["removed"].iloc[0]))

    def test_retries_rate_limits(self):
        api = FakeApi({"unlicensed": [observation(1)]})
        api.failures = [429, 503]
        store = backfill.Store(self.dir / "bf")
        self.assertEqual(backfill.run(store, client(store, api)), "synced")
        self.assertEqual(store.compact(), 1)

    def test_refuses_other_http_errors(self):
        api = FakeApi({"unlicensed": [observation(1)]})
        api.failures = [403]
        store = backfill.Store(self.dir / "bf")
        with self.assertRaises(backfill.BackfillError):
            backfill.run(store, client(store, api))


class MergeTests(unittest.TestCase):
    def test_merge_keeps_export_rows_and_adds_backfill(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            pd.DataFrame({
                "id": [1, 2],
                "decimalLatitude": [1.0, 2.0],
                "decimalLongitude": [1.0, 2.0],
                "taxonID": [10, 10],
                "taxonRank": ["species", "species"],
                "eventDate": pd.to_datetime(["2020-01-01", "2020-01-02"]).date,
            }).to_parquet(directory / "gbif.parquet")
            con = duckdb.connect()
            con.execute(
                f"""COPY (SELECT id, CAST(decimalLatitude AS DECIMAL(8,3)) decimalLatitude,
                    CAST(decimalLongitude AS DECIMAL(8,3)) decimalLongitude, taxonID, taxonRank,
                    eventDate FROM '{directory / "gbif.parquet"}')
                    TO '{directory / update_database.GBIF_BASE_NAME}' (FORMAT PARQUET)"""
            )
            store = backfill.Store(directory / "backfill")
            store.add([
                {**backfill.parse_observation(observation(2, lat=50.0))},  # in export: ignored
                backfill.parse_observation(observation(3)),
                backfill.parse_observation(observation(4)),
                {"id": 4, "removed": True},  # later removal wins
            ])
            store.flush()
            store.compact()
            stats = update_database.merge_observations(directory, store.extra_path)
            frame = read(directory / "observations.parquet")
            self.assertEqual(list(frame["id"]), [1, 2, 3])
            self.assertEqual(float(frame.iloc[1]["decimalLatitude"]), 2.0)
            self.assertEqual(stats["rows"], 3)

    def test_merge_without_backfill_links_the_export(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            pd.DataFrame({
                "id": [1], "decimalLatitude": [1.0], "decimalLongitude": [1.0], "taxonID": [10],
                "taxonRank": ["species"], "eventDate": pd.to_datetime(["2020-01-01"]).date,
            }).to_parquet(directory / "raw.parquet")
            duckdb.execute(
                f"""COPY (SELECT id, CAST(decimalLatitude AS DECIMAL(8,3)) decimalLatitude,
                    CAST(decimalLongitude AS DECIMAL(8,3)) decimalLongitude, taxonID, taxonRank,
                    eventDate FROM '{directory / "raw.parquet"}')
                    TO '{directory / "observations.parquet"}' (FORMAT PARQUET)"""
            )
            # An existing database from before backfills becomes the export build.
            update_database.merge_observations(directory, directory / "missing.parquet")
            self.assertTrue((directory / update_database.GBIF_BASE_NAME).exists())
            self.assertEqual(len(read(directory / "observations.parquet")), 1)


if __name__ == "__main__":
    unittest.main()

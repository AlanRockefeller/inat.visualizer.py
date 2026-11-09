import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

try:
    from fastapi.testclient import TestClient

    from webapp import server
except (ImportError, RuntimeError):
    # Desktop-only environments do not install requirements-web.txt. Starlette
    # raises RuntimeError rather than ImportError when httpx is missing.
    server = None

SF = (37.7749, -122.4194)
NEW_YORK = (40.7128, -74.0060)

# Taxon 1 is the parent of 2; taxon 3 is unrelated.
TAXONOMY = [
    {"id": 1, "parent_id": None},
    {"id": 2, "parent_id": 1},
    {"id": 3, "parent_id": None},
]
OBSERVATIONS = [
    {"id": 10, "lat": SF[0], "lon": SF[1], "taxonID": 2, "eventDate": "2020-03-15"},
    {"id": 11, "lat": SF[0] + 0.01, "lon": SF[1], "taxonID": 2, "eventDate": "2021-03-20"},
    # ISO week 53; the graph folds it into week 52 of the same year.
    {"id": 12, "lat": SF[0], "lon": SF[1] + 0.01, "taxonID": 3, "eventDate": "2020-12-31"},
    {"id": 13, "lat": NEW_YORK[0], "lon": NEW_YORK[1], "taxonID": 2, "eventDate": "2020-04-01"},
]


def write_observations(path, rows):
    frame = pd.DataFrame(
        {
            "id": [row["id"] for row in rows],
            "decimalLatitude": [row["lat"] for row in rows],
            "decimalLongitude": [row["lon"] for row in rows],
            "taxonID": [row["taxonID"] for row in rows],
            "taxonRank": ["species"] * len(rows),
            "eventDate": pd.to_datetime([row["eventDate"] for row in rows]).date,
        }
    )
    frame.to_parquet(path, index=False)


@unittest.skipIf(
    server is None, "fastapi/httpx are not installed (pip install -r requirements-web.txt)"
)
class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.observations = os.path.join(self.tmp.name, "observations.parquet")
        self.taxonomy = os.path.join(self.tmp.name, "taxonomy.parquet")
        write_observations(self.observations, OBSERVATIONS)
        pd.DataFrame(TAXONOMY).astype({"parent_id": "Int32"}).to_parquet(
            self.taxonomy, index=False
        )
        patches = [
            patch.object(server, "OBSERVATIONS_FILE", server.Path(self.observations)),
            patch.object(server, "TAXONOMY_FILE", server.Path(self.taxonomy)),
            patch.object(server, "descendant_cache", server.LruCache(16)),
            patch.object(server, "result_cache", server.LruCache(16)),
            patch.object(server, "query_semaphore", None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.tmp.cleanup)

    def seasonal(self, **params):
        query = {
            "lat": SF[0],
            "lon": SF[1],
            "radius": 50,
            "date_from": "2000-01-01",
            "date_to": "2030-01-01",
            **params,
        }
        return self.client.get("/api/seasonal", params=query)

    def abundance(self, **params):
        query = {
            "season_start": "03-01",
            "season_end": "05-31",
            "date_from": "2000-01-01",
            "date_to": "2030-01-01",
            "cell_deg": 1.0,
            **params,
        }
        return self.client.get("/api/abundance", params=query)

    def test_stats(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_observations"], 4)
        self.assertEqual(body["newest_date"], "2021-03-20")

    def test_seasonal_expands_descendants_and_bins_every_view(self):
        body = self.seasonal(taxon_id=1).json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["taxa_included"], 2)
        self.assertEqual(body["bins"]["monthly"][2], 2)
        self.assertEqual(sum(body["bins"]["monthly"]), 2)
        daily = body["bins"]["daily"]
        self.assertEqual(len(daily), 366)
        # Daily bins follow a leap-year calendar: 2020-03-15 is day 75, and
        # 2021-03-20 (day 79 of a common year) lands on Mar 20 = day 80.
        self.assertEqual((daily[74], daily[79]), (1, 1))
        self.assertEqual(len(body["bins"]["weekly"]), 52)

    def test_seasonal_all_organisms_folds_iso_week_53_into_week_52(self):
        body = self.seasonal().json()
        self.assertEqual(body["total"], 3)  # New York is outside the radius.
        self.assertEqual(body["bins"]["weekly"][51], 1)

    def test_seasonal_exclude_removes_descendants(self):
        body = self.seasonal(exclude_taxon_id=1).json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["bins"]["monthly"][11], 1)

    def test_security_headers(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "strict-origin-when-cross-origin")

    def test_seasonal_across_antimeridian(self):
        fiji_side = {"id": 20, "lat": -16.5, "lon": 179.9, "taxonID": 3, "eventDate": "2020-06-01"}
        far_side = {"id": 21, "lat": -16.5, "lon": -179.9, "taxonID": 3, "eventDate": "2020-06-02"}
        write_observations(self.observations, OBSERVATIONS + [fiji_side, far_side])
        body = self.seasonal(lat=-16.5, lon=179.95, radius=100).json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["bins"]["monthly"][5], 2)

    def test_cache_discards_results_computed_before_database_swap(self):
        cache = server.LruCache(4)
        stale_signature = ("old",)
        cache.put("key", "stale", stale_signature)
        self.assertIsNone(cache.get("key"))
        cache.put("key", "fresh", server.database_signature())
        self.assertEqual(cache.get("key"), "fresh")

    def test_descendants_of_cyclic_taxonomy_terminate(self):
        pd.DataFrame([{"id": 1, "parent_id": 3}, {"id": 2, "parent_id": 1}, {"id": 3, "parent_id": 2}]).astype(
            {"parent_id": "Int32"}
        ).to_parquet(self.taxonomy, index=False)
        body = self.seasonal(taxon_id=1).json()
        self.assertEqual(body["taxa_included"], 3)

    def test_seasonal_csv(self):
        response = self.seasonal(taxon_id=2, format="csv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/csv"))
        lines = response.text.strip().splitlines()
        self.assertEqual(lines[0], "id,decimalLatitude,decimalLongitude,eventDate,taxonID")
        self.assertEqual(sorted(line.split(",")[0] for line in lines[1:]), ["10", "11"])

    def test_seasonal_csv_row_limit(self):
        with patch.object(server, "MAX_CSV_ROWS", 1):
            response = self.seasonal(taxon_id=2, format="csv")
        self.assertEqual(response.status_code, 413)

    def test_abundance_cells(self):
        body = self.abundance(taxon_id=1).json()
        self.assertEqual(body["total"], 3)
        counts = sorted(cell["count"] for cell in body["cells"])
        self.assertEqual(counts, [1, 2])  # San Francisco cell and New York cell.
        top = max(body["cells"], key=lambda cell: cell["count"])
        self.assertEqual((top["cell_lat"], top["cell_lon"]), (37.0, -123.0))
        self.assertGreaterEqual(top["radius_km"], 1)

    def test_abundance_season_wrapping_past_new_year(self):
        body = self.abundance(taxon_id=3, season_start="12-01", season_end="02-29").json()
        self.assertEqual(body["total"], 1)

    def test_validation_errors(self):
        self.assertEqual(self.seasonal(radius=5000).status_code, 422)
        self.assertEqual(self.seasonal(lat=95).status_code, 422)
        self.assertEqual(
            self.seasonal(date_from="2022-01-01", date_to="2021-01-01").status_code, 422
        )
        self.assertEqual(self.seasonal(date_from="2022-02-30").status_code, 422)
        self.assertEqual(self.seasonal(taxon_id=list(range(1, 20))).status_code, 422)
        self.assertEqual(self.abundance().status_code, 422)  # taxon_id is required
        self.assertEqual(self.abundance(taxon_id=1, cell_deg=3).status_code, 422)
        self.assertEqual(self.abundance(taxon_id=1, season_start="13-01").status_code, 422)

    def test_results_refresh_when_database_file_changes(self):
        self.assertEqual(self.seasonal().json()["total"], 3)
        extra = {"id": 14, "lat": SF[0], "lon": SF[1], "taxonID": 3, "eventDate": "2022-06-01"}
        write_observations(self.observations, OBSERVATIONS + [extra])
        stat = os.stat(self.observations)
        os.utime(self.observations, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        self.assertEqual(self.seasonal().json()["total"], 4)

    def test_missing_database_returns_503(self):
        os.remove(self.observations)
        self.assertEqual(self.client.get("/api/stats").status_code, 503)

    def test_index_page_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("app.js", response.text)


if __name__ == "__main__":
    unittest.main()

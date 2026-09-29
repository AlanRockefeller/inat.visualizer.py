import os
import tempfile
import unittest

import pandas as pd

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from visualizer import (
    SearchCancelled,
    run_local_abundance_query,
    run_local_observation_query,
    season_filter_sql,
)


def write_observations(directory, rows):
    parquet_path = os.path.join(directory, "observations.parquet")
    frame = pd.DataFrame(rows)
    frame["eventDate"] = pd.to_datetime(frame["eventDate"]).dt.date
    frame.to_parquet(parquet_path)
    return parquet_path


def obs(obs_id, lat, lon, date, taxon):
    return {
        "id": obs_id,
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "eventDate": date,
        "taxonID": taxon,
    }


class TestSeasonFilter(unittest.TestCase):
    def test_non_wrapping_range_uses_between(self):
        sql, params = season_filter_sql((10, 1), (11, 30))
        self.assertIn("BETWEEN", sql)
        self.assertEqual(params, [1001, 1130])

    def test_wrapping_range_uses_or(self):
        sql, params = season_filter_sql((11, 15), (2, 10))
        self.assertIn(" OR ", sql)
        self.assertEqual(params, [1115, 210])


class TestAbundanceQuery(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def query(self, rows, **overrides):
        path = write_observations(self._tmp.name, rows)
        kwargs = {
            "date_from": "2000-01-01",
            "date_to": "2030-12-31",
            "season_start": (1, 1),
            "season_end": (12, 31),
            "cell_deg": 1.0,
            "taxon_ids": [10],
        }
        kwargs.update(overrides)
        return run_local_abundance_query(path, **kwargs)

    def test_counts_target_and_total_per_cell(self):
        cells = self.query(
            [
                obs(1, 37.2, -122.4, "2024-10-05", 10),
                obs(2, 37.8, -122.9, "2024-10-06", 10),
                obs(3, 37.5, -122.5, "2024-10-07", 20),
                obs(4, 37.5, -122.5, "2024-10-08", 20),
                obs(5, 51.5, -0.1, "2024-10-09", 10),
            ]
        )
        by_cell = {(c["cell_lat"], c["cell_lon"]): c for c in cells}
        bay = by_cell[(37.0, -123.0)]
        self.assertEqual(bay["count"], 2)
        self.assertEqual(bay["total"], 4)
        self.assertAlmostEqual(bay["share"], 0.5)
        self.assertEqual(by_cell[(51.0, -1.0)]["count"], 1)
        self.assertEqual(cells[0]["count"], 2, "cells are sorted by count")

    def test_cells_without_target_are_omitted(self):
        cells = self.query(
            [obs(1, 10.5, 10.5, "2024-06-01", 20), obs(2, 20.5, 20.5, "2024-06-01", 10)]
        )
        self.assertEqual([(c["cell_lat"], c["cell_lon"]) for c in cells], [(20.0, 20.0)])

    def test_negative_coordinates_floor_to_southwest_corner(self):
        cells = self.query([obs(1, -33.9, -70.6, "2024-06-01", 10)], cell_deg=0.5)
        self.assertEqual((cells[0]["cell_lat"], cells[0]["cell_lon"]), (-34.0, -71.0))

    def test_non_wrapping_season(self):
        cells = self.query(
            [
                obs(1, 1.5, 1.5, "2020-09-30", 10),
                obs(2, 1.5, 1.5, "2021-10-01", 10),
                obs(3, 1.5, 1.5, "2022-11-30", 10),
                obs(4, 1.5, 1.5, "2023-12-01", 10),
            ],
            season_start=(10, 1),
            season_end=(11, 30),
        )
        self.assertEqual(cells[0]["count"], 2)

    def test_wrapping_season_crosses_new_year(self):
        cells = self.query(
            [
                obs(1, 1.5, 1.5, "2020-12-31", 10),
                obs(2, 1.5, 1.5, "2021-01-01", 10),
                obs(3, 1.5, 1.5, "2021-02-10", 10),
                obs(4, 1.5, 1.5, "2021-02-11", 10),
                obs(5, 1.5, 1.5, "2021-11-14", 10),
                obs(6, 1.5, 1.5, "2021-11-15", 10),
            ],
            season_start=(11, 15),
            season_end=(2, 10),
        )
        self.assertEqual(cells[0]["count"], 4)

    def test_date_range_limits_years(self):
        cells = self.query(
            [
                obs(1, 1.5, 1.5, "2019-10-10", 10),
                obs(2, 1.5, 1.5, "2022-10-10", 10),
            ],
            date_from="2020-01-01",
            date_to="2023-12-31",
        )
        self.assertEqual(cells[0]["count"], 1)

    def test_excluded_taxa_are_not_counted_but_stay_in_total(self):
        cells = self.query(
            [
                obs(1, 1.5, 1.5, "2024-10-10", 10),
                obs(2, 1.5, 1.5, "2024-10-10", 11),
            ],
            taxon_ids=[10, 11],
            exclude_taxon_ids=[11],
        )
        self.assertEqual((cells[0]["count"], cells[0]["total"]), (1, 2))

    def test_empty_taxon_list_raises(self):
        with self.assertRaisesRegex(ValueError, "organism is required"):
            self.query([obs(1, 1.5, 1.5, "2024-10-10", 10)], taxon_ids=[])

    def test_cancel_callback_raises(self):
        with self.assertRaises(SearchCancelled):
            self.query(
                [obs(1, 1.5, 1.5, "2024-10-10", 10)], cancel_callback=lambda: True
            )


class TestLargeTaxonLists(unittest.TestCase):
    """Higher taxa expand to tens of thousands of IDs; both queries must cope."""

    def test_thousands_of_taxon_ids(self):
        taxon_ids = list(range(1000, 31000))
        rows = [
            obs(1, 37.7749, -122.4194, "2024-10-10", 1500),
            obs(2, 37.7749, -122.4194, "2024-10-10", 30999),
            obs(3, 37.7749, -122.4194, "2024-10-10", 5),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_observations(tmpdir, rows)
            observations, count = run_local_observation_query(
                path,
                "2024-01-01",
                "2024-12-31",
                37.7749,
                -122.4194,
                1,
                taxon_ids=taxon_ids,
                exclude_taxon_ids=[30999],
            )
            cells = run_local_abundance_query(
                path, "2024-01-01", "2024-12-31", (1, 1), (12, 31), 1.0, taxon_ids
            )
        self.assertEqual(count, 1)
        self.assertEqual([o["id"] for o in observations], [1])
        self.assertEqual((cells[0]["count"], cells[0]["total"]), (2, 3))


if __name__ == "__main__":
    unittest.main()

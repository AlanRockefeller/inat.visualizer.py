import os
import tempfile
import unittest

import pandas as pd

from inat_core import (
    calculate_local_search_bounds,
    descendant_taxon_ids,
    leap_aligned_day_of_year,
    longitude_filter_sql,
    run_local_date_counts,
    run_local_observation_query,
    seasonal_bin_counts,
)


def write_observations(directory, rows):
    path = os.path.join(directory, "observations.parquet")
    frame = pd.DataFrame(rows)
    frame["eventDate"] = pd.to_datetime(frame["eventDate"]).dt.date
    frame.to_parquet(path, index=False)
    return path


def obs(obs_id, lat, lon, date, taxon=1):
    return {
        "id": obs_id,
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "taxonID": taxon,
        "taxonRank": "species",
        "eventDate": date,
    }


class LongitudeBoundsTests(unittest.TestCase):
    def test_plain_box_uses_between(self):
        sql, params = longitude_filter_sql(-10.0, 10.0)
        self.assertEqual(sql, "decimalLongitude BETWEEN ? AND ?")
        self.assertEqual(params, [-10.0, 10.0])

    def test_box_crossing_antimeridian_wraps(self):
        for lon_min, lon_max in ((170.0, 190.0), (-190.0, -170.0)):
            sql, params = longitude_filter_sql(lon_min, lon_max)
            self.assertEqual(sql, "(decimalLongitude >= ? OR decimalLongitude <= ?)")
            self.assertEqual(params, [170.0, -170.0])

    def test_box_around_the_world_matches_everything(self):
        self.assertEqual(longitude_filter_sql(-180.0, 180.0), ("TRUE", []))

    def test_circle_containing_a_pole_spans_all_longitudes(self):
        lat_min, lat_max, lon_min, lon_max = calculate_local_search_bounds(
            89.9, 10.0, 100
        )
        self.assertEqual((lon_min, lon_max), (-180.0, 180.0))
        self.assertEqual(lat_max, 90.0)
        self.assertLess(lat_min, 89.9)


class AntimeridianQueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = write_observations(
            self.tmp.name,
            [
                obs(1, -16.5, 179.9, "2020-06-01"),
                obs(2, -16.5, -179.9, "2020-06-02"),
                obs(3, -16.5, 180.0, "2020-06-03"),
                obs(4, -16.5, 170.0, "2020-06-04"),
            ],
        )

    def test_observation_query_finds_both_sides(self):
        for lon in (179.9, -179.9):
            rows, count = run_local_observation_query(
                self.path, "2000-01-01", "2030-01-01", -16.5, lon, 100
            )
            self.assertEqual(sorted(row["id"] for row in rows), [1, 2, 3])
            self.assertEqual(count, 3)

    def test_observation_query_without_estimate_counts_rows(self):
        rows, count = run_local_observation_query(
            self.path,
            "2000-01-01",
            "2030-01-01",
            -16.5,
            179.9,
            100,
            estimate_count=False,
        )
        self.assertEqual(count, len(rows))

    def test_date_counts_find_both_sides(self):
        frame = run_local_date_counts(
            self.path, "2000-01-01", "2030-01-01", -16.5, -179.95, 100
        )
        self.assertEqual(int(frame["count"].sum()), 3)


class LeapAlignedDailyBinTests(unittest.TestCase):
    def test_same_calendar_date_shares_a_bin_in_every_year(self):
        dates = pd.to_datetime(
            pd.Series(
                [
                    "2020-02-28",
                    "2021-02-28",
                    "2020-02-29",
                    "2020-03-01",
                    "2021-03-01",
                    "2021-12-31",
                ]
            )
        )
        self.assertEqual(
            list(leap_aligned_day_of_year(dates)), [59, 59, 60, 61, 61, 366]
        )
        daily = seasonal_bin_counts(dates, "daily")
        self.assertEqual((daily[58], daily[59], daily[60], daily[365]), (2, 1, 2, 1))
        self.assertEqual(sum(daily), 6)


class DescendantTaxonTests(unittest.TestCase):
    def write_taxonomy(self, rows):
        path = os.path.join(self.tmp.name, "taxonomy.parquet")
        pd.DataFrame(rows).astype({"id": "int32", "parent_id": "Int32"}).to_parquet(
            path, index=False
        )
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_returns_taxon_and_descendants(self):
        path = self.write_taxonomy(
            [
                {"id": 1, "parent_id": None},
                {"id": 2, "parent_id": 1},
                {"id": 3, "parent_id": 2},
                {"id": 4, "parent_id": None},
            ]
        )
        self.assertEqual(sorted(descendant_taxon_ids(path, 1)), [1, 2, 3])
        self.assertEqual(descendant_taxon_ids(path, 99), [99])

    def test_parent_cycle_terminates(self):
        path = self.write_taxonomy(
            [
                {"id": 1, "parent_id": 3},
                {"id": 2, "parent_id": 1},
                {"id": 3, "parent_id": 2},
            ]
        )
        self.assertEqual(sorted(descendant_taxon_ids(path, 1)), [1, 2, 3])

    def test_callbacks_receive_connection_and_cancellation(self):
        path = self.write_taxonomy([{"id": 1, "parent_id": None}])
        seen = []
        descendant_taxon_ids(path, 1, connection_callback=seen.append)
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[1])
        from inat_core import SearchCancelled

        with self.assertRaises(SearchCancelled):
            descendant_taxon_ids(path, 1, cancel_callback=lambda: True)


if __name__ == "__main__":
    unittest.main()

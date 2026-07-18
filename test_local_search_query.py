import os
import tempfile
import unittest

import pandas as pd

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from visualizer import run_local_observation_query


class TestLocalSearchQuery(unittest.TestCase):
    def test_filters_by_radius_and_taxon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "observations.parquet")
            pd.DataFrame(
                [
                    {
                        "id": 1,
                        "decimalLatitude": 37.7749,
                        "decimalLongitude": -122.4194,
                        "eventDate": "2024-01-15",
                        "taxonID": 10,
                    },
                    {
                        "id": 2,
                        "decimalLatitude": 37.7750,
                        "decimalLongitude": -122.4195,
                        "eventDate": "2024-02-15",
                        "taxonID": 20,
                    },
                    {
                        "id": 3,
                        "decimalLatitude": 40.7128,
                        "decimalLongitude": -74.0060,
                        "eventDate": "2024-03-15",
                        "taxonID": 10,
                    },
                ]
            ).to_parquet(parquet_path)

            observations, estimated_count = run_local_observation_query(
                parquet_path,
                "2024-01-01",
                "2024-12-31",
                37.7749,
                -122.4194,
                1,
                taxon_ids=[10],
            )

        self.assertEqual(estimated_count, 1)
        self.assertEqual([obs["id"] for obs in observations], [1])

    def test_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "observations.parquet")
            pd.DataFrame([{"id": 1, "eventDate": "2024-01-15"}]).to_parquet(
                parquet_path
            )

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                run_local_observation_query(
                    parquet_path,
                    "2024-01-01",
                    "2024-12-31",
                    37.7749,
                    -122.4194,
                    1,
                )


if __name__ == "__main__":
    unittest.main()

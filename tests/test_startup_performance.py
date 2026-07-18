"""Tests for startup caches that avoid repeated expensive initialization."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from visualizer import (
    INatSeasonalVisualizer,
    read_database_stats_cache,
    write_database_stats_cache,
)


class StartupPerformanceTests(unittest.TestCase):
    def test_database_stats_cache_matches_only_the_same_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "observations.parquet"
            cache_path = Path(temp_dir) / "database_stats.json"
            database_path.write_bytes(b"first snapshot")

            write_database_stats_cache(cache_path, database_path, 123, 45)

            self.assertEqual(
                read_database_stats_cache(cache_path, database_path),
                (123, 45),
            )
            database_path.write_bytes(b"different snapshot")
            self.assertIsNone(read_database_stats_cache(cache_path, database_path))

    def test_load_database_stats_uses_cache_without_opening_duckdb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "observations.parquet"
            cache_path = Path(temp_dir) / "database_stats.json"
            database_path.write_bytes(b"snapshot")
            write_database_stats_cache(cache_path, database_path, 321, 54)
            window = SimpleNamespace(
                observations_file=str(database_path),
                database_stats_cache_file=str(cache_path),
                total_observations=0,
                unique_taxa=0,
            )

            with patch("visualizer.duckdb.connect") as connect:
                INatSeasonalVisualizer.load_database_stats(window)

            connect.assert_not_called()
            self.assertEqual(window.total_observations, 321)
            self.assertEqual(window.unique_taxa, 54)

    def test_database_stats_query_populates_cache_after_snapshot_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "observations.parquet"
            cache_path = Path(temp_dir) / "database_stats.json"
            database_path.write_bytes(b"snapshot")
            connection = MagicMock()
            connection.execute.return_value.fetchall.return_value = [("taxonID",)]
            connection.execute.return_value.fetchone.return_value = (999, 88)
            window = SimpleNamespace(
                observations_file=str(database_path),
                database_stats_cache_file=str(cache_path),
                total_observations=0,
                unique_taxa=0,
            )

            with patch("visualizer.duckdb.connect", return_value=connection):
                INatSeasonalVisualizer.load_database_stats(window)

            self.assertEqual(window.total_observations, 999)
            self.assertEqual(window.unique_taxa, 88)
            self.assertEqual(
                read_database_stats_cache(cache_path, database_path),
                (999, 88),
            )
            connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

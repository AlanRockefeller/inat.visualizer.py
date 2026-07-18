"""Tests for bounded app-owned iNaturalist HTTP response caching."""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyinaturalist

from visualizer import (
    DEFAULT_HTTP_CACHE_MAX_MB,
    HTTP_CACHE_MAX_MB_ENV,
    INatSeasonalVisualizer,
    configured_http_cache_max_bytes,
    maintain_http_cache,
    maintain_legacy_pyinaturalist_cache,
    sqlite_cache_disk_usage,
)


class HttpCacheManagementTests(unittest.TestCase):
    def _create_cache(self, path: Path):
        session = pyinaturalist.ClientSession(
            cache_file=str(path), max_retries=0, timeout=10
        )
        self.addCleanup(session.close)
        return session

    @staticmethod
    def _insert_response(path: Path, *, expires: int, size: int = 1024 * 1024) -> None:
        with sqlite3.connect(path) as connection:
            connection.execute(
                "INSERT INTO responses (key, value, expires) VALUES (?, ?, ?)",
                (f"response-{expires}", sqlite3.Binary(b"x" * size), expires),
            )

    def test_cache_budget_defaults_and_environment_override(self) -> None:
        self.assertEqual(
            configured_http_cache_max_bytes(environ={}),
            DEFAULT_HTTP_CACHE_MAX_MB * 1024 * 1024,
        )
        self.assertEqual(
            configured_http_cache_max_bytes(environ={HTTP_CACHE_MAX_MB_ENV: "64"}),
            64 * 1024 * 1024,
        )

    def test_cache_disk_usage_includes_wal_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "api_cache.db"
            cache_path.write_bytes(b"d" * 10)
            Path(f"{cache_path}-wal").write_bytes(b"w" * 20)
            Path(f"{cache_path}-shm").write_bytes(b"s" * 30)

            self.assertEqual(sqlite_cache_disk_usage(cache_path), 60)

    def test_expired_responses_are_removed_and_disk_space_is_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "api_cache.db"
            session = self._create_cache(cache_path)
            self._insert_response(cache_path, expires=int(time.time()) - 60)

            maintain_http_cache(session, cache_path, 64 * 1024)

            self.assertEqual(len(session.cache.responses), 0)
            self.assertLessEqual(cache_path.stat().st_size, 64 * 1024)

    def test_app_cache_is_cleared_when_valid_entries_exceed_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "api_cache.db"
            session = self._create_cache(cache_path)
            self._insert_response(cache_path, expires=int(time.time()) + 3600)

            maintain_http_cache(session, cache_path, 64 * 1024)

            self.assertEqual(len(session.cache.responses), 0)
            self.assertLessEqual(cache_path.stat().st_size, 64 * 1024)

    def test_legacy_cache_keeps_valid_entries_even_if_still_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "legacy.db"
            session = self._create_cache(cache_path)
            self._insert_response(cache_path, expires=int(time.time()) + 3600)

            maintain_http_cache(
                session,
                cache_path,
                64 * 1024,
                clear_valid_if_oversize=False,
            )

            self.assertEqual(len(session.cache.responses), 1)
            self.assertGreater(cache_path.stat().st_size, 64 * 1024)

    def test_legacy_cleanup_removes_only_expired_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "legacy.db"
            session = self._create_cache(cache_path)
            self._insert_response(cache_path, expires=int(time.time()) - 60)
            session.close()

            with patch("visualizer.PYINATURALIST_DEFAULT_CACHE_FILE", str(cache_path)):
                maintain_legacy_pyinaturalist_cache(
                    Path(temp_dir) / "app_cache.db", 64 * 1024
                )

            with sqlite3.connect(cache_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[
                    0
                ]
            self.assertEqual(count, 0)
            self.assertLessEqual(cache_path.stat().st_size, 64 * 1024)

    def test_taxon_lookup_uses_app_owned_cache(self) -> None:
        session = MagicMock()
        harness = SimpleNamespace(
            taxon_cache={},
            http_cache_file="/runtime/inat_api_cache.db",
            http_cache_max_bytes=128 * 1024 * 1024,
            api_call_count=0,
            update_api_call_count=MagicMock(),
            save_taxon_cache=MagicMock(),
        )

        with (
            patch(
                "visualizer.pyinaturalist.ClientSession", return_value=session
            ) as client_session,
            patch(
                "visualizer.pyinaturalist.get_taxa",
                return_value={"results": [{"id": 47170}]},
            ) as get_taxa,
            patch("visualizer.maintain_http_cache") as maintain_cache,
        ):
            result = INatSeasonalVisualizer.get_taxon_id(harness, "Agaricales")

        self.assertEqual(result, 47170)
        client_session.assert_called_once_with(
            cache_file="/runtime/inat_api_cache.db", max_retries=0, timeout=10
        )
        get_taxa.assert_called_once_with(q="Agaricales", limit=1, session=session)
        self.assertEqual(maintain_cache.call_count, 2)
        session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

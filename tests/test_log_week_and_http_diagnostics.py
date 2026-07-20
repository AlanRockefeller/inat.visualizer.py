import logging
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from requests import Response
from requests.exceptions import HTTPError

from visualizer import (
    HTTP_ERROR_BODY_LOG_LIMIT,
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    INatSeasonalVisualizer,
    calendar_aligned_week_numbers,
    create_log_file_handler,
    http_error_details,
    private_text_identifier,
    privacy_safe_message,
    privacy_safe_path,
)


class WeekBoundaryTests(unittest.TestCase):
    def test_iso_year_boundaries_stay_at_the_correct_calendar_edge(self) -> None:
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2018-12-31",  # ISO week 1 of the next year
                    "2020-12-28",  # ISO week 53
                    "2021-01-01",  # ISO week 53 of the previous year
                    "2021-01-04",  # Ordinary ISO week 1
                    "2021-06-15",  # Ordinary ISO week 24
                ]
            )
        )

        self.assertEqual(
            calendar_aligned_week_numbers(dates).tolist(),
            [52, 52, 1, 1, 24],
        )


class HttpDiagnosticTests(unittest.TestCase):
    def test_falsey_error_response_keeps_status_and_bounded_body(self) -> None:
        response = Response()
        response.status_code = 400
        response._content = b"invalid taxon filter\n" * 100
        error = HTTPError("bad request", response=response)

        status, body = http_error_details(error)

        self.assertEqual(status, 400)
        self.assertIsNotNone(body)
        self.assertLessEqual(len(body or ""), HTTP_ERROR_BODY_LOG_LIMIT)
        self.assertNotIn("\n", body or "")


class LogRotationTests(unittest.TestCase):
    def test_oversized_log_rotates_and_retention_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            log_path = Path(temp_name, "application.log")
            log_path.write_bytes(b"x" * (LOG_MAX_BYTES + 1))
            handler = create_log_file_handler(log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            try:
                handler.emit(
                    logging.LogRecord(
                        "test",
                        logging.INFO,
                        __file__,
                        1,
                        "new session",
                        (),
                        None,
                    )
                )
            finally:
                handler.close()

            self.assertEqual(handler.maxBytes, LOG_MAX_BYTES)
            self.assertEqual(handler.backupCount, LOG_BACKUP_COUNT)
            self.assertEqual(log_path.read_text(encoding="utf-8"), "new session\n")
            self.assertTrue(Path(f"{log_path}.1").exists())


class LogPrivacyTests(unittest.TestCase):
    def test_home_paths_and_user_text_are_redacted(self) -> None:
        private_path = Path.home() / "private" / "cache.db"
        text_id = private_text_identifier("Suriname")

        self.assertEqual(privacy_safe_path(private_path), "~/private/cache.db")
        self.assertEqual(
            privacy_safe_message(f"failed at {private_path}"),
            "failed at ~/private/cache.db",
        )
        self.assertTrue(text_id.startswith("text-"))
        self.assertNotIn("Suriname", text_id)
        self.assertEqual(text_id, private_text_identifier("Suriname"))

    def test_descendant_cache_log_removes_the_complete_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "taxon_cache.json"
            cache_path.write_text(
                json.dumps({"Psathyrella_descendants": [1, 2]}), encoding="utf-8"
            )
            harness = SimpleNamespace(taxon_cache_file=str(cache_path), taxon_cache={})

            with self.assertLogs(level=logging.DEBUG) as captured_logs:
                INatSeasonalVisualizer.load_taxon_cache(harness)

        output = "\n".join(captured_logs.output)
        self.assertIn("descendant taxon IDs for Psathyrella", output)
        self.assertNotIn("Psathyrella_", output)


if __name__ == "__main__":
    unittest.main()

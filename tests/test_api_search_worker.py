"""Tests for responsive live iNaturalist searches."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QCoreApplication, QTimer

from visualizer import (
    ApiSearchWorker,
    LocalSearchWorker,
    SearchCancelled,
    fetch_all_observation_pages,
)


class ApiSearchWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_network_wait_does_not_block_qt_event_processing(self) -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()

        def slow_fetch(*_args, **_kwargs):
            worker_started.set()
            release_worker.wait(2)
            return [], None

        worker = ApiSearchWorker({}, "", "", {})
        timer_fired = []
        with patch("visualizer.fetch_all_observation_pages", side_effect=slow_fetch):
            worker.start()
            try:
                self.assertTrue(worker_started.wait(1))
                QTimer.singleShot(0, lambda: timer_fired.append(True))

                deadline = time.monotonic() + 1
                while not timer_fired and time.monotonic() < deadline:
                    self.app.processEvents()

                self.assertEqual(timer_fired, [True])
                self.assertTrue(worker.isRunning())
            finally:
                release_worker.set()
                worker.wait(1000)

    def test_worker_resolves_taxon_before_fetching_observations(self) -> None:
        captured_params = []
        finished_results = []

        def fetch_pages(params, **_kwargs):
            captured_params.append(params.copy())
            return [{"id": 123}], None

        worker = ApiSearchWorker({}, "Agaricales", "", {})
        worker.search_finished.connect(
            lambda observations, warnings: finished_results.append(
                (observations, warnings)
            )
        )

        with (
            patch(
                "visualizer.pyinaturalist.get_taxa",
                return_value={"results": [{"id": 47170}]},
            ),
            patch("visualizer.fetch_all_observation_pages", side_effect=fetch_pages),
        ):
            worker.run()

        self.assertEqual(captured_params, [{"taxon_id": 47170}])
        self.assertEqual(finished_results, [([{"id": 123}], [])])

    def test_rate_limit_wait_can_be_cancelled_immediately(self) -> None:
        cancel_requested = False

        def progress(message, _percentage):
            nonlocal cancel_requested
            if message.startswith("Waiting briefly"):
                cancel_requested = True

        response = {"results": [{"id": 1}], "total_results": 2}
        with patch("visualizer.fetch_observations_page", return_value=response):
            with self.assertRaises(SearchCancelled):
                fetch_all_observation_pages(
                    {"taxon_id": list(range(500))},
                    progress_callback=progress,
                    cancel_callback=lambda: cancel_requested,
                )

    def test_get_pagination_has_no_extra_fixed_delay(self) -> None:
        responses = [
            {"results": [{"id": 1}], "total_results": 2},
            {"results": [{"id": 2}], "total_results": 2},
        ]

        with (
            patch("visualizer.fetch_observations_page", side_effect=responses),
            patch("visualizer.wait_for_search_delay") as wait_for_delay,
        ):
            observations, error = fetch_all_observation_pages({"taxon_id": 1})

        self.assertIsNone(error)
        self.assertEqual([item["id"] for item in observations], [1, 2])
        wait_for_delay.assert_not_called()

    def test_cancel_before_run_emits_cancelled_without_fetching(self) -> None:
        worker = ApiSearchWorker(
            {}, "", "", {}, "/runtime/inat_api_cache.db", 64 * 1024 * 1024
        )
        cancelled = []
        worker.search_cancelled.connect(lambda: cancelled.append(True))
        worker.cancel()
        request_session = MagicMock()

        with (
            patch(
                "visualizer.pyinaturalist.ClientSession",
                return_value=request_session,
            ) as client_session,
            patch("visualizer.fetch_all_observation_pages") as fetch_pages,
        ):
            worker.run()

        fetch_pages.assert_not_called()
        client_session.assert_called_once_with(
            cache_file="/runtime/inat_api_cache.db",
            max_retries=0,
            timeout=10,
        )
        request_session.hooks.setdefault.assert_called_once_with("response", [])
        request_session.hooks.setdefault.return_value.append.assert_called_once()
        request_session.close.assert_called_once_with()
        self.assertEqual(cancelled, [True])

    def test_local_cancel_interrupts_duckdb_connection(self) -> None:
        worker = LocalSearchWorker("observations.parquet", "", "", 0, 0, 1)
        connection = MagicMock()
        worker._set_connection(connection)

        worker.cancel()

        connection.interrupt.assert_called_once_with()
        self.assertTrue(worker.is_cancelled())


if __name__ == "__main__":
    unittest.main()

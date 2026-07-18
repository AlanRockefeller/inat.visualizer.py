"""Tests for optional local-database startup behavior."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from requests.exceptions import RequestException
from PyQt6.QtCore import Qt

from visualizer import (
    DATABASE_FILE_INFO,
    INatSeasonalVisualizer,
    SPLASH_WINDOW_FLAGS,
    available_database_updates,
    database_download_choice_message,
    missing_database_files,
)


class DatabaseDownloadChoiceTests(unittest.TestCase):
    def test_splash_is_not_a_globally_topmost_window(self) -> None:
        self.assertFalse(SPLASH_WINDOW_FLAGS & Qt.WindowType.WindowStaysOnTopHint)

    def test_startup_dialog_hides_and_then_restores_visible_splash(self) -> None:
        splash = MagicMock()
        splash.isVisible.return_value = True
        window = SimpleNamespace(splash_screen=splash)
        dialog = MagicMock()
        dialog.exec.return_value = 42

        with patch("visualizer.QApplication.processEvents") as process_events:
            result = INatSeasonalVisualizer._exec_startup_dialog(window, dialog)

        self.assertEqual(result, 42)
        dialog.setWindowModality.assert_called_once_with(
            Qt.WindowModality.ApplicationModal
        )
        splash.hide.assert_called_once_with()
        splash.show.assert_called_once_with()
        self.assertEqual(process_events.call_count, 2)

    def test_startup_dialog_restores_splash_if_dialog_raises(self) -> None:
        splash = MagicMock()
        splash.isVisible.return_value = True
        window = SimpleNamespace(splash_screen=splash)
        dialog = MagicMock()
        dialog.exec.side_effect = RuntimeError("dialog failed")

        with (
            patch("visualizer.QApplication.processEvents"),
            self.assertRaisesRegex(RuntimeError, "dialog failed"),
        ):
            INatSeasonalVisualizer._exec_startup_dialog(window, dialog)

        splash.hide.assert_called_once_with()
        splash.show.assert_called_once_with()

    def test_prompt_explains_download_and_api_tradeoffs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_files = missing_database_files(temp_dir)

        message = database_download_choice_message(missing_files)

        self.assertIn("approximately 1 GB", message)
        self.assertIn("Graph with local data", message)
        self.assertIn("disk space", message)
        self.assertIn("API-only mode", message)
        self.assertIn("internet connection", message)
        self.assertIn("rate-limited", message)

    def test_taxonomy_only_message_keeps_local_search_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "observations.parquet").touch()
            missing_files = missing_database_files(temp_dir)

        message = database_download_choice_message(missing_files)

        self.assertIn("approximately 8.3 MB", message)
        self.assertIn("still support 'Graph with local data'", message)
        self.assertIn("higher taxa", message)

    def test_declining_download_continues_in_api_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            window = SimpleNamespace(
                working_dir=temp_dir,
                observations_file=str(Path(temp_dir, "observations.parquet")),
                prompt_for_database_download=MagicMock(return_value=False),
                local_database_available=True,
            )

            with patch("visualizer.requests.get") as get_mock:
                INatSeasonalVisualizer.download_missing_files(window)

        window.prompt_for_database_download.assert_called_once()
        get_mock.assert_not_called()
        self.assertFalse(window.local_database_available)

    def test_declining_taxonomy_keeps_existing_local_database_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations_path = Path(temp_dir, "observations.parquet")
            observations_path.touch()
            window = SimpleNamespace(
                working_dir=temp_dir,
                observations_file=str(observations_path),
                prompt_for_database_download=MagicMock(return_value=False),
                local_database_available=False,
            )

            INatSeasonalVisualizer.download_missing_files(window)

        self.assertTrue(window.local_database_available)

    def test_download_failure_continues_without_local_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            window = SimpleNamespace(
                working_dir=temp_dir,
                observations_file=str(Path(temp_dir, "observations.parquet")),
                prompt_for_database_download=MagicMock(return_value=True),
                local_database_available=True,
                human_readable_size=MagicMock(return_value="1 GB"),
                status_bar=MagicMock(),
                enhanced_progress=MagicMock(),
            )

            with (
                patch(
                    "visualizer.requests.get",
                    side_effect=RuntimeError("network unavailable"),
                ),
                patch("visualizer.QMessageBox.warning") as warning_mock,
                patch("visualizer.QApplication.processEvents"),
            ):
                INatSeasonalVisualizer.download_database_files(
                    window,
                    missing_database_files(temp_dir),
                    replacing_existing=False,
                )

        self.assertFalse(window.local_database_available)
        warning_mock.assert_called_once()

    def test_update_check_is_skipped_for_api_only_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("visualizer.requests.head") as head_mock:
                updates = available_database_updates(temp_dir)

        self.assertEqual(updates, [])
        head_mock.assert_not_called()

    def test_head_finds_installed_database_with_different_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations_path = Path(temp_dir, "observations.parquet")
            observations_path.write_bytes(b"old")
            response = SimpleNamespace(
                status_code=200,
                headers={
                    "content-length": "12",
                    "last-modified": "Thu, 02 Jan 2025 00:00:00 GMT",
                },
                raise_for_status=MagicMock(),
            )

            with patch("visualizer.requests.head", return_value=response) as head_mock:
                updates = available_database_updates(temp_dir)

        self.assertEqual([name for name, _info in updates], ["observations.parquet"])
        self.assertEqual(updates[0][1]["size"], 12)
        call = head_mock.call_args
        self.assertEqual(
            call.args[0], DATABASE_FILE_INFO["observations.parquet"]["url"]
        )
        self.assertNotIn("If-Modified-Since", call.kwargs["headers"])
        self.assertTrue(call.kwargs["allow_redirects"])

    def test_same_size_does_not_offer_update_even_if_server_timestamp_is_newer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = b"same database"
            Path(temp_dir, "observations.parquet").write_bytes(database)
            response = SimpleNamespace(
                status_code=200,
                headers={
                    "content-length": str(len(database)),
                    "last-modified": "Sat, 18 Jul 2026 03:51:22 GMT",
                },
                raise_for_status=MagicMock(),
            )

            with patch("visualizer.requests.head", return_value=response):
                updates = available_database_updates(temp_dir)

        self.assertEqual(updates, [])

    def test_matching_observations_suppress_older_taxonomy_offer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations = b"current observations"
            Path(temp_dir, "observations.parquet").write_bytes(observations)
            Path(temp_dir, "taxonomy.parquet").write_bytes(b"new local taxonomy")
            response = SimpleNamespace(
                status_code=200,
                headers={"content-length": str(len(observations))},
                raise_for_status=MagicMock(),
            )

            with patch("visualizer.requests.head", return_value=response) as head_mock:
                updates = available_database_updates(temp_dir)

        self.assertEqual(updates, [])
        head_mock.assert_called_once()
        self.assertEqual(
            head_mock.call_args.args[0],
            DATABASE_FILE_INFO["observations.parquet"]["url"],
        )

    def test_observations_update_includes_different_companion_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "observations.parquet").write_bytes(b"old observations")
            Path(temp_dir, "taxonomy.parquet").write_bytes(b"old taxonomy")
            observation_response = SimpleNamespace(
                status_code=200,
                headers={"content-length": "100"},
                raise_for_status=MagicMock(),
            )
            taxonomy_response = SimpleNamespace(
                status_code=200,
                headers={"content-length": "50"},
                raise_for_status=MagicMock(),
            )

            with patch(
                "visualizer.requests.head",
                side_effect=[observation_response, taxonomy_response],
            ):
                updates = available_database_updates(temp_dir)

        self.assertEqual(
            [name for name, _info in updates],
            ["observations.parquet", "taxonomy.parquet"],
        )

    def test_different_size_is_offered_without_using_server_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations_path = Path(temp_dir, "observations.parquet")
            observations_path.write_bytes(b"new local DWCA build")
            response = SimpleNamespace(
                status_code=200,
                headers={
                    "content-length": "3",
                    "last-modified": "Sat, 03 May 2025 06:32:58 GMT",
                },
                raise_for_status=MagicMock(),
            )

            with patch("visualizer.requests.head", return_value=response):
                updates = available_database_updates(temp_dir)

        self.assertEqual([name for name, _info in updates], ["observations.parquet"])

    def test_failed_head_check_does_not_interrupt_offline_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "observations.parquet").write_bytes(b"current")

            with patch(
                "visualizer.requests.head",
                side_effect=RequestException("offline"),
            ):
                updates = available_database_updates(temp_dir)

        self.assertEqual(updates, [])

    def test_database_update_check_starts_in_background(self) -> None:
        worker = MagicMock()
        window = SimpleNamespace(
            local_database_available=True,
            database_update_worker=None,
            working_dir="/runtime",
            _offer_database_updates=MagicMock(),
            _on_database_update_worker_done=MagicMock(),
        )

        with patch(
            "visualizer.DatabaseUpdateCheckWorker", return_value=worker
        ) as worker_class:
            INatSeasonalVisualizer.start_database_update_check(window)

        worker_class.assert_called_once_with("/runtime")
        worker.updates_ready.connect.assert_called_once_with(
            window._offer_database_updates
        )
        worker.finished.connect.assert_called_once_with(
            window._on_database_update_worker_done
        )
        worker.start.assert_called_once_with()
        self.assertIs(window.database_update_worker, worker)

    def test_database_update_check_is_not_started_in_api_only_mode(self) -> None:
        window = SimpleNamespace(
            local_database_available=False,
            database_update_worker=None,
            working_dir="/runtime",
        )

        with patch("visualizer.DatabaseUpdateCheckWorker") as worker_class:
            INatSeasonalVisualizer.start_database_update_check(window)

        worker_class.assert_not_called()

    def test_declining_database_update_leaves_files_untouched(self) -> None:
        update = ("observations.parquet", DATABASE_FILE_INFO["observations.parquet"])
        window = SimpleNamespace(
            working_dir="/unused",
            prompt_for_database_update=MagicMock(return_value=False),
            download_database_files=MagicMock(),
        )

        with patch("visualizer.available_database_updates", return_value=[update]):
            INatSeasonalVisualizer.check_for_database_updates(window)

        window.prompt_for_database_update.assert_called_once_with([update])
        window.download_database_files.assert_not_called()

    def test_accepting_database_update_starts_replacement_download(self) -> None:
        update = ("observations.parquet", DATABASE_FILE_INFO["observations.parquet"])
        window = SimpleNamespace(
            working_dir="/unused",
            prompt_for_database_update=MagicMock(return_value=True),
            download_database_files=MagicMock(),
        )

        with patch("visualizer.available_database_updates", return_value=[update]):
            INatSeasonalVisualizer.check_for_database_updates(window)

        window.download_database_files.assert_called_once_with(
            [update], replacing_existing=True
        )

    def test_failed_update_preserves_installed_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations_path = Path(temp_dir, "observations.parquet")
            observations_path.write_bytes(b"old database")
            info = {**DATABASE_FILE_INFO["observations.parquet"], "size": 20}
            response = MagicMock()
            response.headers = {"content-length": "20"}
            response.iter_content.return_value = [b"incomplete"]
            window = SimpleNamespace(
                working_dir=temp_dir,
                observations_file=str(observations_path),
                human_readable_size=MagicMock(return_value="20 B"),
                status_bar=MagicMock(),
                enhanced_progress=MagicMock(),
            )

            with (
                patch("visualizer.requests.get", return_value=response),
                patch("visualizer.QMessageBox.warning") as warning_mock,
                patch("visualizer.QApplication.processEvents"),
            ):
                INatSeasonalVisualizer.download_database_files(
                    window,
                    [("observations.parquet", info)],
                    replacing_existing=True,
                )

            self.assertEqual(observations_path.read_bytes(), b"old database")
            self.assertFalse(Path(f"{observations_path}.part").exists())

        self.assertIn("left unchanged", warning_mock.call_args.args[2])

    def test_successful_update_replaces_file_and_preserves_remote_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            observations_path = Path(temp_dir, "observations.parquet")
            observations_path.write_bytes(b"old database")
            new_database = b"new database contents"
            remote_modified = "Thu, 02 Jan 2025 00:00:00 GMT"
            info = {
                **DATABASE_FILE_INFO["observations.parquet"],
                "size": len(new_database),
            }
            response = MagicMock()
            response.headers = {
                "content-length": str(len(new_database)),
                "last-modified": remote_modified,
            }
            response.iter_content.return_value = [new_database[:5], new_database[5:]]
            window = SimpleNamespace(
                working_dir=temp_dir,
                observations_file=str(observations_path),
                human_readable_size=MagicMock(return_value="21 B"),
                status_bar=MagicMock(),
                enhanced_progress=MagicMock(),
            )

            with (
                patch("visualizer.requests.get", return_value=response),
                patch("visualizer.QApplication.processEvents"),
            ):
                INatSeasonalVisualizer.download_database_files(
                    window,
                    [("observations.parquet", info)],
                    replacing_existing=True,
                )

            self.assertEqual(observations_path.read_bytes(), new_database)
            expected_mtime = datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp()
            self.assertEqual(observations_path.stat().st_mtime, expected_mtime)

    def test_taxonomy_update_invalidates_only_descendant_cache_entries(self) -> None:
        window = SimpleNamespace(
            taxon_cache={
                "Agaricales": 47170,
                "Agaricales_descendants": [47170, 48723],
                "Fungi_descendants": [47170],
            },
            save_taxon_cache=MagicMock(),
        )

        INatSeasonalVisualizer.invalidate_descendant_taxon_cache(window)

        self.assertEqual(window.taxon_cache, {"Agaricales": 47170})
        window.save_taxon_cache.assert_called_once_with()

    def test_api_search_does_not_require_local_taxonomy(self) -> None:
        def text_field(value: str) -> SimpleNamespace:
            return SimpleNamespace(text=lambda: value)

        worker = MagicMock()
        window = SimpleNamespace(
            canvas=object(),
            api_search_worker=None,
            lat_input=text_field("37.7749"),
            lon_input=text_field("-122.4194"),
            radius_input=text_field("25"),
            organism_input=text_field("Agaricales"),
            exclude_input=text_field(""),
            date_from=text_field("2025-01-01"),
            date_to=text_field("2025-12-31"),
            view_combo=SimpleNamespace(currentText=lambda: "Weekly"),
            taxon_cache={"Agaricales": 47170},
            http_cache_file="/runtime/inat_api_cache.db",
            http_cache_max_bytes=128 * 1024 * 1024,
            local_database_available=False,
            enhanced_progress=MagicMock(),
            status_bar=MagicMock(),
            local_search_button=MagicMock(),
            search_button=MagicMock(),
            cancel_search_button=MagicMock(),
            show_url_button=MagicMock(),
            _on_api_search_progress=MagicMock(),
            _on_api_call_completed=MagicMock(),
            _on_api_taxon_resolved=MagicMock(),
            _on_api_search_finished=MagicMock(),
            _on_api_search_failed=MagicMock(),
            _on_api_search_cancelled=MagicMock(),
            _on_api_search_worker_done=MagicMock(),
            show_placeholder=MagicMock(),
        )

        with patch("visualizer.ApiSearchWorker", return_value=worker) as worker_class:
            INatSeasonalVisualizer.search_observations(window)

        (
            params,
            organism,
            exclude,
            taxon_cache,
            http_cache_file,
            http_cache_max_bytes,
        ) = worker_class.call_args.args
        self.assertNotIn("taxon_id", params)
        self.assertEqual(organism, "Agaricales")
        self.assertEqual(exclude, "")
        self.assertEqual(taxon_cache, {"Agaricales": 47170})
        self.assertEqual(http_cache_file, "/runtime/inat_api_cache.db")
        self.assertEqual(http_cache_max_bytes, 128 * 1024 * 1024)
        worker.start.assert_called_once_with()
        window.enhanced_progress.start_progress.assert_called_once_with(
            0, "Starting live iNaturalist search..."
        )
        window.local_search_button.setEnabled.assert_called_once_with(False)
        window.search_button.setEnabled.assert_called_once_with(False)
        window.search_button.setText.assert_called_once_with(
            "Searching live iNat data..."
        )
        window.show_url_button.setEnabled.assert_called_once_with(False)
        window.cancel_search_button.setEnabled.assert_called_once_with(True)

    def test_api_only_local_button_stays_disabled_after_live_search(self) -> None:
        window = SimpleNamespace(
            local_database_available=False,
            local_search_button=MagicMock(),
            search_button=MagicMock(),
            cancel_search_button=MagicMock(),
            show_url_button=MagicMock(),
            api_search_worker=object(),
        )

        INatSeasonalVisualizer._on_api_search_worker_done(window)

        window.local_search_button.setEnabled.assert_called_once_with(False)
        window.search_button.setEnabled.assert_called_once_with(True)
        window.search_button.setText.assert_called_once_with(
            "Graph with live iNat data"
        )
        window.show_url_button.setEnabled.assert_called_once_with(True)
        window.cancel_search_button.setEnabled.assert_called_once_with(False)
        self.assertIsNone(window.api_search_worker)

    def test_cancel_button_requests_live_search_cancellation(self) -> None:
        api_worker = MagicMock()
        api_worker.isRunning.return_value = True
        window = SimpleNamespace(
            api_search_worker=api_worker,
            local_search_worker=None,
            cancel_search_button=MagicMock(),
            status_bar=MagicMock(),
            enhanced_progress=MagicMock(),
        )

        INatSeasonalVisualizer.cancel_search(window)

        api_worker.cancel.assert_called_once_with()
        window.cancel_search_button.setText.assert_called_once_with("Cancelling...")
        window.cancel_search_button.setEnabled.assert_called_once_with(False)
        status = window.status_bar.showMessage.call_args.args[0]
        self.assertIn("current network request", status)

    def test_cancel_button_requests_local_search_cancellation(self) -> None:
        local_worker = MagicMock()
        local_worker.isRunning.return_value = True
        window = SimpleNamespace(
            api_search_worker=None,
            local_search_worker=local_worker,
            cancel_search_button=MagicMock(),
            status_bar=MagicMock(),
            enhanced_progress=MagicMock(),
        )

        INatSeasonalVisualizer.cancel_search(window)

        local_worker.cancel.assert_called_once_with()
        window.status_bar.showMessage.assert_called_once_with(
            "Cancelling local search..."
        )


if __name__ == "__main__":
    unittest.main()

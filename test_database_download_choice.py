"""Tests for optional local-database startup behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from visualizer import (
    INatSeasonalVisualizer,
    database_download_choice_message,
    missing_database_files,
)


class DatabaseDownloadChoiceTests(unittest.TestCase):
    def test_prompt_explains_download_and_api_tradeoffs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_files = missing_database_files(temp_dir)

        message = database_download_choice_message(missing_files)

        self.assertIn("approximately 1 GB", message)
        self.assertIn("fast Local Search", message)
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
        self.assertIn("still support Local Search", message)
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
                INatSeasonalVisualizer.download_missing_files(window)

        self.assertFalse(window.local_database_available)
        warning_mock.assert_called_once()

    def test_api_search_does_not_require_local_taxonomy(self) -> None:
        def text_field(value: str) -> SimpleNamespace:
            return SimpleNamespace(text=lambda: value)

        window = SimpleNamespace(
            canvas=object(),
            lat_input=text_field("37.7749"),
            lon_input=text_field("-122.4194"),
            radius_input=text_field("25"),
            organism_input=text_field("Agaricales"),
            exclude_input=text_field(""),
            date_from=text_field("2025-01-01"),
            date_to=text_field("2025-12-31"),
            view_combo=SimpleNamespace(currentText=lambda: "Weekly"),
            taxonomy_file="/missing/taxonomy.parquet",
            get_taxon_id=MagicMock(return_value=47170),
            get_descendant_taxon_ids=MagicMock(),
            fetch_all_observations=MagicMock(return_value=([], None)),
            show_placeholder=MagicMock(),
            update_status_bar=MagicMock(),
        )

        INatSeasonalVisualizer.search_observations(window)

        window.get_descendant_taxon_ids.assert_not_called()
        window.fetch_all_observations.assert_called_once()
        params = window.fetch_all_observations.call_args.args[0]
        self.assertEqual(params["taxon_id"], 47170)


if __name__ == "__main__":
    unittest.main()

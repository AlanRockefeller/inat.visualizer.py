"""Tests for the main-screen text hierarchy and File menu actions."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from matplotlib.figure import Figure

from visualizer import INatSeasonalVisualizer


class FakeSettings:
    def value(self, _key, default=None):
        return default


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)


class FakeAction:
    def __init__(self, text, _parent) -> None:
        self.text = text
        self.triggered = FakeSignal()


class FakeMenu:
    def __init__(self) -> None:
        self.actions = []

    def addAction(self, action) -> None:
        self.actions.append(action)


class FakeMenuBar:
    def __init__(self) -> None:
        self.file_menu = FakeMenu()

    def addMenu(self, name):
        return self.file_menu if name == "File" else None


class MainScreenLayoutTests(unittest.TestCase):
    def test_settings_and_stats_use_half_size_text(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        harness = SimpleNamespace(
            ax=axis,
            canvas=MagicMock(),
            settings=FakeSettings(),
            graph_font_size=12,
            scale_factor=1.0,
            lat_input=MagicMock(text=lambda: "37.0"),
            lon_input=MagicMock(text=lambda: "-122.0"),
            radius_input=MagicMock(text=lambda: "25"),
            date_from=MagicMock(text=lambda: "2025-01-01"),
            date_to=MagicMock(text=lambda: "2025-12-31"),
            color_input=MagicMock(text=lambda: "#1f77b4"),
            bg_color_input=MagicMock(text=lambda: ""),
            local_database_available=True,
            total_observations=100,
            unique_taxa=20,
            info_bar=MagicMock(),
            update_status_bar=MagicMock(),
            get_contrasting_text_color=MagicMock(return_value="white"),
        )

        INatSeasonalVisualizer.show_placeholder(harness)

        self.assertEqual(len(axis.texts), 2)
        welcome_text, detail_text = axis.texts
        self.assertNotIn("Current Settings:", welcome_text.get_text())
        self.assertIn("Graph with local data", welcome_text.get_text())
        self.assertIn("Graph with live iNat data", welcome_text.get_text())
        self.assertTrue(detail_text.get_text().startswith("Current Settings:"))
        self.assertIn("Local Database Stats:", detail_text.get_text())
        self.assertEqual(welcome_text.get_fontsize(), 12)
        self.assertEqual(detail_text.get_fontsize(), 6)

    def test_fetch_taxon_ids_is_available_from_file_menu(self) -> None:
        menu_bar = FakeMenuBar()
        fetch_taxon_ids = MagicMock()
        harness = SimpleNamespace(
            menuBar=lambda: menu_bar,
            save_settings=MagicMock(),
            fetch_taxon_ids=fetch_taxon_ids,
            export_graph=MagicMock(),
            export_data=MagicMock(),
            close=MagicMock(),
        )

        with patch("visualizer.QAction", FakeAction):
            INatSeasonalVisualizer.init_menu_bar(harness)

        actions = {action.text: action for action in menu_bar.file_menu.actions}
        self.assertIn("Fetch Taxon IDs", actions)
        self.assertEqual(
            actions["Fetch Taxon IDs"].triggered.callbacks,
            [fetch_taxon_ids],
        )


if __name__ == "__main__":
    unittest.main()

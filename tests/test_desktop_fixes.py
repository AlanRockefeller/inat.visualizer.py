import os
import tempfile
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings, Qt

import visualizer
from visualizer import (
    HISTORY_RESULTS_RETAINED,
    INatSeasonalVisualizer,
    fetch_all_observation_pages,
    observation_export_rows,
)


class FakeLineEdit:
    """Rejects non-string values the way PyQt6's QLineEdit.setText does."""

    def __init__(self):
        self._text = ""

    def setText(self, value):
        if not isinstance(value, str):
            raise TypeError(
                f"setText(): argument 1 has unexpected type {type(value).__name__!r}"
            )
        self._text = value

    def text(self):
        return self._text


class FakeItem:
    def __init__(self, label):
        self._text = label
        self._data = {}

    def text(self):
        return self._text

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)


class FakeHistoryList:
    def __init__(self):
        self._items = []
        self._current = None

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def item(self, index):
        return self._items[index]

    def currentItem(self):
        return self._current

    def setCurrentItem(self, item):
        self._current = item

    def setCurrentRow(self, index):
        self._current = self._items[index]


class LoadSettingsTests(unittest.TestCase):
    def test_saved_float_coordinates_load_into_line_edits(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = QSettings(
            os.path.join(tmp.name, "settings.ini"), QSettings.Format.IniFormat
        )
        settings.setValue("latitude", 37.7749)
        settings.setValue("longitude", -122.4194)
        settings.setValue("radius", 50.0)
        settings.sync()
        settings = QSettings(
            os.path.join(tmp.name, "settings.ini"), QSettings.Format.IniFormat
        )

        harness = SimpleNamespace(
            settings=settings,
            default_lat="0.0000",
            default_lon="0.0000",
            default_radius=25,
            lat_input=FakeLineEdit(),
            lon_input=FakeLineEdit(),
            radius_input=FakeLineEdit(),
            color_input=FakeLineEdit(),
            bg_color_input=FakeLineEdit(),
            season_input=MagicMock(),
            grid_combo=MagicMock(),
            rank_combo=MagicMock(),
            min_obs_input=MagicMock(),
        )
        INatSeasonalVisualizer.load_settings(harness)

        self.assertEqual(float(harness.lat_input.text()), 37.7749)
        self.assertEqual(float(harness.lon_input.text()), -122.4194)
        self.assertEqual(float(harness.radius_input.text()), 50.0)


class ExportRowsTests(unittest.TestCase):
    def test_local_rows_use_flat_columns(self):
        rows = observation_export_rows(
            [
                {
                    "id": 1,
                    "decimalLatitude": Decimal("37.775"),
                    "decimalLongitude": Decimal("-122.419"),
                    "eventDate": "2020-03-01",
                    "taxonID": 47170,
                }
            ],
            "Local",
        )
        self.assertEqual(
            rows,
            [
                {
                    "id": 1,
                    "latitude": Decimal("37.775"),
                    "longitude": Decimal("-122.419"),
                    "date": "2020-03-01",
                    "taxonID": 47170,
                }
            ],
        )

    def test_api_rows_read_geojson_and_nested_taxon(self):
        rows = observation_export_rows(
            [
                {
                    "id": 2,
                    "geojson": {"coordinates": [-122.419, 37.775]},
                    "observed_on": "2020-03-01",
                    "taxon": {"id": 47170},
                },
                {"id": 3},
            ],
            "API",
        )
        self.assertEqual(rows[0]["latitude"], 37.775)
        self.assertEqual(rows[0]["longitude"], -122.419)
        self.assertEqual(rows[0]["taxonID"], 47170)
        self.assertEqual(
            rows[1],
            {
                "id": 3,
                "latitude": None,
                "longitude": None,
                "date": None,
                "taxonID": None,
            },
        )


class ResultCapTests(unittest.TestCase):
    def test_stops_at_the_api_window_and_keeps_fetched_pages(self):
        def page(params, **_kwargs):
            start = (params["page"] - 1) * params["per_page"]
            return {
                "results": [{"id": start + i} for i in range(params["per_page"])],
                "total_results": 12000,
            }

        warnings = []
        with patch("visualizer.fetch_observations_page", side_effect=page):
            observations, error = fetch_all_observation_pages(
                {"taxon_id": 1}, warning_callback=warnings.append
            )

        self.assertIsNone(error)
        self.assertEqual(len(observations), 10000)
        self.assertEqual(len(warnings), 1)
        self.assertIn("12,000", warnings[0])

    def test_small_searches_are_unaffected(self):
        responses = [
            {"results": [{"id": 1}], "total_results": 2},
            {"results": [{"id": 2}], "total_results": 2},
        ]
        warnings = []
        with patch("visualizer.fetch_observations_page", side_effect=responses):
            observations, error = fetch_all_observation_pages(
                {"taxon_id": 1}, warning_callback=warnings.append
            )
        self.assertIsNone(error)
        self.assertEqual(len(observations), 2)
        self.assertEqual(warnings, [])


class GraphAbundanceCellTests(unittest.TestCase):
    def test_uses_the_maps_search_context_not_the_form(self):
        harness = SimpleNamespace(
            abundance_view=SimpleNamespace(
                context={
                    "organism": "Amanita",
                    "exclude": "Amanita muscaria",
                    "date_from": "2010-01-01",
                    "date_to": "2020-12-31",
                }
            ),
            organism_input=FakeLineEdit(),
            exclude_input=FakeLineEdit(),
            date_from=FakeLineEdit(),
            date_to=FakeLineEdit(),
            lat_input=FakeLineEdit(),
            lon_input=FakeLineEdit(),
            radius_input=FakeLineEdit(),
            mode_combo=MagicMock(),
            local_search=MagicMock(),
        )
        harness.organism_input.setText("Boletus")  # edited after the map was made

        INatSeasonalVisualizer.graph_abundance_cell(harness, 37.5, -122.5, 56.0)

        self.assertEqual(harness.organism_input.text(), "Amanita")
        self.assertEqual(harness.exclude_input.text(), "Amanita muscaria")
        self.assertEqual(
            (harness.date_from.text(), harness.date_to.text()),
            ("2010-01-01", "2020-12-31"),
        )
        self.assertEqual(
            (harness.lat_input.text(), harness.lon_input.text()),
            ("37.5000", "-122.5000"),
        )
        harness.mode_combo.setCurrentText.assert_called_once_with(
            visualizer.SEASONAL_MODE_TEXT
        )
        harness.local_search.assert_called_once_with()


class HistoryTests(unittest.TestCase):
    def make_harness(self):
        return SimpleNamespace(
            history_list=FakeHistoryList(),
            mode_combo=MagicMock(),
            plot_observations=MagicMock(),
        )

    def add_item(self, harness, label, observations):
        item = FakeItem(label)
        item.setData(
            Qt.ItemDataRole.UserRole,
            {
                "plot_args": {
                    "observations": observations,
                    "lat": 1.0,
                    "lon": 2.0,
                    "radius": 10.0,
                    "organism": label,
                    "date_from": "2020-01-01",
                    "date_to": "2020-12-31",
                    "view": "weekly",
                    "source": "Local",
                }
            },
        )
        harness.history_list.addItem(item)
        return item

    def test_loading_an_item_replots_its_own_observations(self):
        harness = self.make_harness()
        first = self.add_item(harness, "first", [{"id": 1}])
        self.add_item(harness, "second", [{"id": 2}])
        harness.history_list.setCurrentItem(first)

        INatSeasonalVisualizer.load_history_item(harness)

        harness.mode_combo.setCurrentText.assert_called_once_with(
            visualizer.SEASONAL_MODE_TEXT
        )
        kwargs = harness.plot_observations.call_args.kwargs
        self.assertEqual(kwargs["observations"], [{"id": 1}])
        self.assertEqual(kwargs["organism"], "first")
        self.assertFalse(kwargs["record_history"])

    def test_old_items_drop_their_observations(self):
        harness = self.make_harness()
        for index in range(HISTORY_RESULTS_RETAINED + 2):
            self.add_item(harness, f"search {index}", [{"id": index}])

        INatSeasonalVisualizer._trim_history_results(harness)

        kept = [
            harness.history_list.item(i).data(Qt.ItemDataRole.UserRole)["plot_args"][
                "observations"
            ]
            is not None
            for i in range(harness.history_list.count())
        ]
        self.assertEqual(kept, [False, False] + [True] * HISTORY_RESULTS_RETAINED)

        harness.history_list.setCurrentRow(0)
        with patch("visualizer.QMessageBox.information") as info:
            INatSeasonalVisualizer.load_history_item(harness)
        info.assert_called_once()
        harness.plot_observations.assert_not_called()


if __name__ == "__main__":
    unittest.main()

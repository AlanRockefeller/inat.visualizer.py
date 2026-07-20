"""Tests for map-selection geometry and the redesigned picker workflow."""

import math
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from visualizer import (
    INatSeasonalVisualizer,
    QDialog,
    clamp_latitude_limits,
    geodesic_circle_coordinates,
    geodesic_destination,
    geodesic_distance_km,
    longitude_near_reference,
    normalize_longitude,
    radius_view_limits,
)


class MapGeometryTests(unittest.TestCase):
    def test_latitude_limits_shift_inside_mercator_range(self) -> None:
        self.assertEqual(
            clamp_latitude_limits(-90.0, -80.0),
            (-85.05112878, -75.05112878),
        )
        self.assertEqual(
            clamp_latitude_limits(80.0, 90.0),
            (75.05112878, 85.05112878),
        )

    def test_haversine_delegates_to_shared_geodesic_distance(self) -> None:
        expected = geodesic_distance_km(10.0, 20.0, -30.0, 40.0)

        self.assertEqual(
            INatSeasonalVisualizer.haversine(None, 10.0, 20.0, -30.0, 40.0),
            expected,
        )

    def test_longitudes_normalize_and_unwrap_near_reference(self) -> None:
        self.assertAlmostEqual(normalize_longitude(181.0), -179.0)
        self.assertAlmostEqual(normalize_longitude(-181.0), 179.0)
        self.assertAlmostEqual(longitude_near_reference(-179.0, 181.0), 181.0)

    def test_destination_round_trips_through_distance(self) -> None:
        lat, lon = geodesic_destination(63.0, 10.0, 250.0, 73.0)

        self.assertAlmostEqual(
            geodesic_distance_km(63.0, 10.0, lat, lon), 250.0, places=6
        )

    def test_circle_is_geodesic_and_continuous_across_antimeridian(self) -> None:
        lons, lats = geodesic_circle_coordinates(10.0, 179.8, 100.0, 73)

        distances = [
            geodesic_distance_km(10.0, 179.8, lat, lon)
            for lon, lat in zip(lons, lats, strict=True)
        ]
        self.assertTrue(
            all(math.isclose(value, 100.0, abs_tol=1e-6) for value in distances)
        )
        self.assertGreater(max(lons), 180.0)
        self.assertLess(max(abs(b - a) for a, b in zip(lons, lons[1:])), 1.0)

    def test_radius_view_limits_include_circle_and_clamp_mercator_latitude(
        self,
    ) -> None:
        lons, lats = geodesic_circle_coordinates(84.5, -40.0, 100.0)
        west, east, south, north = radius_view_limits(84.5, -40.0, 100.0)

        self.assertLess(west, min(lons))
        self.assertGreater(east, max(lons))
        self.assertLessEqual(south, min(lats))
        self.assertLessEqual(north, 85.05112878)

    def test_pole_enclosing_circle_is_capped_without_longitude_jumps(self) -> None:
        lons, lats = geodesic_circle_coordinates(85.0, 20.0, 1000.0)
        west, east, south, north = radius_view_limits(85.0, 20.0, 1000.0)

        self.assertLessEqual(
            max(abs(next_lon - lon) for lon, next_lon in zip(lons, lons[1:])),
            180.0,
        )
        self.assertEqual((west, east), (-160.0, 200.0))
        self.assertGreaterEqual(south, -85.05112878)
        self.assertEqual(north, 85.05112878)
        self.assertTrue(all(-85.05112878 <= lat <= 85.05112878 for lat in lats))


class MapDialogWorkflowTests(unittest.TestCase):
    def test_dialog_controls_and_mouse_workflow_offscreen(self) -> None:
        script = textwrap.dedent("""
            import io
            import logging
            import math
            from types import SimpleNamespace
            from PyQt6.QtWidgets import QApplication, QPushButton
            import visualizer

            class FakeSignal:
                def connect(self, callback):
                    pass

            class FakeTileLoaderWorker:
                def __init__(self, _working_dir):
                    self.view_ready = FakeSignal()
                    self.network_error = FakeSignal()
                    self.network_recovered = FakeSignal()
                    self.retry_complete = FakeSignal()
                    self.tiles_skipped = FakeSignal()
                    self.running = False
                    self.requests = []

                def start(self):
                    self.running = True

                def stop(self):
                    self.running = False

                def isRunning(self):
                    return self.running

                def request_view(self, *args):
                    self.requests.append(args)

            visualizer.TileLoaderWorker = FakeTileLoaderWorker
            log_stream = io.StringIO()
            log_handler = logging.StreamHandler(log_stream)
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)
            root_logger.setLevel(logging.DEBUG)
            app = QApplication.instance() or QApplication([])
            dialog = visualizer.MapDialog(None, 37.7749, -122.4194, 10.5)
            assert dialog.worker.requests == []
            dialog.show()
            app.processEvents()
            assert dialog.worker.requests

            assert dialog.windowTitle() == "Choose Search Area"
            assert math.isclose(dialog.radius, 10.5)
            assert math.isclose(dialog.radius_spinbox.value(), 10.5)
            assert not dialog.ax.axison
            assert any(
                button.text() == "Use selected area"
                for button in dialog.findChildren(QPushButton)
            )
            dialog.on_network_error("Network error: internal transport details")
            assert "internal transport details" not in dialog.status_label.text()
            assert "still adjust the selected area" in dialog.status_label.text()
            dialog.on_network_recovered()

            circle_lons, circle_lats = visualizer.geodesic_circle_coordinates(
                dialog.lat, dialog._display_longitude(), dialog.radius
            )
            x0, x1 = dialog.ax.get_xlim()
            y0, y1 = dialog.ax.get_ylim()
            assert x0 <= min(circle_lons) <= max(circle_lons) <= x1
            assert y0 <= min(circle_lats) <= max(circle_lats) <= y1

            original_span = x1 - x0
            dialog.radius_spinbox.setValue(5.0)
            assert math.isclose(dialog.ax.get_xlim()[1] - dialog.ax.get_xlim()[0], original_span)
            dialog.radius_spinbox.setValue(80.0)
            assert dialog.ax.get_xlim()[1] - dialog.ax.get_xlim()[0] > original_span

            old_lat, old_lon = dialog.lat, dialog.lon
            start_x, start_y = dialog.ax.transData.transform((-122.4194, 37.7749))
            press = SimpleNamespace(
                inaxes=dialog.ax, xdata=-122.4194, ydata=37.7749,
                x=start_x, y=start_y, button=1,
            )
            move = SimpleNamespace(
                inaxes=dialog.ax, xdata=-122.3, ydata=37.8,
                x=start_x + 30, y=start_y + 20, button=1,
            )
            dialog.on_mouse_press(press)
            dialog.on_mouse_move(move)
            dialog.on_mouse_release(move)
            assert math.isclose(dialog.lat, old_lat)
            assert math.isclose(dialog.lon, old_lon)

            click_x, click_y = dialog.ax.transData.transform((-122.2, 37.9))
            click = SimpleNamespace(
                inaxes=dialog.ax, xdata=-122.2, ydata=37.9,
                x=click_x, y=click_y, button=1,
            )
            dialog.on_mouse_press(click)
            dialog.on_mouse_release(click)
            assert math.isclose(dialog.lat, 37.9)
            assert math.isclose(dialog.lon, -122.2)

            dialog.fit_selected_area()
            handle_lon = float(dialog.radius_handle.get_xdata()[0])
            handle_lat = float(dialog.radius_handle.get_ydata()[0])
            handle_x, handle_y = dialog.ax.transData.transform((handle_lon, handle_lat))
            handle_press = SimpleNamespace(
                inaxes=dialog.ax, xdata=handle_lon, ydata=handle_lat,
                x=handle_x, y=handle_y, button=1,
            )
            target_lat, target_lon = visualizer.geodesic_destination(
                dialog.lat, dialog.lon, 25.0, 90.0
            )
            target_x, target_y = dialog.ax.transData.transform((target_lon, target_lat))
            handle_move = SimpleNamespace(
                inaxes=dialog.ax, xdata=target_lon, ydata=target_lat,
                x=target_x, y=target_y, button=1,
            )
            dialog.on_mouse_press(handle_press)
            assert dialog._drag_mode == "radius"
            dialog.on_mouse_move(handle_move)
            dialog.on_mouse_release(handle_move)
            assert math.isclose(dialog.radius, 25.0, abs_tol=0.2)
            assert math.isclose(dialog.radius_spinbox.value(), 25.0, abs_tol=0.2)

            dialog.place_search_results = [{
                "lat": 40.123456, "lon": -73.987654, "bounds": None,
                "display_name": "Private Place Name", "admin_level": 20,
            }]
            dialog.select_place_search_result(0)
            dialog.accept()
            log_output = log_stream.getvalue()
            root_logger.removeHandler(log_handler)
            assert "Map picker opened:" in log_output
            assert "Map place selected:" in log_output
            assert "Map picker accepted:" in log_output
            assert "Private Place Name" not in log_output
            assert "40.123456" not in log_output
            assert "-73.987654" not in log_output

            untouched = visualizer.MapDialog(None, 87.0, 25.0, 2000.0)
            assert math.isclose(untouched.lat, visualizer.WEB_MERCATOR_MAX_LATITUDE)
            assert math.isclose(untouched.radius, visualizer.MAP_RADIUS_MAX_KM)
            untouched.accept()
            assert math.isclose(untouched.lat, 87.0)
            assert math.isclose(untouched.lon, 25.0)
            assert math.isclose(untouched.radius, 2000.0)

            changed = visualizer.MapDialog(None, 87.0, 25.0, 2000.0)
            changed.radius_spinbox.setValue(500.0)
            changed.accept()
            assert math.isclose(changed.lat, visualizer.WEB_MERCATOR_MAX_LATITUDE)
            assert math.isclose(changed.radius, 500.0)
            """)
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_main_form_commits_dialog_result_only_after_acceptance(self) -> None:
        harness = SimpleNamespace(
            lat_input=MagicMock(text=lambda: "37.0"),
            lon_input=MagicMock(text=lambda: "-122.0"),
            radius_input=MagicMock(text=lambda: "10.5"),
            default_lat="0",
            default_lon="0",
            default_radius=10,
        )
        accepted_dialog = SimpleNamespace(
            lat=40.1234567,
            lon=-73.9876543,
            radius=12.25,
            exec=MagicMock(return_value=QDialog.DialogCode.Accepted),
        )

        with patch("visualizer.MapDialog", return_value=accepted_dialog):
            INatSeasonalVisualizer.open_map_dialog(harness)

        harness.lat_input.setText.assert_called_once_with("40.123457")
        harness.lon_input.setText.assert_called_once_with("-73.987654")
        harness.radius_input.setText.assert_called_once_with("12.2")

        harness.lat_input.reset_mock()
        harness.lon_input.reset_mock()
        harness.radius_input.reset_mock()
        cancelled_dialog = SimpleNamespace(
            lat=1.0,
            lon=2.0,
            radius=3.0,
            exec=MagicMock(return_value=QDialog.DialogCode.Rejected),
        )
        with patch("visualizer.MapDialog", return_value=cancelled_dialog):
            INatSeasonalVisualizer.open_map_dialog(harness)

        harness.lat_input.setText.assert_not_called()
        harness.lon_input.setText.assert_not_called()
        harness.radius_input.setText.assert_not_called()


if __name__ == "__main__":
    unittest.main()

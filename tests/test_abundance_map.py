"""Tests for abundance-map ranking, geometry, tiles, and the map view widget."""

import hashlib
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visualizer import (
    OSM_BLOCKED_TILE_MD5,
    DiskTileCache,
    abundance_cell_radius_km,
    abundance_grid_array,
    geodesic_distance_km,
    is_blocked_tile_image,
    mercator_rows_to_equirectangular,
    most_specific_standard_place,
    rank_abundance_cells,
)


def cell(lat, lon, count, total, deg=1.0):
    return {
        "cell_lat": lat,
        "cell_lon": lon,
        "cell_deg": deg,
        "count": count,
        "total": total,
        "share": count / total,
        "centroid_lat": lat + deg / 2,
        "centroid_lon": lon + deg / 2,
    }


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.cells = [
            cell(47, -123, 1000, 50000),  # busy city: most obs, low share
            cell(50, 22, 90, 500),  # rural hotspot: high share
            cell(10, 10, 2, 3),  # tiny sample: highest share
        ]

    def test_count_ranking(self):
        ranked = rank_abundance_cells(self.cells, "count", 10)
        self.assertEqual([c["count"] for c in ranked], [1000, 90, 2])

    def test_share_ranking_drops_cells_below_minimum(self):
        ranked = rank_abundance_cells(self.cells, "share", 10)
        self.assertEqual([c["count"] for c in ranked], [90, 1000])

    def test_limit(self):
        self.assertEqual(len(rank_abundance_cells(self.cells, "count", 1, 2)), 2)


class GridTests(unittest.TestCase):
    def test_cells_land_in_expected_rows_and_columns(self):
        grid = abundance_grid_array(
            [cell(-90, -180, 5, 10), cell(89, 179, 7, 7), cell(37, -123, 3, 6)],
            1.0,
            "count",
        )
        self.assertEqual(grid.shape, (180, 360))
        self.assertEqual(grid[0, 0], 5)
        self.assertEqual(grid[179, 359], 7)
        self.assertEqual(grid[127, 57], 3)
        self.assertEqual(int(np.isfinite(grid).sum()), 3)

    def test_share_metric_is_percent(self):
        grid = abundance_grid_array([cell(0, 0, 1, 4)], 0.5, "share")
        self.assertEqual(grid.shape, (360, 720))
        self.assertAlmostEqual(grid[180, 360], 25.0)

    def test_longitude_180_wraps_to_first_column(self):
        grid = abundance_grid_array([cell(0, 180, 1, 1)], 1.0, "count")
        self.assertEqual(grid[90, 0], 1)

    def test_radius_covers_whole_cell(self):
        c = cell(47, -123, 1, 1)
        radius = abundance_cell_radius_km(c)
        corner = geodesic_distance_km(47.5, -122.5, 47.0, -123.0)
        self.assertGreaterEqual(radius, corner)
        self.assertLess(radius, corner + 2)


class PlaceNameTests(unittest.TestCase):
    def test_picks_most_specific_admin_place(self):
        response = {
            "results": {
                "standard": [
                    {"display_name": "North America", "admin_level": -10},
                    {"display_name": "Washington, US", "admin_level": 10},
                    {"display_name": "King County, US, WA", "admin_level": 20},
                    {"display_name": "Some Park", "admin_level": None},
                ]
            }
        }
        self.assertEqual(most_specific_standard_place(response), "King County, US, WA")

    def test_no_places(self):
        self.assertIsNone(most_specific_standard_place({"results": {"standard": []}}))
        self.assertIsNone(most_specific_standard_place({}))


class TileTests(unittest.TestCase):
    def test_blocked_tile_detection(self):
        self.assertFalse(is_blocked_tile_image(b"not the blocked tile"))
        self.assertEqual(len(OSM_BLOCKED_TILE_MD5), 1)

    def test_disk_cache_evicts_blocked_tile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskTileCache(tmpdir)
            path = cache.get_path(3, 0, 0)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"placeholder")
            blocked_hash = hashlib.md5(b"placeholder").hexdigest()
            import visualizer

            original = visualizer.OSM_BLOCKED_TILE_MD5
            visualizer.OSM_BLOCKED_TILE_MD5 = frozenset({blocked_hash})
            try:
                self.assertIsNone(cache.get_tile(3, 0, 0))
            finally:
                visualizer.OSM_BLOCKED_TILE_MD5 = original
            self.assertFalse(Path(path).exists())

    def test_mercator_rows_resampled_to_equal_latitude(self):
        # Zoom 1, tile rows 0-1: the whole Mercator world, 512 px tall.
        height = 512
        # Encode each source row number as (high byte, low byte, 0).
        rows = np.repeat(np.arange(height)[:, None], 4, axis=1)
        pixels = np.stack([rows // 256, rows % 256, np.zeros_like(rows)], axis=-1)
        image = Image.fromarray(pixels.astype(np.uint8))
        limit = 85.05112878
        out = np.asarray(mercator_rows_to_equirectangular(image, 0, 1, -limit, limit))
        self.assertEqual(out.shape, (height, 4, 3))
        source_row = out[:, 0, 0].astype(int) * 256 + out[:, 0, 1].astype(int)
        # The equator is the Mercator midpoint and the lat-space midpoint.
        self.assertAlmostEqual(int(source_row[height // 2]), height // 2, delta=1)
        # 60°N sits ~70% of the way up in latitude but only ~64% in Mercator,
        # so the output row there must sample from the Mercator position.
        row_60n = int(round((limit - 60.0) / (2 * limit) * height))
        mercator_y = (1 - math.asinh(math.tan(math.radians(60))) / math.pi) / 2 * height
        self.assertAlmostEqual(int(source_row[row_60n]), mercator_y, delta=2)


class AbundanceMapViewTests(unittest.TestCase):
    def test_view_ranks_selects_and_requests_graph_offscreen(self) -> None:
        script = textwrap.dedent("""
            import math
            from PyQt6.QtWidgets import QApplication
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

            lookups = []

            class FakePlaceNameWorker:
                def __init__(self, pending, _cache):
                    lookups.append(pending)
                    self.name_ready = FakeSignal()
                    self.finished = FakeSignal()
                def start(self):
                    pass
                def cancel(self):
                    pass
                def wait(self, *_args):
                    return True

            visualizer.TileLoaderWorker = FakeTileLoaderWorker
            visualizer.PlaceNameWorker = FakePlaceNameWorker
            app = QApplication.instance() or QApplication([])

            def cell(lat, lon, count, total):
                return {
                    "cell_lat": lat, "cell_lon": lon, "cell_deg": 1.0,
                    "count": count, "total": total, "share": count / total,
                    "centroid_lat": lat + 0.5, "centroid_lon": lon + 0.5,
                }

            view = visualizer.AbundanceMapView(None)
            view.resize(1200, 800)
            view.show()
            app.processEvents()
            cells = [cell(47, -123, 1000, 50000), cell(50, 22, 90, 500), cell(10, 10, 2, 3)]
            view.set_results(cells, 1.0, "count", 10, {
                "organism": "Amanita muscaria", "season_label": "Oct 1 – Nov 30",
            })
            assert view.table.rowCount() == 3
            assert view.table.item(0, 2).text() == "1,000"
            assert len(view._heatmap_images) == 3
            assert lookups and len(lookups[-1]) == 3
            assert view.worker.requests

            view.set_ranking("share", 10)
            assert view.table.rowCount() == 2
            assert view.table.item(0, 2).text() == "90"
            view.set_ranking("count", 10)

            import numpy as np
            from matplotlib.colors import to_hex
            tile = np.zeros((4, 4, 3), dtype=np.uint8)
            view.current_job_id = 1
            view.on_view_ready(1, tile, (-180.0, 180.0, -85.0, 85.0))
            tile_images = [i for i in view.ax.images if i not in view._heatmap_images]
            assert len(tile_images) == 1
            assert view.map_toggle_button.text() == "Hide map"

            requests_before = len(view.worker.requests)
            view.map_toggle_button.click()
            assert not view.map_visible
            assert view.map_toggle_button.text() == "Show map"
            assert not tile_images[0].get_visible()
            assert not view._map_attribution.get_visible()
            assert to_hex(view.figure.get_facecolor()) == "#000000"
            assert all(i.get_alpha() == 1.0 for i in view._heatmap_images)
            view.request_tiles_for_current_view()
            view.on_view_ready(1, tile, (-180.0, 180.0, -85.0, 85.0))
            assert len(view.worker.requests) == requests_before
            assert len([i for i in view.ax.images if i not in view._heatmap_images]) == 1
            view.set_ranking("share", 10)
            view.set_ranking("count", 10)
            assert all(i.get_alpha() == 1.0 for i in view._heatmap_images)

            view.map_toggle_button.click()
            assert view.map_visible
            assert tile_images[0].get_visible()
            assert view._map_attribution.get_visible()
            assert to_hex(view.figure.get_facecolor()) != "#000000"
            assert len(view.worker.requests) == requests_before + 1

            view._on_place_name(visualizer.abundance_cell_key(cells[0]), "King County, US, WA")
            assert view.table.item(0, 1).text().startswith("King County")

            view._on_map_click(47.4, -122.2)
            assert view.selected_cell() is cells[0]
            assert view._highlight is not None

            emitted = []
            view.graph_requested.connect(lambda *args: emitted.append(args))
            view.request_graph()
            lat, lon, radius = emitted[0]
            assert math.isclose(lat, 47.5) and math.isclose(lon, -122.5)
            assert radius == visualizer.abundance_cell_radius_km(cells[0])
            view.shutdown()
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


if __name__ == "__main__":
    unittest.main()

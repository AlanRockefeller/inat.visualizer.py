import unittest
import time
import os
from threading import Event

# Must be set before PyQt6 initializes
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication
from visualizer import TileLoaderWorker

print("Starting script...")
try:
    # We need a QCoreApplication for signals to work between threads correctly
    app = QApplication.instance() or QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    print("QCoreApplication created.")
except Exception as e:
    raise RuntimeError(f"Failed to create QCoreApplication: {e}") from e


class MockResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class TestWorkerSignals(unittest.TestCase):
    def setUp(self):
        self.worker = TileLoaderWorker(".")
        self.worker.disk_cache.put_tile = MagicMock()
        # Clean up any state
        self.worker._skipped_tiles.clear()
        self.worker._network_suspended_until = 0.0
        self.worker._notified_error = False

    def tearDown(self):
        self.worker.stop()

    def test_view_ready_not_emitted_if_no_tiles(self):
        """Regression test: view_ready should not be emitted if zero tiles were loaded."""
        session = MagicMock()
        session.get.return_value = MockResponse(404)

        # Mock disk cache to miss
        self.worker.disk_cache.get_tile = MagicMock(return_value=None)
        self.worker.disk_cache.get_path = MagicMock()
        self.worker.disk_cache.get_path.return_value.exists.return_value = False

        view_ready_emitted = False
        failure_messages = []

        def on_view_ready(*args):
            nonlocal view_ready_emitted
            view_ready_emitted = True

        self.worker.view_ready.connect(on_view_ready)
        self.worker.network_error.connect(failure_messages.append)

        # Call _process_job directly for synchronous test of the logic
        with self.assertLogs(level="DEBUG") as captured_logs:
            self.worker._process_job(session, 1, 0, -180, 180, -85, 85)

        self.assertFalse(
            view_ready_emitted,
            "view_ready should not be emitted when no tiles are loaded",
        )
        self.assertEqual(
            failure_messages,
            ["Map tiles are unavailable for the selected view."],
        )
        map_logs = [
            message for message in captured_logs.output if "Map tile view:" in message
        ]
        self.assertEqual(len(map_logs), 1)
        self.assertIn("status=unavailable", map_logs[0])
        self.assertNotIn("cache hit", "\n".join(captured_logs.output))

    def test_corrupt_http_200_tile_fails_without_poisoning_the_worker(self):
        session = MagicMock()
        session.get.side_effect = [
            MockResponse(200, b"<html>captive portal</html>"),
            MockResponse(200, self._png_bytes()),
        ]
        self.worker.disk_cache.get_tile = MagicMock(return_value=None)
        self.worker.disk_cache.get_path = MagicMock()
        self.worker.disk_cache.get_path.return_value.exists.return_value = False
        ready_jobs = []
        recovered = []
        self.worker.view_ready.connect(
            lambda job_id, _image, _extent: ready_jobs.append(job_id)
        )
        self.worker.network_recovered.connect(lambda: recovered.append(True))

        with self.assertLogs(level="WARNING") as captured_logs:
            self.worker._process_job(session, 1, 0, 0, 1, 0, 1)
        self.worker._process_job(session, 2, 0, 0, 1, 0, 1)

        self.assertIn("error_type=UnidentifiedImageError", "\n".join(captured_logs.output))
        self.assertEqual(ready_jobs, [2])
        self.assertEqual(recovered, [True])
        self.worker.disk_cache.put_tile.assert_called_once()

    def test_run_continues_after_unexpected_job_failure(self):
        first_job_seen = Event()
        second_job_seen = Event()

        def process_job(_session, job_id, *_bounds):
            if job_id == 1:
                first_job_seen.set()
                raise RuntimeError("unexpected failure")
            second_job_seen.set()

        with patch.object(self.worker, "_process_job", side_effect=process_job):
            self.worker.start()
            self.worker.request_view(1, 0, 0, 1, 0, 1)
            self.assertTrue(first_job_seen.wait(2.0))
            self.worker.request_view(2, 0, 0, 1, 0, 1)
            self.assertTrue(second_job_seen.wait(2.0))

    def test_corrupt_retry_tile_remains_skipped_and_is_not_cached(self):
        session = MagicMock()
        session.get.return_value = MockResponse(200, b"not an image")
        tile_key = (1, 0, 0)
        self.worker._skipped_tiles.add(tile_key)

        with self.assertLogs(level="WARNING"):
            recovered = self.worker._retry_skipped_tiles(session)

        self.assertFalse(recovered)
        self.assertIn(tile_key, self.worker._skipped_tiles)
        self.worker.disk_cache.put_tile.assert_not_called()

    @staticmethod
    def _png_bytes():
        tile = Image.new("RGB", (256, 256), color="red")
        buffer = BytesIO()
        tile.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_tiles_skipped_emitted_on_suspension(self):
        """Verify tiles_skipped is emitted when network is suspended."""
        session = MagicMock()
        self.worker._network_suspended_until = time.time() + 10.0

        # Mock disk cache to miss
        self.worker.disk_cache.get_tile = MagicMock(return_value=None)
        self.worker.disk_cache.get_path = MagicMock()
        self.worker.disk_cache.get_path.return_value.exists.return_value = False

        tiles_skipped_emitted = False

        def on_tiles_skipped():
            nonlocal tiles_skipped_emitted
            tiles_skipped_emitted = True

        self.worker.tiles_skipped.connect(on_tiles_skipped)

        self.worker._process_job(session, 1, 0, -180, 180, -85, 85)

        self.assertTrue(
            tiles_skipped_emitted,
            "tiles_skipped should be emitted when tiles are skipped due to suspension",
        )
        self.assertEqual(len(self.worker._skipped_tiles), 1)

    def test_retry_flow_and_signals(self):
        """Test the full signal flow: error -> tiles_skipped -> network_recovered -> retry_complete."""
        print("Starting test_retry_flow_and_signals...")
        session = MagicMock()

        # 1. First attempt fails with 429
        session.get.return_value = MockResponse(429)
        self.worker.disk_cache.get_tile = MagicMock(return_value=None)
        self.worker.disk_cache.get_path = MagicMock()
        self.worker.disk_cache.get_path.return_value.exists.return_value = False

        signals = {
            "network_error": False,
            "tiles_skipped": False,
            "network_recovered": False,
            "retry_complete": False,
        }

        self.worker.network_error.connect(
            lambda msg: signals.__setitem__("network_error", True)
        )
        self.worker.tiles_skipped.connect(
            lambda: signals.__setitem__("tiles_skipped", True)
        )
        self.worker.network_recovered.connect(
            lambda: signals.__setitem__("network_recovered", True)
        )
        self.worker.retry_complete.connect(
            lambda: signals.__setitem__("retry_complete", True)
        )

        # First pass with zoom 1 (4 tiles)
        print("First pass of _process_job...")
        self.worker._process_job(session, 1, 1, -180, 180, -85, 85)
        print("First pass finished.")

        self.assertTrue(signals["network_error"])
        self.assertTrue(signals["tiles_skipped"])
        self.assertTrue(self.worker._notified_error)
        self.assertEqual(len(self.worker._skipped_tiles), 4)

        # 2. Reset suspension and mock success
        self.worker._network_suspended_until = 0.0

        # Mock valid PNG for Image.open
        red_tile = Image.new("RGB", (256, 256), color="red")
        buf = BytesIO()
        red_tile.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        session.get.return_value = MockResponse(200, img_bytes)

        # We manually add a tile to _skipped_tiles that is NOT in the current job
        # so that _retry_skipped_tiles has something to do even if the main loop finishes.
        # But actually _process_job will call _retry_skipped_tiles with WHATEVER is still skipped.
        # If we have 4 tiles, the main loop will fetch them one by one.
        # The first success will trigger _retry_skipped_tiles for the OTHER THREE.

        # Second pass (triggers recovery and retry)
        # Add a tile that is NOT in the job (1, 1, -180, 180, -85, 85)
        # Job at zoom 1 covers (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        self.worker._skipped_tiles.add((10, 0, 0))
        print("Second pass of _process_job...")
        self.worker._process_job(session, 2, 1, -180, 180, -85, 85)
        print("Second pass finished.")

        self.assertTrue(signals["network_recovered"])
        self.assertTrue(signals["retry_complete"])
        self.assertFalse(self.worker._notified_error)
        self.assertEqual(len(self.worker._skipped_tiles), 0)


if __name__ == "__main__":
    print("Running unittest.main()...")
    unittest.main(verbosity=2)

"""Tests for TileLoaderWorker retry scheduling logic.

These tests verify that tiles_skipped is always emitted when skipped tiles
remain after _retry_skipped_tiles, ensuring the retry timer is re-scheduled.
"""

import time
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class NetworkError(Exception):
    """Stand-in for requests.RequestException in tests."""
    pass


class TileRetryTestHarness:
    """Extracts the retry-scheduling logic from TileLoaderWorker for testing.

    This duplicates the _retry_skipped_tiles control flow without requiring
    Qt, PyQt6, or a running event loop.
    """

    # The exception type to catch (mirroring RequestException)
    NetworkException = NetworkError

    def __init__(self):
        self._skipped_tiles = set()
        self._network_suspended_until = 0.0
        self._notified_error = False
        self.last_network_request_time = 0.0
        self.ram_cache = OrderedDict()
        self._pending_job = None

        # Track signal emissions
        self.tiles_skipped_count = 0
        self.network_error_messages = []

    def tiles_skipped_emit(self):
        self.tiles_skipped_count += 1

    def network_error_emit(self, msg):
        self.network_error_messages.append(msg)

    def retry_skipped_tiles(self, session):
        """Mirrors TileLoaderWorker._retry_skipped_tiles with the fix applied."""
        skipped = sorted(self._skipped_tiles)
        recovered_any = False
        for tile_key in skipped:
            if self._pending_job is not None:
                return recovered_any

            zoom, wrapped_x, y = tile_key
            if tile_key in self.ram_cache:
                self._skipped_tiles.discard(tile_key)
                continue

            if time.time() < self._network_suspended_until:
                break

            now = time.time()
            elapsed = now - self.last_network_request_time
            if elapsed < 0.15:
                time.sleep(0.15 - elapsed)
            self.last_network_request_time = time.time()

            url = f"https://tile.openstreetmap.org/{zoom}/{wrapped_x}/{y}.png"
            try:
                resp = session.get(url, timeout=3.0)
                if resp.status_code == 200:
                    self._skipped_tiles.discard(tile_key)
                    recovered_any = True
                else:
                    if resp.status_code in (403, 429, 418, 408, 500, 502, 503, 504):
                        if not self._notified_error:
                            self._notified_error = True
                            self.network_error_emit(f"HTTP {resp.status_code}")
                        self._network_suspended_until = time.time() + 15.0
                    break
            except self.NetworkException as e:
                if not self._notified_error:
                    self._notified_error = True
                    self.network_error_emit(f"Network error: {e!s}")
                self._network_suspended_until = time.time() + 15.0
                break

        still_skipped = len(self._skipped_tiles) > 0
        if still_skipped:
            self.tiles_skipped_emit()

        return recovered_any


class TestRetryScheduling(unittest.TestCase):

    def test_retry_failure_emits_tiles_skipped(self):
        """When _retry_skipped_tiles fails, tiles_skipped must be emitted."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0), (1, 0, 1), (1, 1, 0)}

        session = MagicMock()
        session.get.return_value = FakeResponse(429)

        result = h.retry_skipped_tiles(session)

        self.assertFalse(result)
        self.assertEqual(h.tiles_skipped_count, 1,
                         "tiles_skipped must be emitted when retry fails with remaining tiles")
        self.assertTrue(len(h._skipped_tiles) > 0)

    def test_retry_success_no_tiles_skipped_emitted(self):
        """When all skipped tiles are recovered, tiles_skipped should NOT be emitted."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0)}

        session = MagicMock()
        session.get.return_value = FakeResponse(200, b"\x89PNG fake")

        result = h.retry_skipped_tiles(session)

        self.assertTrue(result)
        self.assertEqual(h.tiles_skipped_count, 0,
                         "tiles_skipped should not be emitted when all tiles recovered")
        self.assertEqual(len(h._skipped_tiles), 0)

    def test_partial_recovery_emits_tiles_skipped(self):
        """When some tiles recover but others fail, tiles_skipped must be emitted."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0), (1, 0, 1)}

        responses = iter([FakeResponse(200, b"\x89PNG fake"), FakeResponse(503)])
        session = MagicMock()
        session.get.side_effect = lambda *a, **kw: next(responses)

        result = h.retry_skipped_tiles(session)

        self.assertTrue(result)
        self.assertEqual(h.tiles_skipped_count, 1,
                         "tiles_skipped must be emitted for remaining failed tiles")

    def test_network_exception_emits_tiles_skipped(self):
        """Network exception during retry must still emit tiles_skipped."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0), (1, 0, 1)}

        session = MagicMock()
        session.get.side_effect = NetworkError("connection refused")

        result = h.retry_skipped_tiles(session)

        self.assertFalse(result)
        self.assertEqual(h.tiles_skipped_count, 1)
        self.assertGreater(h._network_suspended_until, time.time())

    def test_suspension_respected_during_retry(self):
        """Retry should not make network requests during active suspension."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0)}
        h._network_suspended_until = time.time() + 60  # far future

        session = MagicMock()
        result = h.retry_skipped_tiles(session)

        self.assertFalse(result)
        session.get.assert_not_called()
        self.assertEqual(h.tiles_skipped_count, 1,
                         "tiles_skipped must be emitted even when skipping due to suspension")

    def test_cached_tiles_removed_without_network(self):
        """Tiles found in RAM cache should be cleared without network requests."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0), (1, 0, 1)}
        h.ram_cache[(1, 0, 0)] = "cached_image"
        # (1, 0, 1) is not cached and suspension is active
        h._network_suspended_until = time.time() + 60

        session = MagicMock()
        result = h.retry_skipped_tiles(session)

        self.assertFalse(result)
        self.assertNotIn((1, 0, 0), h._skipped_tiles)
        self.assertIn((1, 0, 1), h._skipped_tiles)
        session.get.assert_not_called()
        self.assertEqual(h.tiles_skipped_count, 1)

    def test_new_job_aborts_retry(self):
        """If a new job is pending, retry should abort without network requests."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0)}
        h._pending_job = (99, 1, 0, 10, 0, 10)  # new job pending

        session = MagicMock()
        result = h.retry_skipped_tiles(session)

        self.assertFalse(result)
        session.get.assert_not_called()
        # Early return skips the tiles_skipped emission (matches production behavior)
        self.assertEqual(h.tiles_skipped_count, 0)

    def test_duplicate_error_notification_suppressed(self):
        """Second failure should not emit a duplicate network_error."""
        h = TileRetryTestHarness()
        h._skipped_tiles = {(1, 0, 0)}
        h._notified_error = True  # already notified

        session = MagicMock()
        session.get.return_value = FakeResponse(429)

        h.retry_skipped_tiles(session)

        self.assertEqual(len(h.network_error_messages), 0,
                         "should not emit duplicate error notification")
        self.assertEqual(h.tiles_skipped_count, 1)

    def test_repeated_failures_always_reschedule(self):
        """Simulate 5 consecutive retry failures; each must emit tiles_skipped."""
        h = TileRetryTestHarness()

        for i in range(5):
            h._skipped_tiles = {(1, 0, 0), (1, 1, 0)}
            h._network_suspended_until = 0  # expired
            h.tiles_skipped_count = 0

            session = MagicMock()
            session.get.return_value = FakeResponse(503)

            h.retry_skipped_tiles(session)

            self.assertEqual(h.tiles_skipped_count, 1,
                             f"Iteration {i}: tiles_skipped must be emitted")
            self.assertGreater(h._network_suspended_until, time.time())


if __name__ == "__main__":
    unittest.main()

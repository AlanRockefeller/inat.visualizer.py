"""Tests for per-user runtime storage paths."""

import tempfile
import unittest
from pathlib import Path

from visualizer import (
    APP_DATA_DIRECTORY_NAME,
    application_data_dir,
    ensure_application_data_dir,
)


class ApplicationDataDirTests(unittest.TestCase):
    def test_source_run_keeps_current_directory(self) -> None:
        current_dir = Path("/project/runtime")

        result = application_data_dir(frozen=False, current_dir=current_dir)

        self.assertEqual(result, current_dir)

    def test_frozen_macos_uses_application_support(self) -> None:
        result = application_data_dir(
            frozen=True,
            platform_name="darwin",
            environ={},
            home_dir="/Users/friend",
        )

        self.assertEqual(
            result,
            Path("/Users/friend/Library/Application Support")
            / APP_DATA_DIRECTORY_NAME,
        )

    def test_frozen_windows_prefers_local_app_data(self) -> None:
        result = application_data_dir(
            frozen=True,
            platform_name="win32",
            environ={"LOCALAPPDATA": r"C:\Users\friend\AppData\Local"},
            home_dir=r"C:\Users\friend",
        )

        self.assertEqual(
            result,
            Path(r"C:\Users\friend\AppData\Local") / APP_DATA_DIRECTORY_NAME,
        )

    def test_frozen_linux_honors_xdg_data_home(self) -> None:
        result = application_data_dir(
            frozen=True,
            platform_name="linux",
            environ={"XDG_DATA_HOME": "/data/friend"},
            home_dir="/home/friend",
        )

        self.assertEqual(result, Path("/data/friend") / APP_DATA_DIRECTORY_NAME)

    def test_frozen_linux_uses_standard_fallback(self) -> None:
        result = application_data_dir(
            frozen=True,
            platform_name="linux",
            environ={},
            home_dir="/home/friend",
        )

        self.assertEqual(
            result,
            Path("/home/friend/.local/share") / APP_DATA_DIRECTORY_NAME,
        )

    def test_ensure_application_data_dir_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = (
                Path(temp_dir)
                / "Library"
                / "Application Support"
                / APP_DATA_DIRECTORY_NAME
            )

            result = ensure_application_data_dir(
                frozen=True,
                platform_name="darwin",
                environ={},
                home_dir=temp_dir,
            )

            self.assertEqual(result, expected)
            self.assertTrue(result.is_dir())


if __name__ == "__main__":
    unittest.main()

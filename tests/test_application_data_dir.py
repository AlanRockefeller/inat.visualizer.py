"""Tests for per-user runtime storage paths."""

import tempfile
import unittest
from pathlib import Path

from startup_config import (
    APP_DATA_DIRECTORY_NAME,
    SOURCE_CACHE_DIRECTORY_NAME,
    application_cache_dir,
    configure_matplotlib_config_dir,
)
from visualizer import (
    application_data_dir,
    ensure_application_data_dir,
)


class ApplicationDataDirTests(unittest.TestCase):
    def test_source_cache_uses_hidden_current_directory(self) -> None:
        current_dir = Path("/project/runtime")

        result = application_cache_dir(frozen=False, current_dir=current_dir)

        self.assertEqual(result, current_dir / SOURCE_CACHE_DIRECTORY_NAME)

    def test_frozen_cache_uses_platform_conventions(self) -> None:
        cases = (
            (
                "darwin",
                {},
                "/Users/friend",
                Path("/Users/friend/Library/Caches") / APP_DATA_DIRECTORY_NAME,
            ),
            (
                "win32",
                {"LOCALAPPDATA": r"C:\Users\friend\AppData\Local"},
                r"C:\Users\friend",
                Path(r"C:\Users\friend\AppData\Local") / APP_DATA_DIRECTORY_NAME,
            ),
            (
                "linux",
                {"XDG_CACHE_HOME": "/cache/friend"},
                "/home/friend",
                Path("/cache/friend") / APP_DATA_DIRECTORY_NAME,
            ),
        )
        for platform_name, environ, home_dir, expected in cases:
            with self.subTest(platform_name=platform_name):
                self.assertEqual(
                    application_cache_dir(
                        frozen=True,
                        platform_name=platform_name,
                        environ=environ,
                        home_dir=home_dir,
                    ),
                    expected,
                )

    def test_matplotlib_fallback_is_persistent_and_respects_explicit_value(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            blocked_home = root / "blocked-home"
            blocked_home.write_text("not a directory", encoding="utf-8")
            current_dir = root / "runtime"
            current_dir.mkdir()
            environ: dict[str, str] = {}

            result = configure_matplotlib_config_dir(
                frozen=False,
                platform_name="linux",
                environ=environ,
                home_dir=blocked_home,
                current_dir=current_dir,
            )

            expected = current_dir / SOURCE_CACHE_DIRECTORY_NAME / "matplotlib"
            self.assertEqual(result, expected)
            self.assertEqual(environ["MPLCONFIGDIR"], str(expected))
            self.assertTrue(expected.is_dir())

            explicit = root / "explicit-matplotlib"
            explicit_environ = {"MPLCONFIGDIR": str(explicit)}
            self.assertEqual(
                configure_matplotlib_config_dir(
                    environ=explicit_environ,
                    current_dir=current_dir,
                ),
                explicit,
            )
            self.assertFalse(explicit.exists())

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
            Path("/Users/friend/Library/Application Support") / APP_DATA_DIRECTORY_NAME,
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

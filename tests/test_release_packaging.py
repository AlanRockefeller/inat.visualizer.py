"""Contract tests for fast-starting PyInstaller release bundles."""

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTests(unittest.TestCase):
    def test_release_uses_onedir_and_smokes_executables_inside_bundle(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("--onedir --windowed", workflow)
        self.assertNotIn("--onefile", workflow)
        self.assertIn(
            "dist/iNat-Seasonal-Visualizer/iNat-Seasonal-Visualizer.exe",
            workflow,
        )
        self.assertIn(
            "dist/iNat-Seasonal-Visualizer/iNat-Seasonal-Visualizer --smoke-test",
            workflow,
        )

    def test_windows_install_guide_names_zipped_onedir_artifact(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        install_guide = (PROJECT_ROOT / "INSTALL.md").read_text(encoding="utf-8")
        archive_name = "iNat-Seasonal-Visualizer-Windows.zip"

        self.assertIn(f"dist/{archive_name}", workflow)
        self.assertIn(archive_name, install_guide)
        self.assertIn("Keep the other extracted", install_guide)


if __name__ == "__main__":
    unittest.main()

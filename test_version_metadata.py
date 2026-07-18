"""Keep user-visible and platform-specific version metadata synchronized."""

import unittest
from pathlib import Path

from inat_visualizer_version import __version__


PROJECT_ROOT = Path(__file__).parent


class VersionMetadataTests(unittest.TestCase):
    def test_windows_metadata_matches_application_version(self) -> None:
        major, minor, patch = (int(part) for part in __version__.split("."))
        metadata = (PROJECT_ROOT / "windows_version_info.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn(f"filevers=({major}, {minor}, {patch}, 0)", metadata)
        self.assertIn(f"prodvers=({major}, {minor}, {patch}, 0)", metadata)
        self.assertIn(f'"FileVersion", "{__version__}"', metadata)
        self.assertIn(f'"ProductVersion", "{__version__}"', metadata)

    def test_changelog_contains_current_version(self) -> None:
        changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn(f"## [{__version__}]", changelog)

    def test_readme_displays_current_version(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn(f"Current source version: **{__version__}**", readme)


if __name__ == "__main__":
    unittest.main()

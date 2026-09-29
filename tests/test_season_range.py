"""Tests for the abundance-mode season picker."""

import os
import subprocess
import sys
import textwrap
import unittest
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from visualizer import (
    around_date_season,
    format_month_day,
    season_day_of_year,
    season_length_days,
)


class SeasonHelperTests(unittest.TestCase):
    def test_day_of_year_uses_leap_year(self):
        self.assertEqual(season_day_of_year((1, 1)), 1)
        self.assertEqual(season_day_of_year((3, 1)), 61)
        self.assertEqual(season_day_of_year((12, 31)), 366)

    def test_length_within_year_and_wrapping(self):
        self.assertEqual(season_length_days((10, 1), (11, 30)), 61)
        self.assertEqual(season_length_days((12, 1), (2, 29)), 91)
        self.assertEqual(season_length_days((5, 5), (5, 5)), 1)

    def test_around_date_wraps_new_year(self):
        start, end = around_date_season(datetime(2026, 1, 5))
        self.assertEqual((start, end), ((12, 22), (1, 19)))

    def test_format(self):
        self.assertEqual(format_month_day((11, 15)), "Nov 15")


class SeasonRangeWidgetTests(unittest.TestCase):
    def test_presets_fields_and_day_lists_offscreen(self) -> None:
        script = textwrap.dedent("""
            from PyQt6.QtWidgets import QApplication
            import visualizer

            app = QApplication.instance() or QApplication([])
            widget = visualizer.SeasonRangeWidget()
            changes = []
            widget.changed.connect(lambda: changes.append(widget.label()))
            assert widget.label() == "Oct 1 – Nov 30"
            assert widget.preset_combo.currentText() == visualizer.SEASON_CUSTOM_TEXT
            assert widget.summary_label.text() == "61 days"

            widget.preset_combo.setCurrentText("Winter (Dec – Feb)")
            assert (widget.start(), widget.end()) == ((12, 1), (2, 29))
            assert widget.summary_label.text() == "91 days · crosses Jan 1"
            assert changes[-1] == "Dec 1 – Feb 29"

            widget.preset_combo.setCurrentText("April")
            assert (widget.start(), widget.end()) == ((4, 1), (4, 30))
            assert widget.end_day.count() == 30

            # Editing a field turns the preset back into a custom range.
            widget.end_month.setCurrentIndex(4)
            assert widget.end() == (5, 30)
            assert widget.preset_combo.currentText() == visualizer.SEASON_CUSTOM_TEXT

            # Switching to a shorter month clamps the day.
            widget.set_range((1, 31), (3, 31))
            widget.start_month.setCurrentIndex(1)
            assert widget.start() == (2, 29)
            assert widget.start_day.count() == 29

            # Choosing month/day that match a preset selects that preset.
            widget.set_range((6, 1), (8, 31))
            assert widget.preset_combo.currentText() == "Summer (Jun – Aug)"

            widget.preset_combo.setCurrentText(visualizer.SEASON_AROUND_TODAY_TEXT)
            assert widget.preset_combo.currentText() == visualizer.SEASON_AROUND_TODAY_TEXT
            assert visualizer.season_length_days(widget.start(), widget.end()) == 29
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

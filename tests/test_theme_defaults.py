"""Tests for cross-platform theme defaults and graph contrast."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from visualizer import (
    DARK_GRAPH_BACKGROUND,
    DEFAULT_THEME,
    INatSeasonalVisualizer,
    LIGHT_BACKGROUND,
    normalize_theme,
)


class FakeSettings:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)


def theme_harness(settings: FakeSettings):
    figure = Figure()
    axis = figure.subplots()
    canvas = MagicMock()
    harness = SimpleNamespace(
        settings=settings,
        app_font_size=12,
        scale_factor=1.0,
        central_widget=MagicMock(),
        figure=figure,
        ax=axis,
        canvas=canvas,
        setStyleSheet=MagicMock(),
    )
    harness.get_contrasting_text_color = lambda color: (
        INatSeasonalVisualizer.get_contrasting_text_color(harness, color)
    )
    return harness


class ThemeDefaultTests(unittest.TestCase):
    def test_missing_or_invalid_preference_defaults_to_dark(self) -> None:
        self.assertEqual(DEFAULT_THEME, "dark")
        self.assertEqual(normalize_theme(None), "dark")
        self.assertEqual(normalize_theme("system"), "dark")

    def test_explicit_light_preference_is_preserved(self) -> None:
        self.assertEqual(normalize_theme("light"), "light")

    def test_clean_start_uses_light_text_on_dark_graph(self) -> None:
        harness = theme_harness(FakeSettings())
        placeholder = harness.ax.text(0.5, 0.5, "Welcome", color="black")

        INatSeasonalVisualizer.apply_stylesheet(harness)

        self.assertEqual(placeholder.get_color(), "white")
        self.assertEqual(harness.ax.get_facecolor(), to_rgba(DARK_GRAPH_BACKGROUND))

    def test_switching_to_dark_recolors_existing_placeholder(self) -> None:
        settings = FakeSettings({"theme": "light"})
        harness = theme_harness(settings)
        placeholder = harness.ax.text(0.5, 0.5, "Welcome", color="black")
        INatSeasonalVisualizer.apply_stylesheet(harness)
        self.assertEqual(harness.ax.get_facecolor(), to_rgba(LIGHT_BACKGROUND))

        settings.values["theme"] = "dark"
        INatSeasonalVisualizer.apply_stylesheet(harness)

        self.assertEqual(placeholder.get_color(), "white")
        self.assertEqual(harness.ax.get_facecolor(), to_rgba(DARK_GRAPH_BACKGROUND))

    def test_text_contrast_uses_custom_graph_background(self) -> None:
        settings = FakeSettings(
            {
                "theme": "dark",
                "window_bg_color": "#000000",
                "graph_bg_color": "#ffffff",
            }
        )
        harness = theme_harness(settings)
        annotation = harness.ax.text(0.5, 0.5, "Annotation", color="white")

        INatSeasonalVisualizer.apply_stylesheet(harness)

        self.assertEqual(annotation.get_color(), "black")

    def test_map_picker_button_has_distinct_accent_style(self) -> None:
        harness = theme_harness(FakeSettings())

        INatSeasonalVisualizer.apply_stylesheet(harness)

        stylesheet = harness.central_widget.setStyleSheet.call_args.args[0]
        self.assertIn("QPushButton#mapPickerButton", stylesheet)
        self.assertIn("background-color: #2f80ed", stylesheet)
        self.assertIn("border-radius: 6px", stylesheet)
        self.assertIn("font-weight: 700", stylesheet)


if __name__ == "__main__":
    unittest.main()

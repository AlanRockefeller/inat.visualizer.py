# Changelog

All notable changes to the iNaturalist Seasonal Visualizer are documented here.

## [Unreleased]

### Changed

- Make dark mode the default on a clean installation while preserving each
  user's explicitly saved light/dark preference.
- Recalculate graph-text contrast from the graph background and recolor
  existing placeholder, annotation, and legend text when switching themes.
- Make the approximately 1 GB local Parquet database optional at startup. The
  choice dialog now explains local-search speed and storage benefits versus the
  online API's connectivity and rate-limit tradeoffs.
- Allow the application to continue in API-only mode when the user skips the
  database download or a download fails, and visibly disable Local Search when
  the observation database is unavailable.

## [1.0.1] - 2026-07-17

### Fixed

- Pin `requests-ratelimiter` and `pyrate-limiter` to versions compatible with
  `pyinaturalist 0.19.0`, fixing the `BucketFullException` startup crash in the
  packaged Windows and macOS applications.
- Store logs, downloaded Parquet databases, and caches in a writable per-user
  application data directory in packaged applications.
- Run local DuckDB searches in a background worker so the interface remains
  responsive, and cleanly stop the worker when the application closes.
- Use POST requests when large taxon filters would exceed a safe URL length.

### Changed

- Publish separate macOS downloads for Apple Silicon and Intel processors.
- Add dependency and frozen-application smoke tests to the release workflow.
- Add application, Windows executable, and macOS bundle version metadata.

## [1.0.0] - 2026-06-14

### Added

- Initial cross-platform release for Windows, macOS, and Linux.
- Seasonal plotting from the iNaturalist API or local Parquet databases.
- Interactive map, taxonomy expansion, caching, themes, and data/graph export.

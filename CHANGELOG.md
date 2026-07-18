# Changelog

All notable changes to the iNaturalist Seasonal Visualizer are documented here.

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

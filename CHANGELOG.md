# Changelog

All notable changes to the iNaturalist Seasonal Visualizer are documented here.

## [1.0.3] - 2026-07-17

### Added

- Add a resumable DWCA database updater that rebuilds and validates both local
  Parquet databases, invalidates stale taxonomy descendant caches, and removes
  the large raw inputs after a successful atomic replacement.
- Check installed local databases for differently sized remote versions with
  HEAD requests, ask before downloading, and atomically replace completed files.

### Fixed

- Keep Windows startup dialogs in front of the splash screen so the optional
  database download choice cannot be hidden behind an uncloseable topmost window.
- Prevent false update prompts after a locally generated database is copied to
  the web server by comparing file sizes instead of `Last-Modified` timestamps
  and using the observation database as the coordinated release marker.
- Keep ISO year-boundary observations in a fixed 52-week seasonal graph by
  assigning January boundary dates to week 1 and December dates to week 52.
- Preserve HTTP error status codes for falsey `requests` responses and include a
  bounded response-body excerpt in diagnostics.

### Changed

- Rotate application logs at 2 MiB with two backups, record versioned startup
  markers, and move per-taxon cache-loading details from INFO to DEBUG.

## [1.0.2] - 2026-07-17

### Changed

- Reduce the main-screen settings and database-statistics text to half the
  welcome-instruction font size, and move Fetch Taxon IDs from the sidebar to
  the File menu.
- Make dark mode the default on a clean installation while preserving each
  user's explicitly saved light/dark preference.
- Recalculate graph-text contrast from the graph background and recolor
  existing placeholder, annotation, and legend text when switching themes.
- Make the approximately 1 GB local Parquet database optional at startup. The
  choice dialog now explains local-search speed and storage benefits versus the
  online API's connectivity and rate-limit tradeoffs.
- Allow the application to continue in API-only mode when the user skips the
  database download or a download fails, and visibly disable local-data graphing
  when the observation database is unavailable.
- Rename the graph buttons to clarify the choice between local database results
  and live iNaturalist data, and reflect those names in the welcome instructions.

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

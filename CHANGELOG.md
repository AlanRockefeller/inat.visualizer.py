# Changelog

All notable changes to the iNaturalist Seasonal Visualizer are documented here.

## [1.0.6] - 2026-07-17

### Added

- Add a Cancel Search button for live and local searches.

### Fixed

- App now starts a lot faster
- Keep the app responsive while it gets live iNaturalist data, and show progress
  so the user knows the search is still running.
- Bound the live API response cache, prune expired entries, and reclaim space
  from the oversized legacy pyinaturalist cache.
- Default the search end date to today instead of limiting live searches to the
  newest date in the periodically refreshed local database.

### Changed

- Package release applications in PyInstaller one-folder mode to avoid
  extracting the full dependency bundle on every launch.
- Remove the extra search mode message from the sidebar.
- Store the API response cache in the app data directory with a configurable
  `128 MB` default budget.
- Rely on pyinaturalist's built-in GET rate limiter and log concise live-search
  summaries instead of full request URLs and HTTP headers.

## [1.0.5] - 2026-07-17

### Changed

- Add step-by-step Windows SmartScreen and macOS Gatekeeper instructions to
  every GitHub release, and keep the version-neutral installation guide in the
  repository for easy reference.

## [1.0.4] - 2026-07-17

### Changed

- Make the splash screen not always on top

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

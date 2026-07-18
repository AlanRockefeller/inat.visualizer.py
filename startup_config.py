"""Early, standard-library-only configuration needed before heavy imports."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

APP_DATA_DIRECTORY_NAME = "iNat Seasonal Visualizer"
SOURCE_CACHE_DIRECTORY_NAME = ".inat-visualizer-cache"


def application_cache_dir(
    *,
    frozen: bool | None = None,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home_dir: str | Path | None = None,
    current_dir: str | Path | None = None,
) -> Path:
    """Return a persistent, per-user directory for disposable app caches."""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if not is_frozen:
        current = Path.cwd() if current_dir is None else Path(current_dir)
        return current / SOURCE_CACHE_DIRECTORY_NAME

    target_platform = sys.platform if platform_name is None else platform_name
    environment = os.environ if environ is None else environ
    home = Path.home() if home_dir is None else Path(home_dir)

    if target_platform == "darwin":
        base_dir = home / "Library" / "Caches"
    elif target_platform.startswith("win"):
        configured_dir = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        base_dir = (
            Path(configured_dir) if configured_dir else home / "AppData" / "Local"
        )
    else:
        configured_dir = environment.get("XDG_CACHE_HOME")
        base_dir = Path(configured_dir) if configured_dir else home / ".cache"

    return base_dir / APP_DATA_DIRECTORY_NAME


def _matplotlib_default_directories(
    *,
    platform_name: str,
    environ: Mapping[str, str],
    home_dir: Path,
) -> tuple[Path, Path]:
    """Return Matplotlib's default config and cache directories."""
    if platform_name.startswith(("linux", "freebsd")):
        config_base = Path(environ.get("XDG_CONFIG_HOME", home_dir / ".config"))
        cache_base = Path(environ.get("XDG_CACHE_HOME", home_dir / ".cache"))
        return config_base / "matplotlib", cache_base / "matplotlib"
    default_dir = home_dir / ".matplotlib"
    return default_dir, default_dir


def _ensure_writable_directory(path: Path) -> bool:
    """Create *path* if possible and report whether it can be written."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return path.is_dir() and os.access(path, os.W_OK)


def configure_matplotlib_config_dir(
    *,
    frozen: bool | None = None,
    platform_name: str | None = None,
    environ: dict[str, str] | None = None,
    home_dir: str | Path | None = None,
    current_dir: str | Path | None = None,
) -> Path | None:
    """Give Matplotlib a persistent fallback when its defaults are unwritable.

    An explicit ``MPLCONFIGDIR`` always wins. When Matplotlib's normal config
    and cache directories work, leave them untouched so user customizations
    continue to apply.
    """
    environment = os.environ if environ is None else environ
    configured = environment.get("MPLCONFIGDIR")
    if configured:
        return Path(configured)

    target_platform = sys.platform if platform_name is None else platform_name
    home = Path.home() if home_dir is None else Path(home_dir)
    default_dirs = _matplotlib_default_directories(
        platform_name=target_platform,
        environ=environment,
        home_dir=home,
    )
    if all(_ensure_writable_directory(path) for path in default_dirs):
        return None

    fallback_dir = (
        application_cache_dir(
            frozen=frozen,
            platform_name=target_platform,
            environ=environment,
            home_dir=home,
            current_dir=current_dir,
        )
        / "matplotlib"
    )
    if not _ensure_writable_directory(fallback_dir):
        return None
    environment["MPLCONFIGDIR"] = os.fspath(fallback_dir)
    return fallback_dir


# Matplotlib selects its config/cache directory during import.
configure_matplotlib_config_dir()

"""Path resolution utilities ensuring predictable filesystem operations."""

from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Assuming this file is located at <project_root>/src/lecture_analyzer/utils/paths.py
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    """Resolves a path relative to the project root unless already absolute."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return get_project_root() / p


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating parents if necessary."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_relative_path(target_path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> str:
    """Returns a path relative to base_path if on the same drive, else returns string of target_path."""
    t = Path(target_path)
    b = resolve_path(base_path) if base_path is not None else get_project_root()
    try:
        return str(t.relative_to(b))
    except (ValueError, TypeError):
        return str(t)

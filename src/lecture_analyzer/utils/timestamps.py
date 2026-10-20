"""Timestamp utilities for consistent event and elapsed time handling."""

from datetime import datetime, timezone
import time
from typing import Optional


def now_utc_iso() -> str:
    """Returns current UTC wall-clock time formatted as standard ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def monotonic_time() -> float:
    """Returns high-resolution monotonic elapsed time in seconds."""
    return time.perf_counter()


def parse_iso(iso_str: str) -> datetime:
    """Parses an ISO8601 timestamp string into a timezone-aware datetime object."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_duration(seconds: float) -> str:
    """Formats duration in seconds into MM:SS or HH:MM:SS format."""
    if seconds < 0:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

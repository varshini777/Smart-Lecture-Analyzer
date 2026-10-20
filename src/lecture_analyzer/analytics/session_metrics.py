"""Session analytics, inter-observation timing, and behavioral distribution metrics."""

from dataclasses import dataclass
import sqlite3
from typing import Dict, List, Optional
import numpy as np

from lecture_analyzer.behavior.labels import ALL_BEHAVIOR_LABELS
from lecture_analyzer.storage.database import Database


@dataclass
class TimingMetrics:
    total_observations: int
    mean_interval_ms: float
    median_interval_ms: float
    std_interval_ms: float
    min_interval_ms: float
    max_interval_ms: float
    effective_fps: float


@dataclass
class SessionSummaryMetrics:
    session_id: str
    duration_seconds: float
    total_observations: int
    track_count: int
    timing: TimingMetrics
    smoothed_behavior_distribution: Dict[str, float]  # Label -> percentage (0.0 - 100.0)
    raw_behavior_distribution: Dict[str, float]
    observability_distribution: Dict[str, float]     # FULL, PARTIAL, UNOBSERVABLE -> %
    mean_face_quality: float


def compute_session_metrics(db: Database, session_id: str) -> Optional[SessionSummaryMetrics]:
    """Calculates comprehensive timing and distribution metrics for a session."""
    session_rows = db.execute_query("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
    if not session_rows:
        return None

    sess = session_rows[0]
    duration_s = float(sess["duration_seconds"]) if sess["duration_seconds"] is not None else 0.0

    obs_rows = db.execute_query(
        "SELECT elapsed_seconds, monotonic_ts, face_quality, observability_level, raw_behavior, smoothed_behavior, track_id "
        "FROM observations WHERE session_id = ? ORDER BY elapsed_seconds ASC;",
        (session_id,)
    )

    total_obs = len(obs_rows)
    if total_obs == 0:
        return SessionSummaryMetrics(
            session_id=session_id,
            duration_seconds=duration_s,
            total_observations=0,
            track_count=0,
            timing=TimingMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            smoothed_behavior_distribution={lbl: 0.0 for lbl in ALL_BEHAVIOR_LABELS},
            raw_behavior_distribution={lbl: 0.0 for lbl in ALL_BEHAVIOR_LABELS},
            observability_distribution={"FULL": 0.0, "PARTIAL_FACE_ONLY": 0.0, "PARTIAL_POSE_ONLY": 0.0, "UNOBSERVABLE": 0.0},
            mean_face_quality=0.0,
        )

    # 1. Timing metrics
    timestamps = [float(r["monotonic_ts"]) for r in obs_rows]
    unique_tracks = set(r["track_id"] for r in obs_rows)

    if len(timestamps) > 1:
        diffs_ms = np.diff(timestamps) * 1000.0
        # Filter out negative or crazy anomalies
        valid_diffs = diffs_ms[diffs_ms >= 0.0]
        if len(valid_diffs) > 0:
            mean_ms = float(np.mean(valid_diffs))
            median_ms = float(np.median(valid_diffs))
            std_ms = float(np.std(valid_diffs))
            min_ms = float(np.min(valid_diffs))
            max_ms = float(np.max(valid_diffs))
            eff_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        else:
            mean_ms = median_ms = std_ms = min_ms = max_ms = eff_fps = 0.0
    else:
        mean_ms = median_ms = std_ms = min_ms = max_ms = eff_fps = 0.0

    timing = TimingMetrics(
        total_observations=total_obs,
        mean_interval_ms=round(mean_ms, 2),
        median_interval_ms=round(median_ms, 2),
        std_interval_ms=round(std_ms, 2),
        min_interval_ms=round(min_ms, 2),
        max_interval_ms=round(max_ms, 2),
        effective_fps=round(eff_fps, 2),
    )

    # 2. Distributions
    smoothed_counts: Dict[str, int] = {lbl: 0 for lbl in ALL_BEHAVIOR_LABELS}
    raw_counts: Dict[str, int] = {lbl: 0 for lbl in ALL_BEHAVIOR_LABELS}
    obs_counts: Dict[str, int] = {"FULL": 0, "PARTIAL_FACE_ONLY": 0, "PARTIAL_POSE_ONLY": 0, "UNOBSERVABLE": 0}
    face_qualities = []

    for r in obs_rows:
        sb = r["smoothed_behavior"]
        rb = r["raw_behavior"]
        ol = r["observability_level"]
        fq = r["face_quality"]

        if sb in smoothed_counts:
            smoothed_counts[sb] += 1
        if rb in raw_counts:
            raw_counts[rb] += 1
        if ol in obs_counts:
            obs_counts[ol] += 1

        if fq is not None:
            face_qualities.append(float(fq))

    smoothed_dist = {lbl: round((cnt / total_obs) * 100.0, 1) for lbl, cnt in smoothed_counts.items()}
    raw_dist = {lbl: round((cnt / total_obs) * 100.0, 1) for lbl, cnt in raw_counts.items()}
    obs_dist = {k: round((cnt / total_obs) * 100.0, 1) for k, cnt in obs_counts.items()}
    mean_fq = float(np.mean(face_qualities)) if face_qualities else 0.0

    return SessionSummaryMetrics(
        session_id=session_id,
        duration_seconds=round(duration_s, 2),
        total_observations=total_obs,
        track_count=len(unique_tracks),
        timing=timing,
        smoothed_behavior_distribution=smoothed_dist,
        raw_behavior_distribution=raw_dist,
        observability_distribution=obs_dist,
        mean_face_quality=round(mean_fq, 2),
    )

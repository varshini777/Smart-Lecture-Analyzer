"""Timestamp-based temporal smoothing and persistence engine for observable behaviors."""

from collections import deque
from dataclasses import dataclass
import logging
from typing import Deque, Dict, List, Optional, Tuple

from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.rules import RuleEvaluationResult
from lecture_analyzer.config import BehaviorConfig

logger = logging.getLogger("lecture_analyzer.behavior.temporal_smoothing")

# Minimum elapsed physical time (seconds) required before transitioning TO a new state.
# Prevents single-frame detector noise or natural transient actions from flipping state.
STATE_PERSISTENCE_THRESHOLDS: Dict[BehaviorCategory, float] = {
    BehaviorCategory.DROWSY_SLEEP_LIKE: 0.6,   # Sustained eye closure >= 0.6s (normal blinks < 0.35s suppressed)
    BehaviorCategory.LOOKING_AWAY: 0.4,        # Head turned away >= 0.4s (brief glances < 0.4s suppressed)
    BehaviorCategory.HEAD_DOWN: 0.45,          # Head down >= 0.45s
    BehaviorCategory.WRITING_READING: 0.45,    # Writing posture >= 0.45s
    BehaviorCategory.PHONE_USE: 0.35,          # Phone present >= 0.35s
    BehaviorCategory.ENGAGED_BEHAVIOR: 0.20,   # Engaged forward posture >= 0.20s (snappy recovery upon opening eyes)
    BehaviorCategory.UNCERTAIN: 0.0,           # Immediate if sensory evidence is missing
}


@dataclass
class SmoothedBehaviorResult:
    """Consolidated behavioral state preserving both raw frame output and smoothed result."""
    track_id: str
    raw_behavior: BehaviorCategory
    raw_score: float
    smoothed_behavior: BehaviorCategory
    smoothed_score: float
    window_duration_seconds: float
    observation_count_in_window: int
    is_observable: bool
    explanation: str
    wall_timestamp: str
    monotonic_ts: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class TemporalBehaviorSmoother:
    """Smooths transient detection flicker using real elapsed monotonic time rather than frame counts.

    Guarantees:
    - Independent history buffers per track_id (zero cross-track contamination).
    - True elapsed time tracking (independent of acquisition/processing FPS).
    - Preserves established state during transient blinks (< 0.4s) and momentary glances (< 0.5s).
    - Immediate reset on tracking gap (> max_gap_seconds) or session boundary.
    - Explicit retention of raw vs. smoothed behavior classifications.
    """

    def __init__(
        self,
        config: Optional[BehaviorConfig] = None,
        max_gap_seconds: float = 2.0,
        persistence_thresholds: Optional[Dict[BehaviorCategory, float]] = None,
    ):
        self.config = config or BehaviorConfig()
        self.window_seconds = self.config.smoothing_window_seconds
        self.max_gap_seconds = max_gap_seconds
        self.persistence_thresholds = persistence_thresholds or dict(STATE_PERSISTENCE_THRESHOLDS)

        # track_id -> deque of (monotonic_ts, raw_behavior, heuristic_score, is_observable)
        self._histories: Dict[str, Deque[Tuple[float, BehaviorCategory, float, bool]]] = {}
        # track_id -> last observed monotonic_ts
        self._last_ts_by_track: Dict[str, float] = {}
        # track_id -> currently established smoothed behavior
        self._current_established_state: Dict[str, BehaviorCategory] = {}
        # track_id -> candidate behavior currently being evaluated for persistence
        self._candidate_state: Dict[str, BehaviorCategory] = {}
        # track_id -> monotonic timestamp when candidate behavior started
        self._candidate_start_ts: Dict[str, float] = {}

    def reset(self, track_id: Optional[str] = None) -> None:
        """Resets history for a specific track or all tracks."""
        if track_id is not None:
            self._histories.pop(track_id, None)
            self._last_ts_by_track.pop(track_id, None)
            self._current_established_state.pop(track_id, None)
            self._candidate_state.pop(track_id, None)
            self._candidate_start_ts.pop(track_id, None)
        else:
            self._histories.clear()
            self._last_ts_by_track.clear()
            self._current_established_state.clear()
            self._candidate_state.clear()
            self._candidate_start_ts.clear()

    def process(self, eval_res: RuleEvaluationResult) -> SmoothedBehaviorResult:
        """Applies timestamp-based sliding window persistence and aggregation."""
        tid = eval_res.track_id
        now = eval_res.monotonic_ts
        raw_cat = eval_res.behavior
        raw_score = eval_res.heuristic_score
        is_obs = eval_res.is_observable

        if not self.config.enable_smoothing:
            return SmoothedBehaviorResult(
                track_id=tid,
                raw_behavior=raw_cat,
                raw_score=raw_score,
                smoothed_behavior=raw_cat,
                smoothed_score=raw_score,
                window_duration_seconds=0.0,
                observation_count_in_window=1,
                is_observable=is_obs,
                explanation=eval_res.explanation,
                wall_timestamp=eval_res.wall_timestamp,
                monotonic_ts=now,
            )

        # Check for gap / stale track resumption
        last_ts = self._last_ts_by_track.get(tid)
        if last_ts is not None and (now - last_ts) > self.max_gap_seconds:
            # Tracking loss occurred: wipe history so old states do not contaminate new track
            self.reset(tid)

        self._last_ts_by_track[tid] = now
        history = self._histories.setdefault(tid, deque())

        # Append current observation
        history.append((now, raw_cat, raw_score, is_obs))

        # Evict entries older than window_seconds
        cutoff = now - self.window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

        n_obs = len(history)
        actual_duration = round(history[-1][0] - history[0][0], 3) if n_obs > 1 else 0.0

        # Unobservable check: if majority of observations in window are unobservable
        unobservable_count = sum(1 for _, _, _, o in history if not o)
        if (unobservable_count / n_obs) >= 0.50:
            self._current_established_state[tid] = BehaviorCategory.UNCERTAIN
            self._candidate_state[tid] = BehaviorCategory.UNCERTAIN
            self._candidate_start_ts[tid] = now
            return SmoothedBehaviorResult(
                track_id=tid,
                raw_behavior=raw_cat,
                raw_score=raw_score,
                smoothed_behavior=BehaviorCategory.UNCERTAIN,
                smoothed_score=raw_score if not is_obs else 0.20,
                window_duration_seconds=actual_duration,
                observation_count_in_window=n_obs,
                is_observable=False,
                explanation="Evidence unobservable or unreliable across majority of time window.",
                wall_timestamp=eval_res.wall_timestamp,
                monotonic_ts=now,
            )

        # Get current established state for this track
        established = self._current_established_state.get(tid)

        if established is None or established == BehaviorCategory.UNCERTAIN:
            # First observable frame or recovering from UNCERTAIN: immediately establish valid observable state
            if raw_cat != BehaviorCategory.UNCERTAIN:
                established = raw_cat
                self._current_established_state[tid] = established
                self._candidate_state[tid] = raw_cat
                self._candidate_start_ts[tid] = now

        # Track candidate state persistence for transitioning between distinct observable behaviors
        prev_cand = self._candidate_state.get(tid)
        if prev_cand != raw_cat and raw_cat != BehaviorCategory.UNCERTAIN:
            self._candidate_state[tid] = raw_cat
            self._candidate_start_ts[tid] = now

        candidate_elapsed = now - self._candidate_start_ts.get(tid, now)
        required_persistence = self.persistence_thresholds.get(raw_cat, 0.4)

        # Check if candidate state has persisted long enough to displace established state
        if candidate_elapsed >= required_persistence and raw_cat != BehaviorCategory.UNCERTAIN:
            # State transition confirmed by elapsed physical time!
            established = raw_cat
            self._current_established_state[tid] = established

        # Compute average score for the established category within the active window
        matching_scores = [s for _, c, s, _ in history if c == established]
        avg_score = sum(matching_scores) / len(matching_scores) if matching_scores else raw_score

        explanation = eval_res.explanation
        if established != raw_cat:
            explanation = (
                f"{eval_res.explanation} [Raw transient state '{raw_cat.value}' "
                f"suppressed by temporal persistence filter ({candidate_elapsed:.2f}s < {required_persistence:.2f}s)]"
            )

        return SmoothedBehaviorResult(
            track_id=tid,
            raw_behavior=raw_cat,
            raw_score=raw_score,
            smoothed_behavior=established,
            smoothed_score=round(avg_score, 2),
            window_duration_seconds=actual_duration,
            observation_count_in_window=n_obs,
            is_observable=True,
            explanation=explanation,
            wall_timestamp=eval_res.wall_timestamp,
            monotonic_ts=now,
        )

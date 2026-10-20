"""Unit tests for anonymous person tracking."""

import pytest
from lecture_analyzer.config import TrackingConfig
from lecture_analyzer.tracking.tracker import AnonymousPersonTracker


def test_tracker_initializes_anonymous_ids():
    tracker = AnonymousPersonTracker(TrackingConfig(min_hits_to_confirm=1))
    dets = [(100, 100, 200, 300), (350, 100, 450, 300)]
    tracks = tracker.update(dets, current_monotonic_ts=1.0)

    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert ids == {"S001", "S002"}


def test_tracker_maintains_identity_across_frames():
    tracker = AnonymousPersonTracker(TrackingConfig(min_hits_to_confirm=1))
    # Frame 1
    tracker.update([(100, 100, 200, 300)], current_monotonic_ts=1.0)
    # Frame 2: slightly shifted box
    tracks_f2 = tracker.update([(105, 102, 205, 302)], current_monotonic_ts=1.1)

    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == "S001"
    assert tracks_f2[0].hits == 2


def test_tracker_cleanup_stale_tracks():
    cfg = TrackingConfig(track_retention_seconds=1.0, min_hits_to_confirm=1)
    tracker = AnonymousPersonTracker(cfg)

    tracker.update([(100, 100, 200, 300)], current_monotonic_ts=1.0)
    assert len(tracker.tracks) == 1

    # Frame at t=1.5 with no detections (within 1.0s grace period)
    tracker.update([], current_monotonic_ts=1.5)
    assert len(tracker.tracks) == 1

    # Frame at t=2.5 (> 1.0s retention elapsed)
    tracker.update([], current_monotonic_ts=2.5)
    assert len(tracker.tracks) == 0


def test_tracker_reset_clears_all_and_resets_id():
    tracker = AnonymousPersonTracker(TrackingConfig(min_hits_to_confirm=1))
    tracker.update([(100, 100, 200, 300)], current_monotonic_ts=1.0)
    assert "S001" in [t.track_id for t in tracker.tracks.values()]

    # Reset at session boundary
    tracker.reset()
    assert len(tracker.tracks) == 0

    # New session: first ID must again be S001
    tracks_new = tracker.update([(50, 50, 150, 250)], current_monotonic_ts=10.0)
    assert len(tracks_new) == 1
    assert tracks_new[0].track_id == "S001"


def test_two_visible_persons_maintain_independent_tracks():
    """Two concurrent people maintain independent anonymous tracks without swapping."""
    tracker = AnonymousPersonTracker(TrackingConfig(min_hits_to_confirm=1))
    person_left = (50, 100, 150, 400)
    person_right = (350, 100, 450, 400)

    # Frame 1
    tracks_1 = tracker.update([person_left, person_right], current_monotonic_ts=1.0)
    assert len(tracks_1) == 2
    id_map = {t.track_id: t.bbox for t in tracks_1}

    # Frame 2: both move slightly
    person_left_moved = (55, 102, 155, 402)
    person_right_moved = (355, 98, 455, 398)
    tracks_2 = tracker.update([person_left_moved, person_right_moved], current_monotonic_ts=1.1)
    assert len(tracks_2) == 2

    # Verify ID persistence
    for t in tracks_2:
        if t.bbox == person_left_moved:
            assert t.track_id == "S001"
        elif t.bbox == person_right_moved:
            assert t.track_id == "S002"


def test_track_reentry_assigns_new_anonymous_id_without_biometrics():
    """When a subject leaves (> retention window) and re-enters, a new anonymous ID is assigned."""
    cfg = TrackingConfig(track_retention_seconds=2.0, min_hits_to_confirm=1)
    tracker = AnonymousPersonTracker(cfg)

    # Person S001 present at t=1.0
    tracker.update([(100, 100, 200, 300)], current_monotonic_ts=1.0)
    assert "S001" in [t.track_id for t in tracker.tracks.values()]

    # Person leaves: no detections from t=1.5 to t=4.0 (> 2.0s retention)
    tracker.update([], current_monotonic_ts=4.0)
    assert len(tracker.tracks) == 0

    # Person re-enters at same location at t=5.0
    tracks_reentry = tracker.update([(100, 100, 200, 300)], current_monotonic_ts=5.0)
    assert len(tracks_reentry) == 1
    # Strict privacy: Cannot re-identify without biometric recognition; must receive new ID S002
    assert tracks_reentry[0].track_id == "S002"


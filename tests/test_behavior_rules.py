"""Unit tests for the explainable behavioral rule engine and temporal smoother."""

import pytest
from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory, is_concentrating, is_distracted
from lecture_analyzer.behavior.rules import BehaviorRuleEngine

from lecture_analyzer.behavior.temporal_smoothing import TemporalBehaviorSmoother
from lecture_analyzer.config import AppConfig


@pytest.fixture
def rule_engine():
    return BehaviorRuleEngine(AppConfig())


@pytest.fixture
def base_features():
    return ObservationFeatures(
        track_id="S001",
        wall_timestamp="2026-10-18T12:00:00Z",
        monotonic_ts=10.0,
        bbox_x1=100, bbox_y1=100, bbox_x2=300, bbox_y2=400,
        ear_avg=0.30, ear_left=0.30, ear_right=0.30, mar=0.20,
        face_quality=0.85, face_detected=True,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
        pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=0.95, head_shoulder_offset=0.5,
        wrists_near_head=False, wrists_near_table=False,
        phone_detected=False, phone_confidence=None,
        observability_level="FULL", observability_reasons="All modalities observable",
    )


def test_engaged_behavior_detection(rule_engine, base_features):
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.ENGAGED_BEHAVIOR
    assert res.heuristic_score > 0.70
    assert "forward-facing" in res.explanation.lower()


def test_looking_away_detection(rule_engine, base_features):
    base_features.head_yaw = 40.0  # Turn right
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.LOOKING_AWAY
    assert "right" in res.explanation.lower()


def test_head_down_detection(rule_engine, base_features):
    base_features.head_pitch = -25.0  # Tilt down
    base_features.wrists_near_table = False
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.HEAD_DOWN


def test_writing_reading_detection(rule_engine, base_features):
    base_features.head_pitch = -25.0  # Tilt down
    base_features.wrists_near_table = True  # Wrists at desk level
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.WRITING_READING


def test_drowsy_sleep_like_detection(rule_engine, base_features):
    base_features.ear_avg = 0.12  # Closed eyes (< 0.20)
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE


def test_phone_use_detection(rule_engine, base_features):
    base_features.phone_detected = True
    base_features.phone_confidence = 0.85
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.PHONE_USE


def test_unobservable_becomes_uncertain(rule_engine, base_features):
    base_features.observability_level = "UNOBSERVABLE"
    base_features.face_detected = False
    base_features.pose_detected = False
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.UNCERTAIN
    assert res.is_observable is False


def test_temporal_smoothing_transient_suppression(rule_engine, base_features):
    smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

    # 4 frames of engaged behavior at t = 10.0, 10.2, 10.4, 10.6
    for t in [10.0, 10.2, 10.4, 10.6]:
        base_features.monotonic_ts = t
        base_features.head_yaw = 0.0
        eval_res = rule_engine.evaluate(base_features)
        smoothed = smoother.process(eval_res)
        assert smoothed.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR

    # 1 single glitch frame of LOOKING_AWAY at t = 10.8
    base_features.monotonic_ts = 10.8
    base_features.head_yaw = 45.0
    glitch_eval = rule_engine.evaluate(base_features)
    smoothed_glitch = smoother.process(glitch_eval)
    # Raw is looking away, but smoothed is still ENGAGED_BEHAVIOR (flicker suppressed)
    assert smoothed_glitch.raw_behavior == BehaviorCategory.LOOKING_AWAY
    assert smoothed_glitch.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR


def test_temporal_smoothing_stale_gap_wipes_history(rule_engine, base_features):
    smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

    base_features.monotonic_ts = 10.0
    eval_res = rule_engine.evaluate(base_features)
    smoother.process(eval_res)

    # Big gap of 5.0 seconds (e.g. participant left and came back or tracking paused)
    base_features.monotonic_ts = 15.0
    base_features.head_yaw = 45.0  # Looking away
    eval_res2 = rule_engine.evaluate(base_features)
    smoothed = smoother.process(eval_res2)

    # Because gap > 2.0s, old history was reset, smoothed instantly matches current observation
    assert smoothed.smoothed_behavior == BehaviorCategory.LOOKING_AWAY


# ==============================================================================
# Task 1 & 2 Regression Tests: Missing-Feature Safeguards & Temporal Persistence
# ==============================================================================

def test_missing_face_does_not_trigger_drowsy(rule_engine, base_features):
    """A. Missing face must not automatically trigger DROWSY_SLEEP_LIKE."""
    base_features.face_detected = False
    base_features.ear_avg = None
    base_features.head_pose_valid = False
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.DROWSY_SLEEP_LIKE
    assert res.behavior == BehaviorCategory.UNCERTAIN


def test_missing_or_invalid_ear_does_not_trigger_drowsy(rule_engine, base_features):
    """B. Missing or degenerate EAR (None or 0.0) must not be interpreted as closed eyes."""
    # Case 1: EAR is None
    base_features.ear_avg = None
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.DROWSY_SLEEP_LIKE

    # Case 2: EAR is 0.0 (degenerate calculation / division artifact)
    base_features.ear_avg = 0.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.DROWSY_SLEEP_LIKE


def test_invalid_head_pose_with_zero_angles_not_treated_as_engaged(rule_engine, base_features):
    """C. Invalid yaw or pitch set to zero must not be interpreted as a confirmed forward-facing head."""
    base_features.head_pose_valid = False
    base_features.head_yaw = 0.0
    base_features.head_pitch = 0.0
    base_features.head_roll = 0.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.ENGAGED_BEHAVIOR
    assert res.behavior == BehaviorCategory.UNCERTAIN


def test_missing_posture_not_interpreted_as_upright_or_writing(rule_engine, base_features):
    """D. Missing posture must not be interpreted as a valid upright or writing posture."""
    base_features.pose_detected = False
    base_features.torso_upright_ratio = None
    base_features.wrists_near_table = False
    base_features.head_pitch = -25.0  # Downward pitch
    res = rule_engine.evaluate(base_features)
    # Should not be WRITING_READING because posture/wrists are unobserved
    assert res.behavior == BehaviorCategory.HEAD_DOWN


def test_person_without_face_never_engaged(rule_engine, base_features):
    """F. A person without a visible face must not automatically become ENGAGED_BEHAVIOR."""
    base_features.face_detected = False
    base_features.ear_avg = None
    base_features.head_pose_valid = False
    base_features.pose_detected = True
    base_features.torso_upright_ratio = 1.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.ENGAGED_BEHAVIOR
    assert res.behavior == BehaviorCategory.UNCERTAIN


def test_low_quality_face_landmarks_prevent_confident_classifications(rule_engine, base_features):
    """G. Low-quality face landmarks must not trigger confident classifications."""
    base_features.face_quality = 0.15  # Well below 0.35 threshold
    base_features.head_yaw = 0.0
    base_features.head_pitch = 0.0
    base_features.ear_avg = 0.30
    res = rule_engine.evaluate(base_features)
    # Cannot be confidently ENGAGED because face quality is degraded
    assert res.behavior != BehaviorCategory.ENGAGED_BEHAVIOR
    assert res.behavior == BehaviorCategory.UNCERTAIN


def test_temporal_smoothing_persistent_drowsy_transition(rule_engine, base_features):
    """Persistent eye closure >= 1.5s successfully transitions to DROWSY_SLEEP_LIKE."""
    smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

    # Initially engaged at t = 10.0
    base_features.monotonic_ts = 10.0
    eval_res = rule_engine.evaluate(base_features)
    s = smoother.process(eval_res)
    assert s.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR

    # Eyes close at t = 10.5: brief blink (< 1.5s) suppressed at t = 11.0 (elapsed 0.5s)
    base_features.ear_avg = 0.12
    base_features.monotonic_ts = 11.0
    eval_drowsy_1 = rule_engine.evaluate(base_features)
    s1 = smoother.process(eval_drowsy_1)
    assert s1.raw_behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert s1.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR  # Suppressed!

    # Eyes still closed at t = 12.6 (elapsed 1.6s >= 1.5s threshold since start at 11.0)
    base_features.monotonic_ts = 12.6
    eval_drowsy_2 = rule_engine.evaluate(base_features)
    s2 = smoother.process(eval_drowsy_2)
    assert s2.raw_behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert s2.smoothed_behavior == BehaviorCategory.DROWSY_SLEEP_LIKE  # Transition confirmed!


def test_temporal_smoothing_concurrent_tracks_isolation(rule_engine, base_features):
    """Multiple concurrent tracks do not contaminate each other's smoothing history."""
    smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

    # Track 1 (S001) is engaged
    base_features.track_id = "S001"
    base_features.monotonic_ts = 10.0
    base_features.head_yaw = 0.0
    eval_1 = rule_engine.evaluate(base_features)
    res_1 = smoother.process(eval_1)
    assert res_1.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR

    # Track 2 (S002) is looking away
    base_features.track_id = "S002"
    base_features.monotonic_ts = 10.0
    base_features.head_yaw = 45.0
    eval_2 = rule_engine.evaluate(base_features)
    res_2 = smoother.process(eval_2)
    assert res_2.smoothed_behavior == BehaviorCategory.LOOKING_AWAY

    # Verify Track 1 still retains ENGAGED_BEHAVIOR state without contamination
    base_features.track_id = "S001"
    base_features.monotonic_ts = 10.3
    base_features.head_yaw = 0.0
    eval_1b = rule_engine.evaluate(base_features)
    res_1b = smoother.process(eval_1b)
    assert res_1b.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR


def test_temporal_smoothing_irregular_frame_intervals(rule_engine, base_features):
    """Irregular frame arrival (e.g. 1.88 FPS or variable camera lag) operates reliably."""
    smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

    # Variable deltas: 0.1s, 0.6s, 0.3s
    timestamps = [1.0, 1.1, 1.7, 2.0]
    for ts in timestamps:
        base_features.monotonic_ts = ts
        eval_res = rule_engine.evaluate(base_features)
        res = smoother.process(eval_res)
        assert res.smoothed_behavior == BehaviorCategory.ENGAGED_BEHAVIOR


def test_looking_away_upward_pitch(rule_engine, base_features):
    """Head tilted upward (pitch > 25 deg) is classified as LOOKING_AWAY."""
    base_features.head_pitch = 35.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.LOOKING_AWAY
    assert "upward" in res.explanation.lower()


def test_natural_lateral_head_roll_remains_engaged(rule_engine, base_features):
    """Natural lateral head tilt (|roll| <= 45 deg) with open eyes and forward gaze remains ENGAGED_BEHAVIOR."""
    base_features.head_roll = 25.0
    base_features.head_yaw = 5.0
    base_features.head_pitch = 2.0
    base_features.ear_avg = 0.28
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.ENGAGED_BEHAVIOR


def test_extreme_lateral_head_roll_triggers_looking_away(rule_engine, base_features):
    """Extreme lateral head roll (|roll| > 45 deg) with eyes open is classified as LOOKING_AWAY."""
    base_features.head_roll = 55.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.LOOKING_AWAY
    assert "lateral" in res.explanation.lower()


def test_extreme_lateral_head_roll_with_closed_eyes_triggers_drowsy(rule_engine, base_features):
    """Extreme lateral head roll (|roll| > 45 deg) with closed eyes is classified as DROWSY_SLEEP_LIKE."""
    base_features.head_roll = 55.0
    base_features.ear_avg = 0.12
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE


def test_distant_pose_writing_reading_detection(rule_engine, base_features):
    """Distant student with undetectable face mesh but hands at desk level is classified as WRITING_READING."""
    base_features.face_detected = False
    base_features.ear_avg = None
    base_features.head_pose_valid = False
    base_features.pose_detected = True
    base_features.torso_upright_ratio = 0.99
    base_features.wrists_near_table = True
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.WRITING_READING
    assert res.is_observable is True
    assert res.heuristic_score >= 0.75
    assert is_concentrating(res.behavior) is True


def test_concentration_and_distraction_taxonomy_helpers():
    """Verify construct-valid grouping into concentration vs distraction."""
    # Concentrating: Engaged listening or active desk coursework
    assert is_concentrating(BehaviorCategory.ENGAGED_BEHAVIOR) is True
    assert is_concentrating(BehaviorCategory.WRITING_READING) is True
    assert is_concentrating(BehaviorCategory.LOOKING_AWAY) is False
    assert is_concentrating(BehaviorCategory.PHONE_USE) is False
    assert is_concentrating(BehaviorCategory.HEAD_DOWN) is False
    assert is_concentrating(BehaviorCategory.DROWSY_SLEEP_LIKE) is False
    assert is_concentrating(BehaviorCategory.UNCERTAIN) is False

    # Distracted: Visual orientation away, device use, slump, or fatigue
    assert is_distracted(BehaviorCategory.LOOKING_AWAY) is True
    assert is_distracted(BehaviorCategory.PHONE_USE) is True
    assert is_distracted(BehaviorCategory.HEAD_DOWN) is True
    assert is_distracted(BehaviorCategory.DROWSY_SLEEP_LIKE) is True
    assert is_distracted(BehaviorCategory.ENGAGED_BEHAVIOR) is False
    assert is_distracted(BehaviorCategory.WRITING_READING) is False
    assert is_distracted(BehaviorCategory.UNCERTAIN) is False


def test_hand_covering_face_triggers_drowsy_distracted(rule_engine, base_features):
    """When a student covers their face with hands, it must be classified as DROWSY_SLEEP_LIKE / distracted."""
    base_features.hands_covering_face = True
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert res.is_observable is True
    assert res.heuristic_score >= 0.80
    assert is_distracted(res.behavior) is True
    assert is_concentrating(res.behavior) is False
    assert "face covered" in res.explanation.lower() or "hand" in res.explanation.lower()


def test_face_rubbing_with_open_eyes_not_engaged(rule_engine, base_features):
    """Rubbing face or resting chin on hand with forward gaze and open eyes must NEVER be ENGAGED_BEHAVIOR."""
    base_features.wrists_near_head = True
    base_features.ear_avg = 0.32
    base_features.head_yaw = 0.0
    base_features.head_pitch = 0.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior != BehaviorCategory.ENGAGED_BEHAVIOR
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert is_distracted(res.behavior) is True
    assert is_concentrating(res.behavior) is False


def test_face_occluded_by_hands_triggers_drowsy(rule_engine, base_features):
    """When hands occlude or obscure the face, the rule engine classifies as DROWSY_SLEEP_LIKE rather than UNCERTAIN."""
    base_features.face_detected = False
    base_features.ear_avg = None
    base_features.head_pose_valid = False
    base_features.hands_covering_face = True
    base_features.wrists_near_head = True
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert res.is_observable is True
    assert is_distracted(res.behavior) is True


def test_propping_head_on_hand_triggers_drowsy(rule_engine, base_features):
    """Propping head or chin up with a hand with head tilt is classified as DROWSY_SLEEP_LIKE."""
    base_features.wrists_near_head = True
    base_features.head_roll = 25.0
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert is_distracted(res.behavior) is True


def test_hands_covering_face_with_closed_eyes_high_confidence(rule_engine, base_features):
    """Covering face with hands while eyes are closed yields high confidence drowsiness (score >= 0.90)."""
    base_features.hands_covering_face = True
    base_features.ear_avg = 0.10
    res = rule_engine.evaluate(base_features)
    assert res.behavior == BehaviorCategory.DROWSY_SLEEP_LIKE
    assert res.heuristic_score >= 0.90





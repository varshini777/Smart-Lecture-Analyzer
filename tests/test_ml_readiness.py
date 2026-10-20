"""Unit tests for ML readiness gating and model training guardrails."""

import pandas as pd
import pytest

from lecture_analyzer.ml.readiness import MLReadinessChecker
from lecture_analyzer.ml.train import MLTrainingOrchestrator, ReadinessGateViolationError
from lecture_analyzer.storage.database import Database


@pytest.fixture
def readiness_checker():
    return MLReadinessChecker()


def test_empty_dataframe_fails_readiness(readiness_checker):
    df = pd.DataFrame()
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("DATASET_NON_EMPTY" in f for f in report.failed_gates)


def test_insufficient_participants_fails_gate(readiness_checker):
    # Only 2 participants (need >= 10)
    data = {
        "participant_id": ["P01"] * 50 + ["P02"] * 50,
        "session_id": ["S01"] * 50 + ["S02"] * 50,
        "provenance": ["REAL_HUMAN"] * 100,
        "human_behavior_label": ["ENGAGED_BEHAVIOR"] * 100,
        "ear_avg": [0.3] * 100,
        "mar": [0.2] * 100,
    }
    df = pd.DataFrame(data)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("PARTICIPANT_DIVERSITY" in f for f in report.failed_gates)


def test_disallowed_provenance_fails_gate(readiness_checker):
    # Has simulated data
    data = {
        "participant_id": [f"P{i:02d}" for i in range(12) for _ in range(10)],
        "session_id": [f"S{i:02d}" for i in range(12) for _ in range(10)],
        "provenance": ["SIMULATED"] * 120,
        "human_behavior_label": ["ENGAGED_BEHAVIOR"] * 120,
        "ear_avg": [0.3] * 120,
        "mar": [0.2] * 120,
    }
    df = pd.DataFrame(data)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("STRICT_PROVENANCE" in f for f in report.failed_gates)


def test_training_orchestrator_raises_on_gating_failure(monkeypatch, tmp_path):
    db_path = tmp_path / "empty.db"
    db = Database(db_path)
    trainer = MLTrainingOrchestrator(db, models_dir=str(tmp_path))

    # Attempting to train on empty database must raise ReadinessGateViolationError
    with pytest.raises(ReadinessGateViolationError, match="ML Training Gate FAILED"):
        trainer.train_and_evaluate(candidate_df=None)


def test_multiple_sessions_single_participant_fails_participant_diversity(readiness_checker):
    # 10 sessions from only 1 participant must pass session diversity but fail participant diversity
    data = {
        "participant_id": ["P01"] * 100,
        "session_id": [f"S{i:02d}" for i in range(10) for _ in range(10)],
        "provenance": ["REAL_HUMAN"] * 100,
        "human_behavior_label": ["ENGAGED_BEHAVIOR"] * 100,
        "ear_avg": [0.3] * 100,
        "mar": [0.2] * 100,
    }
    df = pd.DataFrame(data)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("PARTICIPANT_DIVERSITY" in f for f in report.failed_gates)
    assert any("SESSION_DIVERSITY" in p for p in report.passed_gates)
    diversity_failure = [f for f in report.failed_gates if "PARTICIPANT_DIVERSITY" in f][0]
    assert "Multiple sessions from the same participant do not increase participant diversity" in diversity_failure


def test_insufficient_annotated_duration_fails_gate(readiness_checker):
    # 10 participants and sessions, but elapsed duration total is only 50s (< 600s)
    records = []
    for i in range(10):
        records.append({
            "participant_id": f"P{i:02d}",
            "session_id": f"S{i:02d}",
            "elapsed_seconds": 0.0,
            "provenance": "REAL_HUMAN",
            "human_behavior_label": "ENGAGED_BEHAVIOR",
            "ear_avg": 0.3,
            "mar": 0.2,
        })
        records.append({
            "participant_id": f"P{i:02d}",
            "session_id": f"S{i:02d}",
            "elapsed_seconds": 5.0,  # 5s per session * 10 = 50s total
            "provenance": "REAL_HUMAN",
            "human_behavior_label": "ENGAGED_BEHAVIOR",
            "ear_avg": 0.3,
            "mar": 0.2,
        })
    df = pd.DataFrame(records)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("ANNOTATED_DURATION" in f for f in report.failed_gates)


def test_fully_compliant_dataset_passes_readiness(readiness_checker):
    # 10 participants, 10 sessions, 2 classes with >= 40 samples each and >= 3 participants each,
    # total duration > 600s, no missing features
    records = []
    for i in range(10):
        pid = f"P{i:02d}"
        sid = f"S{i:02d}"
        for t in range(50):
            elapsed = float(t * 1.5)  # 0 to 73.5s per session * 10 sessions = 735s total
            lbl = "ENGAGED_BEHAVIOR" if t % 2 == 0 else "LOOKING_AWAY"
            records.append({
                "participant_id": pid,
                "session_id": sid,
                "elapsed_seconds": elapsed,
                "provenance": "REAL_HUMAN",
                "human_behavior_label": lbl,
                "ear_avg": 0.32,
                "mar": 0.15,
                "head_yaw": 2.0,
                "head_pitch": -4.0,
                "head_roll": 1.0,
                "face_quality": 0.9,
                "torso_upright_ratio": 1.1,
            })
    df = pd.DataFrame(records)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is True
    assert len(report.failed_gates) == 0
    assert "READY" in report.verdict_message


def test_critical_feature_missingness_fails_gate(readiness_checker):
    # Fully compliant dataset except head_pitch is 50% missing (overall missingness is low ~7%)
    records = []
    for i in range(10):
        pid = f"P{i:02d}"
        sid = f"S{i:02d}"
        for t in range(50):
            elapsed = float(t * 1.5)
            lbl = "ENGAGED_BEHAVIOR" if t % 2 == 0 else "LOOKING_AWAY"
            records.append({
                "participant_id": pid,
                "session_id": sid,
                "elapsed_seconds": elapsed,
                "provenance": "REAL_HUMAN",
                "human_behavior_label": lbl,
                "ear_avg": 0.32,
                "mar": 0.15,
                "head_yaw": 2.0,
                "head_pitch": None if t < 25 else -4.0,  # 50% missing!
                "head_roll": 1.0,
                "face_quality": 0.9,
                "torso_upright_ratio": 1.1,
            })
    df = pd.DataFrame(records)
    report = readiness_checker.check_dataframe_readiness(df)
    assert report.is_ready_for_training is False
    assert any("CRITICAL_FEATURE_COMPLETENESS" in f for f in report.failed_gates)
    assert any("DATA_COMPLETENESS" in p for p in report.passed_gates)



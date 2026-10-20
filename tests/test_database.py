"""Unit tests for SQLite database schema, persistence, and integrity."""

import tempfile
from pathlib import Path
import pytest

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import (
    AnnotationRecord,
    AnnotationRepository,
    ObservationRepository,
    SessionRecord,
    SessionRepository,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        yield db


def test_database_initialization_and_reopen(temp_db):
    db_path = temp_db.db_path
    assert db_path.exists()

    # Query migrations
    rows = temp_db.execute_query("SELECT * FROM schema_migrations;")
    assert len(rows) == 1
    assert rows[0]["version"] == 1

    # Reopen same database
    db2 = Database(db_path)
    rows2 = db2.execute_query("SELECT * FROM schema_migrations;")
    assert len(rows2) == 1


def test_session_lifecycle_and_cascading_delete(temp_db):
    sess_repo = SessionRepository(temp_db)
    obs_repo = ObservationRepository(temp_db)
    annot_repo = AnnotationRepository(temp_db)

    # 1. Create session
    sess = SessionRecord(
        session_id="SES_TEST_001",
        participant_code="P_TEST",
        start_time="2026-10-18T10:00:00Z",
        end_time=None,
        duration_seconds=None,
        collection_source="WEBCAM",
        provenance="REAL_HUMAN",
        collection_mode="LIVE_CAMERA",
        consent_confirmed=True,
        recording_path=None,
        feature_schema_version="1.0.0",
        pipeline_version="0.1.0",
        model_versions=None,
        quality_status="PENDING_AUDIT",
        error_message=None,
        notes="Test session",
    )
    sess_repo.create(sess)

    fetched = sess_repo.get("SES_TEST_001")
    assert fetched is not None
    assert fetched.participant_code == "P_TEST"

    # 2. Insert observation
    feats = ObservationFeatures(
        track_id="S001",
        wall_timestamp="2026-10-18T10:00:01Z",
        monotonic_ts=10.0,
        bbox_x1=10, bbox_y1=10, bbox_x2=100, bbox_y2=100,
        ear_avg=0.25, ear_left=0.25, ear_right=0.25, mar=0.3,
        face_quality=0.8, face_detected=True,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
        pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=0.9, head_shoulder_offset=0.4,
        wrists_near_head=False, wrists_near_table=False,
        phone_detected=False, phone_confidence=None,
        observability_level="FULL", observability_reasons="None",
    )
    smoothed = SmoothedBehaviorResult(
        track_id="S001",
        raw_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
        raw_score=0.85,
        smoothed_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
        smoothed_score=0.85,
        window_duration_seconds=1.0,
        observation_count_in_window=1,
        is_observable=True,
        explanation="Normal",
        wall_timestamp="2026-10-18T10:00:01Z",
        monotonic_ts=10.0,
    )
    obs_repo.insert_batch("SES_TEST_001", [(feats, smoothed, 1, 1.0)])
    assert obs_repo.count("SES_TEST_001") == 1

    # 3. Add Annotation
    annot = AnnotationRecord(
        id=None,
        session_id="SES_TEST_001",
        track_id="S001",
        start_seconds=0.0,
        end_seconds=2.0,
        behavior_label="ENGAGED_BEHAVIOR",
        annotator_id="HUMAN_ANN_1",
        confidence_level="HIGH",
        notes=None,
        created_at="2026-10-18T10:05:00Z",
    )
    annot_id = annot_repo.save(annot)
    assert annot_id is not None
    assert len(annot_repo.list_by_session("SES_TEST_001")) == 1

    # 4. Cascading delete
    deleted = sess_repo.delete("SES_TEST_001")
    assert deleted is True
    assert sess_repo.get("SES_TEST_001") is None
    # Observations and annotations cascaded
    assert obs_repo.count("SES_TEST_001") == 0
    assert len(annot_repo.list_by_session("SES_TEST_001")) == 0

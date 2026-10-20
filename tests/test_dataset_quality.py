"""Unit tests for session data quality auditing."""

import tempfile
from pathlib import Path
import pytest

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.dataset.quality_audit import AuditVerdict, QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import (
    AnnotationRecord,
    AnnotationRepository,
    ObservationRepository,
    SessionRecord,
    SessionRepository,
)


@pytest.fixture
def audit_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_audit.db"
        db = Database(db_path)
        yield db


def test_audit_empty_session_verdict_fail(audit_db):
    sess_repo = SessionRepository(audit_db)
    sess_repo.create(SessionRecord(
        session_id="SES_EMPTY",
        participant_code="P01",
        start_time="2026-10-18T12:00:00Z",
        end_time="2026-10-18T12:00:10Z",
        duration_seconds=10.0,
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
        notes=None,
    ))

    auditor = QualityAuditor(audit_db)
    report = auditor.audit_session("SES_EMPTY")

    assert report.verdict == AuditVerdict.FAIL
    assert "Zero observation records" in report.reasons[0]


def test_audit_valid_session_with_annotations(audit_db):
    sess_repo = SessionRepository(audit_db)
    obs_repo = ObservationRepository(audit_db)
    annot_repo = AnnotationRepository(audit_db)

    sess_repo.create(SessionRecord(
        session_id="SES_VALID",
        participant_code="P02",
        start_time="2026-10-18T12:00:00Z",
        end_time="2026-10-18T12:00:10Z",
        duration_seconds=10.0,
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
        notes=None,
    ))

    # Add 35 regular observations at ~30 FPS
    records = []
    for i in range(35):
        t_mono = 10.0 + (i * 0.033)
        el_s = i * 0.033
        feats = ObservationFeatures(
            track_id="S001",
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=t_mono,
            bbox_x1=10, bbox_y1=10, bbox_x2=100, bbox_y2=100,
            ear_avg=0.3, ear_left=0.3, ear_right=0.3, mar=0.2,
            face_quality=0.8, face_detected=True,
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
            pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=0.9, head_shoulder_offset=0.4,
            wrists_near_head=False, wrists_near_table=False,
            phone_detected=False, phone_confidence=None,
            observability_level="FULL", observability_reasons="All observable",
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
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=t_mono,
        )
        records.append((feats, smoothed, i + 1, el_s))

    obs_repo.insert_batch("SES_VALID", records)

    # Add annotation covering the interval
    annot_repo.save(AnnotationRecord(
        id=None,
        session_id="SES_VALID",
        track_id="S001",
        start_seconds=0.0,
        end_seconds=2.0,
        behavior_label="ENGAGED_BEHAVIOR",
        annotator_id="A1",
        confidence_level="HIGH",
        notes=None,
        created_at="2026-10-18T12:05:00Z",
    ))

    auditor = QualityAuditor(audit_db)
    report = auditor.audit_session("SES_VALID")

    assert report.verdict in [AuditVerdict.PASS, AuditVerdict.PASS_WITH_WARNINGS]
    assert report.observation_count == 35
    assert report.effective_fps > 20.0
    assert report.descriptive_agreement_percentage == 100.0  # Automated engaged matched annotated engaged

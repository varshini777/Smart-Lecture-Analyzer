"""Unit tests for provenance-filtered dataset export and manifest verification."""

import tempfile
from pathlib import Path
import pytest

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.dataset.export_dataset import PREDICTIVE_FEATURE_COLUMNS, DatasetExporter
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import (
    AnnotationRecord,
    AnnotationRepository,
    ObservationRepository,
    SessionRecord,
    SessionRepository,
)


@pytest.fixture
def populated_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_export.db"
        db = Database(db_path)
        sess_repo = SessionRepository(db)
        obs_repo = ObservationRepository(db)
        annot_repo = AnnotationRepository(db)

        # 1. Genuine Human Session
        sess_real = SessionRecord(
            session_id="SES_REAL_01",
            participant_code="P001",
            start_time="2026-10-18T10:00:00Z",
            end_time="2026-10-18T10:00:10Z",
            duration_seconds=10.0,
            collection_source="WEBCAM",
            provenance="REAL_HUMAN",
            collection_mode="LIVE_CAMERA",
            consent_confirmed=True,
            recording_path=None,
            feature_schema_version="1.0.0",
            pipeline_version="0.1.0",
            model_versions=None,
            quality_status="PASS",
            error_message=None,
            notes=None,
        )
        sess_repo.create(sess_real)

        # 2. Simulated Session
        sess_sim = SessionRecord(
            session_id="SES_SIM_01",
            participant_code="P_SIM",
            start_time="2026-10-18T11:00:00Z",
            end_time="2026-10-18T11:00:10Z",
            duration_seconds=10.0,
            collection_source="SYNTHETIC",
            provenance="SIMULATED",
            collection_mode="SYNTHETIC_DEMO",
            consent_confirmed=False,
            recording_path=None,
            feature_schema_version="1.0.0",
            pipeline_version="0.1.0",
            model_versions=None,
            quality_status="PASS",
            error_message=None,
            notes=None,
        )
        sess_repo.create(sess_sim)

        # Add observations to real session
        feats = ObservationFeatures(
            track_id="S001",
            wall_timestamp="2026-10-18T10:00:01Z",
            monotonic_ts=10.0,
            bbox_x1=10, bbox_y1=10, bbox_x2=100, bbox_y2=100,
            ear_avg=0.28, ear_left=0.28, ear_right=0.28, mar=0.3,
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
        obs_repo.insert_batch("SES_REAL_01", [(feats, smoothed, 1, 1.0)])

        # Add Human Ground Truth Annotation
        annot_repo.save(AnnotationRecord(
            id=None,
            session_id="SES_REAL_01",
            track_id="S001",
            start_seconds=0.0,
            end_seconds=5.0,
            behavior_label="ENGAGED_BEHAVIOR",
            annotator_id="HUMAN_ANN_1",
            confidence_level="HIGH",
            notes=None,
            created_at="2026-10-18T10:02:00Z"
        ))

        yield db, tmpdir


def test_export_real_human_excludes_simulated(populated_db):
    db, tmpdir = populated_db
    exporter = DatasetExporter(db, exports_dir=tmpdir)

    csv_path, manifest = exporter.export_dataset(
        provenance=DataProvenance.REAL_HUMAN,
        require_annotations=True,
        require_passed_audit=False,
    )

    assert csv_path is not None
    assert manifest is not None
    assert manifest.unique_session_count == 1
    assert "SES_REAL_01" in manifest.included_session_ids
    assert "SES_SIM_01" not in manifest.included_session_ids
    assert manifest.total_observation_rows == 1
    assert manifest.sha256_hash != ""

    # Verify predictive features do NOT contain participant_id or session_id
    assert "participant_id" not in PREDICTIVE_FEATURE_COLUMNS
    assert "session_id" not in PREDICTIVE_FEATURE_COLUMNS
    assert "wall_timestamp" not in PREDICTIVE_FEATURE_COLUMNS

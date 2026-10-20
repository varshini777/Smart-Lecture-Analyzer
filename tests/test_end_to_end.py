"""End-to-end integration test verifying the complete data flow from frame to export."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from lecture_analyzer.behavior.feature_extraction import extract_features_from_state
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.rules import BehaviorRuleEngine
from lecture_analyzer.behavior.temporal_smoothing import TemporalBehaviorSmoother
from lecture_analyzer.collection.session_service import SessionService
from lecture_analyzer.config import AppConfig
from lecture_analyzer.dataset.export_dataset import DatasetExporter
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.dataset.quality_audit import QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import AnnotationRecord, AnnotationRepository
from lecture_analyzer.vision.camera import FramePacket
from lecture_analyzer.vision.pipeline import VisionPipeline


@pytest.fixture
def integrated_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "e2e.db"
        rec_dir = tmp_path / "recordings"
        exp_dir = tmp_path / "exports"

        config = AppConfig()
        config.storage.database_path = str(db_path)
        config.storage.recordings_dir = str(rec_dir)
        config.storage.exports_dir = str(exp_dir)

        db = Database(db_path)
        pipeline = VisionPipeline(config)
        session_service = SessionService(config, db)
        rule_engine = BehaviorRuleEngine(config)
        smoother = TemporalBehaviorSmoother(config.behavior)
        auditor = QualityAuditor(db)
        exporter = DatasetExporter(db, exports_dir=str(exp_dir))

        yield {
            "config": config,
            "db": db,
            "pipeline": pipeline,
            "session_service": session_service,
            "rule_engine": rule_engine,
            "smoother": smoother,
            "auditor": auditor,
            "exporter": exporter,
        }


def test_complete_end_to_end_pipeline(integrated_environment):
    env = integrated_environment
    session_service: SessionService = env["session_service"]
    pipeline: VisionPipeline = env["pipeline"]
    rule_engine: BehaviorRuleEngine = env["rule_engine"]
    smoother: TemporalBehaviorSmoother = env["smoother"]
    auditor: QualityAuditor = env["auditor"]
    exporter: DatasetExporter = env["exporter"]
    db: Database = env["db"]

    # 1. Start Session
    session_id = session_service.start_session(
        collection_source="TEST_FIXTURE_ARRAY",
        provenance=DataProvenance.PIPELINE_VALIDATION,
        collection_mode="CONTROLLED_STUDY",
        participant_code="P_VAL_01",
        consent_confirmed=True,
        record_video=False,
    )
    assert session_service.is_session_active is True

    # 2. Process synthetic test frames
    # Generate 5 test frames
    for i in range(5):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw a synthetic human-like box to simulate person detection
        packet = FramePacket(
            frame=frame,
            frame_index=i + 1,
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=10.0 + i * 0.1,
            acquisition_latency_ms=5.0,
            width=1280,
            height=720,
        )

        analysis = pipeline.process_frame(packet)
        assert analysis is not None

        # Even with black frame (0 persons detected), pipeline returns clean empty result without crashing
        obs_to_record = []
        for tid, state in analysis.tracked_persons.items():
            feats = extract_features_from_state(state, packet.wall_timestamp, packet.monotonic_ts)
            rule_eval = rule_engine.evaluate(feats)
            smoothed = smoother.process(rule_eval)
            obs_to_record.append((feats, smoothed, packet.frame_index))

        # Add a synthetic validation observation to verify data flow
        from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
        synth_feats = ObservationFeatures(
            track_id="S001",
            wall_timestamp=packet.wall_timestamp,
            monotonic_ts=packet.monotonic_ts,
            bbox_x1=50, bbox_y1=50, bbox_x2=200, bbox_y2=400,
            ear_avg=0.30, ear_left=0.30, ear_right=0.30, mar=0.20,
            face_quality=0.85, face_detected=True,
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
            pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=0.95, head_shoulder_offset=0.5,
            wrists_near_head=False, wrists_near_table=False,
            phone_detected=False, phone_confidence=None,
            observability_level="FULL", observability_reasons="All observable",
        )
        synth_eval = rule_engine.evaluate(synth_feats)
        synth_smoothed = smoother.process(synth_eval)
        obs_to_record.append((synth_feats, synth_smoothed, packet.frame_index))

        session_service.record_frame_observations(obs_to_record)

    # 3. Stop Session
    completed_session = session_service.stop_session()
    assert completed_session is not None
    assert completed_session.session_id == session_id
    assert session_service.is_session_active is False

    # 4. Add independent human annotation
    annot_repo = AnnotationRepository(db)
    annot_repo.save(AnnotationRecord(
        id=None,
        session_id=session_id,
        track_id=None,
        start_seconds=0.0,
        end_seconds=1.0,
        behavior_label=BehaviorCategory.ENGAGED_BEHAVIOR.value,
        annotator_id="INDEPENDENT_EVALUATOR",
        confidence_level="HIGH",
        notes="Validation run",
        created_at="2026-10-18T12:01:00Z",
    ))

    # 5. Run Quality Audit
    audit_report = auditor.audit_session(session_id, persist_result=True)
    assert audit_report is not None
    assert audit_report.session_id == session_id

    # 6. Verify Dataset Export under PIPELINE_VALIDATION
    csv_path, manifest = exporter.export_dataset(
        provenance=DataProvenance.PIPELINE_VALIDATION,
        require_annotations=False,
        require_passed_audit=False,
    )
    assert manifest is not None
    assert manifest.allowed_provenance == "PIPELINE_VALIDATION"

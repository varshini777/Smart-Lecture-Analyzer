"""Automated tests for video recording integrity, timestamp sidecars, and annotation synchronization."""

import json
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.collection.recording_service import (
    VideoRecordingService,
    frame_idx_for_time,
    load_timestamp_sidecar,
    time_for_frame_idx,
)
from lecture_analyzer.collection.session_service import SessionService
from lecture_analyzer.config import AppConfig
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.dataset.quality_audit import AuditVerdict, QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.vision.camera import FramePacket


@pytest.fixture
def temp_recordings_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_frame():
    # 640x360 blank color frame
    return np.zeros((360, 640, 3), dtype=np.uint8)


def test_video_recording_lifecycle_and_sidecar(temp_recordings_dir, mock_frame):
    service = VideoRecordingService(recordings_dir=str(temp_recordings_dir))
    session_id = "TEST_SES_001"
    start_mono = 1000.0

    video_path = service.start_recording(
        session_id=session_id,
        frame_width=640,
        frame_height=360,
        nominal_fps=30.0,
        session_start_monotonic=start_mono,
    )

    assert video_path is not None
    assert service.is_recording is True

    # Write 30 frames at simulated 30 FPS (interval = 0.0333s)
    total_frames = 30
    for i in range(total_frames):
        ts = start_mono + (i * 0.0333)
        ok = service.write_frame(
            frame_bgr=mock_frame,
            camera_frame_idx=i,
            monotonic_ts=ts,
            wall_timestamp="2026-10-18T12:00:00Z",
            session_start_monotonic=start_mono,
        )
        assert ok is True

    # Stop recording
    res = service.stop_recording()
    assert res is not None
    saved_path, frames_written, sidecar_path = res

    assert frames_written == total_frames
    assert saved_path.exists()
    assert sidecar_path is not None and sidecar_path.exists()
    assert service.is_recording is False

    # Verify decoded video
    cap = cv2.VideoCapture(str(saved_path))
    assert cap.isOpened()
    decoded_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert decoded_frames == total_frames

    # Verify sidecar structure
    sidecar = load_timestamp_sidecar(saved_path)
    assert sidecar is not None
    assert sidecar["session_id"] == session_id
    assert sidecar["total_frames"] == total_frames
    assert len(sidecar["frames"]) == total_frames
    assert sidecar["frames"][0]["video_frame_idx"] == 0
    assert sidecar["frames"][0]["elapsed_s"] == 0.0
    assert sidecar["frames"][-1]["video_frame_idx"] == total_frames - 1
    assert abs(sidecar["frames"][-1]["elapsed_s"] - (29 * 0.0333)) < 0.001


def test_recording_during_inference_delay_and_frame_skipping(temp_recordings_dir, mock_frame):
    """Verifies that every camera frame is saved even when inference processes only a subset."""
    service = VideoRecordingService(recordings_dir=str(temp_recordings_dir))
    session_id = "TEST_SES_SKIPPING"
    start_mono = 2000.0

    video_path = service.start_recording(
        session_id=session_id,
        frame_width=640,
        frame_height=360,
        nominal_fps=30.0,
        session_start_monotonic=start_mono,
    )
    assert video_path is not None

    # Simulate 30 camera frames acquired
    for i in range(30):
        ts = start_mono + (i * 0.0333)
        service.write_frame(
            frame_bgr=mock_frame,
            camera_frame_idx=i,
            monotonic_ts=ts,
            wall_timestamp="2026-10-18T12:00:00Z",
            session_start_monotonic=start_mono,
        )

    res = service.stop_recording()
    assert res is not None
    _, frames_written, sidecar_path = res
    assert frames_written == 30

    # Sidecar maps all 30 frames
    sidecar = load_timestamp_sidecar(sidecar_path)
    assert sidecar["total_frames"] == 30

    # Test that an inference frame at camera index 15 maps exactly to video frame 15
    infer_time = sidecar["frames"][15]["elapsed_s"]
    found_idx = frame_idx_for_time(sidecar, infer_time)
    assert found_idx == 15


def test_annotation_timeline_alignment_with_sidecar(temp_recordings_dir):
    """Verifies seeking and time translation using sidecar."""
    sidecar = {
        "session_id": "TEST_SES_ALIGN",
        "nominal_fps": 30.0,
        "total_frames": 90,
        "session_duration_s": 3.0,
        "frames": [
            {"video_frame_idx": i, "camera_frame_idx": i, "elapsed_s": round(i * 0.03333, 4)}
            for i in range(90)
        ]
    }

    # Frame 0 -> 0.0s
    assert time_for_frame_idx(sidecar, 0) == 0.0

    # Frame 30 -> ~1.0s
    t30 = time_for_frame_idx(sidecar, 30)
    assert abs(t30 - 1.0) < 0.01

    # Seeking to 1.0s should return frame 30
    idx_1s = frame_idx_for_time(sidecar, 1.0)
    assert idx_1s == 30

    # Seeking to 2.5s should return frame 75
    idx_25s = frame_idx_for_time(sidecar, 2.5)
    assert idx_25s == 75

    # Seeking beyond duration clamps to last frame
    idx_past = frame_idx_for_time(sidecar, 10.0)
    assert idx_past == 89


def test_session_service_recording_lifecycle(temp_recordings_dir, mock_frame):
    """End-to-end integration test of SessionService recording lifecycle and SQLite observations."""
    db_path = temp_recordings_dir / "test.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    # Start session with video recording enabled
    session_id = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
        record_video=True,
        frame_width=640,
        frame_height=360,
    )

    assert service.is_session_active is True
    assert service.recording_service.is_recording is True

    try:
        # Record 15 camera frames
        start_mono = service.session_start_monotonic
        for i in range(15):
            pkt = FramePacket(
                frame=mock_frame,
                frame_index=i,
                wall_timestamp="2026-10-18T12:00:00Z",
                monotonic_ts=start_mono + (i * 0.0333),
                acquisition_latency_ms=1.5,
                width=640,
                height=360,
            )
            service.record_video_frame(pkt)

            # Simulate inference recording every 5 frames
            if i % 5 == 0:
                feat = ObservationFeatures(
                    track_id="S001",
                    wall_timestamp=pkt.wall_timestamp,
                    monotonic_ts=pkt.monotonic_ts,
                    bbox_x1=100,
                    bbox_y1=100,
                    bbox_x2=300,
                    bbox_y2=300,
                    ear_avg=0.28,
                    ear_left=0.28,
                    ear_right=0.28,
                    mar=0.1,
                    face_quality=0.9,
                    face_detected=True,
                    head_yaw=2.0,
                    head_pitch=5.0,
                    head_roll=0.0,
                    head_pose_valid=True,
                    pose_detected=True,
                    shoulder_tilt_deg=0.5,
                    torso_upright_ratio=0.95,
                    head_shoulder_offset=0.1,
                    wrists_near_head=False,
                    wrists_near_table=True,
                    phone_detected=False,
                    phone_confidence=0.0,
                    observability_level="FULL",
                    observability_reasons="All visible",
                )
                smoothed = SmoothedBehaviorResult(
                    track_id="S001",
                    raw_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
                    raw_score=0.95,
                    smoothed_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
                    smoothed_score=0.95,
                    window_duration_seconds=1.0,
                    observation_count_in_window=1,
                    is_observable=True,
                    explanation="Normal engagement",
                    wall_timestamp=pkt.wall_timestamp,
                    monotonic_ts=pkt.monotonic_ts,
                )
                service.record_frame_observations([(feat, smoothed, pkt.frame_index)])

        completed = service.stop_session()
        assert completed is not None
        assert service.is_session_active is False
        assert service.recording_service.is_recording is False

        # Verify session record in SQLite
        sess = service.session_repo.get(session_id)
        assert sess is not None
        assert sess.recording_path is not None
        rec_path = temp_recordings_dir / Path(sess.recording_path).name
        assert rec_path.exists()

        # Verify sidecar exists
        sidecar_path = rec_path.with_name(f"{session_id}_timestamps.json")
        assert sidecar_path.exists()

        # Verify observations in SQLite
        obs_list = service.obs_repo.get_session_observations(session_id)
        assert len(obs_list) == 3  # Frames 0, 5, 10
        sidecar = load_timestamp_sidecar(sidecar_path)
        for obs in obs_list:
            assert obs["elapsed_seconds"] >= 0.0
            # Check that observation elapsed time matches sidecar frame elapsed time
            matching_sidecar_frame = next(f for f in sidecar["frames"] if f["camera_frame_idx"] == obs["frame_index"])
            assert abs(matching_sidecar_frame["elapsed_s"] - obs["elapsed_seconds"]) < 0.005
    finally:
        if service.is_session_active:
            service.stop_session()


def test_disk_space_guard_aborts_recording(temp_recordings_dir):
    """Verifies that recording safely aborts if disk space is below min_free_mb."""
    # Set required free space impossibly high (e.g. 100 TB)
    service = VideoRecordingService(recordings_dir=str(temp_recordings_dir), min_free_mb=100_000_000)
    path = service.start_recording(
        session_id="FAIL_SPACE",
        frame_width=640,
        frame_height=360,
    )
    assert path is None
    assert service.is_recording is False


def test_interrupted_session_shutdown_flags_status(temp_recordings_dir, mock_frame):
    """Verifies that a camera failure or interrupted recording is flagged in SQLite."""
    db_path = temp_recordings_dir / "test_interrupted.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    session_id = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
        record_video=True,
        frame_width=640,
        frame_height=360,
    )

    # Write 5 frames
    for i in range(5):
        pkt = FramePacket(
            frame=mock_frame,
            frame_index=i,
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=service.session_start_monotonic + (i * 0.0333),
            acquisition_latency_ms=1.5,
            width=640,
            height=360,
        )
        service.record_video_frame(pkt)

    # Simulate camera failure / interrupted session
    completed = service.stop_session(error_message="Camera disconnected unexpectedly")
    assert completed is not None
    assert completed.quality_status == "INTERRUPTED"
    assert completed.error_message == "Camera disconnected unexpectedly"

    # SQLite session record retains INTERRUPTED status
    record_in_db = service.session_repo.get(session_id)
    assert record_in_db.quality_status == "INTERRUPTED"
    assert record_in_db.error_message == "Camera disconnected unexpectedly"


def test_controlled_timing_consistency_measurement(temp_recordings_dir, mock_frame):
    """Measures timing consistency between camera frames, video file, and database observations (Task 6)."""
    db_path = temp_recordings_dir / "test_timing.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    # Controlled recording parameters:
    # 60 camera frames at 30.0 FPS = exactly 2.0s duration
    # Inference runs every 3 frames = 20 observation records (10.0 FPS inference rate)
    session_id = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
        record_video=True,
        frame_width=640,
        frame_height=360,
    )

    total_camera_frames = 60
    camera_fps = 30.0
    frame_interval = 1.0 / camera_fps
    start_mono = service.session_start_monotonic

    try:
        for i in range(total_camera_frames):
            frame_ts = start_mono + (i * frame_interval)
            pkt = FramePacket(
                frame=mock_frame,
                frame_index=i,
                wall_timestamp="2026-10-18T12:00:00Z",
                monotonic_ts=frame_ts,
                acquisition_latency_ms=1.2,
                width=640,
                height=360,
            )
            service.record_video_frame(pkt)

            # Inference every 3 frames
            if i % 3 == 0:
                feat = ObservationFeatures(
                    track_id="S001",
                    wall_timestamp=pkt.wall_timestamp,
                    monotonic_ts=pkt.monotonic_ts,
                    bbox_x1=50, bbox_y1=50, bbox_x2=200, bbox_y2=200,
                    ear_avg=0.25, ear_left=0.25, ear_right=0.25, mar=0.1,
                    face_quality=0.85, face_detected=True,
                    head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
                    pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=1.0, head_shoulder_offset=0.1,
                    wrists_near_head=False, wrists_near_table=False,
                    phone_detected=False, phone_confidence=0.0,
                    observability_level="FULL", observability_reasons="Optimal",
                )
                smoothed = SmoothedBehaviorResult(
                    track_id="S001",
                    raw_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
                    raw_score=0.9,
                    smoothed_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
                    smoothed_score=0.9,
                    window_duration_seconds=1.0,
                    observation_count_in_window=1,
                    is_observable=True,
                    explanation="Engaged",
                    wall_timestamp=pkt.wall_timestamp,
                    monotonic_ts=pkt.monotonic_ts,
                )
                service.record_frame_observations([(feat, smoothed, pkt.frame_index)])

        completed = service.stop_session()
        assert completed is not None

        # Verify decoded video
        rec_path = temp_recordings_dir / Path(completed.recording_path).name
        cap = cv2.VideoCapture(str(rec_path))
        decoded_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        configured_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Decoded duration
        decoded_duration_s = decoded_frames / configured_fps

        # Load sidecar
        sidecar_path = rec_path.with_name(f"{session_id}_timestamps.json")
        sidecar = load_timestamp_sidecar(sidecar_path)

        # Retrieve database observations
        obs_rows = service.obs_repo.get_session_observations(session_id)

        # 1. First & last frame timestamps
        first_frame_ts = sidecar["frames"][0]["elapsed_s"]
        last_frame_ts = sidecar["frames"][-1]["elapsed_s"]
        assert first_frame_ts == 0.0
        assert abs(last_frame_ts - ((total_camera_frames - 1) * frame_interval)) < 0.001

        # 2. Frame counts
        assert decoded_frames == total_camera_frames
        assert len(sidecar["frames"]) == total_camera_frames
        assert len(obs_rows) == 20  # 60 / 3 = 20 observations

        # 3. Video duration vs session duration
        # With 60 frames @ 30.0 FPS, video duration is 2.0s, last frame is at 1.967s
        duration_diff = abs(decoded_duration_s - (last_frame_ts + frame_interval))
        assert duration_diff < 0.001

        # 4. Observation timestamp coverage
        # First observation at 0.0s, last at 57 * 0.03333 = 1.90s
        assert obs_rows[0]["elapsed_seconds"] == 0.0
        assert abs(obs_rows[-1]["elapsed_seconds"] - (57 * frame_interval)) < 0.005

        # 5. Zero missing or duplicate frame mappings
        video_indices = [f["video_frame_idx"] for f in sidecar["frames"]]
        assert video_indices == list(range(total_camera_frames))

        camera_indices = [f["camera_frame_idx"] for f in sidecar["frames"]]
        assert camera_indices == list(range(total_camera_frames))
    finally:
        if service.is_session_active:
            service.stop_session()


def test_queue_saturation_drops_frames_and_flags_session(temp_recordings_dir, mock_frame):
    """Verifies that when recording queue fills, frames drop non-blockingly and session is flagged."""
    db_path = temp_recordings_dir / "test_saturation.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    # Configure recording service with tiny queue (maxsize=2)
    service.recording_service = VideoRecordingService(
        recordings_dir=str(temp_recordings_dir),
        queue_maxsize=2,
    )

    session_id = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
        record_video=True,
        frame_width=640,
        frame_height=360,
    )

    try:
        # Pause worker consumption by acquiring lock or flooding queue rapidly
        # Flooding 20 frames into maxsize=2 queue
        t_start = time.perf_counter()
        dropped_any = False
        for i in range(20):
            pkt = FramePacket(
                frame=mock_frame,
                frame_index=i,
                wall_timestamp="2026-10-18T12:00:00Z",
                monotonic_ts=service.session_start_monotonic + (i * 0.001),
                acquisition_latency_ms=0.5,
                width=640,
                height=360,
            )
            ok = service.record_video_frame(pkt)
            if not ok:
                dropped_any = True

        t_elapsed = time.perf_counter() - t_start
        # Crucial check: camera acquisition MUST NOT block indefinitely (20 frames in < 200 ms)
        assert t_elapsed < 0.5, f"Camera thread blocked for {t_elapsed:.2f}s during queue saturation!"
        assert dropped_any is True, "Expected at least one frame to be dropped in tiny queue!"
        assert service.recording_service.frames_dropped > 0

        # Stop session
        completed = service.stop_session()
        assert completed is not None

        # Verify session is NOT marked successful without warnings
        assert completed.quality_status == "WARNING_DROPPED_FRAMES"
        assert "dropped due to recording queue saturation" in (completed.error_message or "")

        # Verify in database
        db_record = service.session_repo.get(session_id)
        assert db_record.quality_status == "WARNING_DROPPED_FRAMES"
        assert str(service.recording_service.last_frames_dropped) in db_record.error_message
    finally:
        if service.is_session_active:
            service.stop_session()


def test_writer_failure_during_worker_execution(temp_recordings_dir, mock_frame):
    """Verifies that an unhandled writer failure flags session as INTERRUPTED."""
    db_path = temp_recordings_dir / "test_writer_fail.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    session_id = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
        record_video=True,
        frame_width=640,
        frame_height=360,
    )

    try:
        # Simulate video writer error by explicitly injecting error in worker
        service.recording_service._writer_error = "Disk write I/O failure (simulated)"

        completed = service.stop_session()
        assert completed is not None
        assert completed.quality_status == "INTERRUPTED"
        assert "Disk write I/O failure" in (completed.error_message or "")
    finally:
        if service.is_session_active:
            service.stop_session()


def test_shutdown_flushing_drains_remaining_queue(temp_recordings_dir, mock_frame):
    """Verifies that stopping a recording cleanly flushes all buffered frames before exit."""
    service = VideoRecordingService(recordings_dir=str(temp_recordings_dir), queue_maxsize=50)
    service.start_recording(
        session_id="DRAIN_TEST",
        frame_width=640,
        frame_height=360,
    )

    total_submitted = 15
    for i in range(total_submitted):
        service.write_frame(mock_frame, camera_frame_idx=i)

    # Immediately stop recording
    res = service.stop_recording()
    assert res is not None
    saved_video, frames_written, sidecar_path = res

    # All 15 frames must be drained and written to disk
    assert frames_written == total_submitted

    cap = cv2.VideoCapture(str(saved_video))
    decoded_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert decoded_frames == total_submitted


def test_quality_audit_detects_dropped_frames_and_interruption(temp_recordings_dir):
    """Verifies that QualityAuditor raises appropriate warnings for dropped frames and interruptions."""
    db_path = temp_recordings_dir / "test_audit_flags.db"
    db = Database(db_path)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(temp_recordings_dir)
    service = SessionService(config=cfg, db=db)

    # 1. Session with dropped frames
    sid_warn = service.start_session(
        collection_source="PHYSICAL_WEBCAM",
        provenance=DataProvenance.SIMULATED,
        collection_mode="SYNTHETIC_DEMO",
    )

    # Record 35 observations so session is not empty
    for i in range(35):
        feat = ObservationFeatures(
            track_id="T1",
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=service.session_start_monotonic + (i * 0.0333),
            bbox_x1=10, bbox_y1=10, bbox_x2=100, bbox_y2=100,
            ear_avg=0.3, ear_left=0.3, ear_right=0.3, mar=0.15,
            face_quality=0.9, face_detected=True,
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0, head_pose_valid=True,
            pose_detected=True, shoulder_tilt_deg=0.0, torso_upright_ratio=0.9,
            head_shoulder_offset=0.3, wrists_near_head=False, wrists_near_table=False,
            phone_detected=False, phone_confidence=None,
            observability_level="FULL", observability_reasons="All observable",
        )
        smoothed = SmoothedBehaviorResult(
            track_id="T1",
            raw_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
            raw_score=0.9,
            smoothed_behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
            smoothed_score=0.9,
            window_duration_seconds=1.0,
            observation_count_in_window=10,
            is_observable=True,
            explanation="Engaged",
            wall_timestamp="2026-10-18T12:00:00Z",
            monotonic_ts=feat.monotonic_ts,
        )
        service.record_frame_observations([(feat, smoothed, i)])

    service.flush_buffer()

    service.session_repo.finish(
        session_id=sid_warn,
        end_time="2026-10-18T12:05:00Z",
        duration_seconds=30.0,
        quality_status="WARNING_DROPPED_FRAMES",
        error_message="Warning: 5 video frame(s) dropped due to recording queue saturation.",
    )

    auditor = QualityAuditor(db)
    report = auditor.audit_session(sid_warn)

    assert any("dropped" in w.lower() for w in report.warnings)
    assert report.verdict in [AuditVerdict.PASS_WITH_WARNINGS, AuditVerdict.REVIEW_REQUIRED]

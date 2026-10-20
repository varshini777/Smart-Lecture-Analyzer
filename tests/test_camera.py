"""Comprehensive regression and unit tests for CameraService, CaptureThread, and LiveMonitor preview."""

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, PropertyMock, patch
import numpy as np
import pytest

from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtWidgets import QApplication

from lecture_analyzer.config import AppConfig, CameraConfig
from lecture_analyzer.storage.database import Database
from lecture_analyzer.collection.session_service import SessionService
from lecture_analyzer.vision.camera import CameraLifecycleState, CameraService, FramePacket
from lecture_analyzer.ui.live_monitor import (
    CameraCaptureThread,
    CameraInferenceWorker,
    LiveMonitorTab,
    VideoDisplayWidget,
)


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication fixture for PySide6 widgets and thread event loops."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_db():
    """Provides an isolated temporary database for test sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_camera.db"
        db = Database(db_path)
        yield db


@pytest.fixture
def mock_frame():
    """Generates a standard 720p dummy color frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)



# ==============================================================================
# 1. CameraService Tests
# ==============================================================================

def test_camera_service_open_success(mock_frame):
    """Test successful camera opening with verified initial test frame."""
    cfg = CameraConfig(index=0, width=1280, height=720, backend="DEFAULT")
    service = CameraService(cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        opened = service.open()

    assert opened is True
    assert service.is_opened is True
    assert service.state == CameraLifecycleState.READY
    assert service.actual_width == 1280
    assert service.actual_height == 720
    assert service.first_frame_acquired is True
    assert service.last_error is None


def test_camera_service_open_failure_all_backends():
    """Test camera opening failure across all fallback backends."""
    cfg = CameraConfig(index=0, width=1280, height=720, backend="DEFAULT")
    service = CameraService(cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        opened = service.open()

    assert opened is False
    assert service.is_opened is False
    assert service.state == CameraLifecycleState.CLOSED
    assert service.cap is None
    assert service.last_error is not None
    assert "Failed to open video source" in service.last_error


def test_camera_service_initial_frame_read_failure_fallback(mock_frame):
    """Test fallback when first backend opens but fails to read a valid frame."""
    cfg = CameraConfig(index=0, width=1280, height=720, backend="DEFAULT")
    service = CameraService(cfg)

    # First attempt: isOpened() is True, but read() fails (e.g. format negotiation failure)
    cap_dshow = MagicMock()
    cap_dshow.isOpened.return_value = True
    cap_dshow.read.return_value = (False, None)

    # Second attempt (MSMF fallback): isOpened() is True and read() succeeds
    cap_msmf = MagicMock()
    cap_msmf.isOpened.return_value = True
    cap_msmf.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", side_effect=[cap_dshow, cap_msmf]):
        opened = service.open()

    assert opened is True
    assert service.is_opened is True
    assert service.actual_backend == "MSMF"
    cap_dshow.release.assert_called_once()


def test_camera_service_read_frame_success(mock_frame):
    """Test regular frame reading, metadata generation, and state transition to STREAMING."""
    cfg = CameraConfig(index=0, width=1280, height=720)
    service = CameraService(cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        service.open()
        assert service.state == CameraLifecycleState.READY

        packet = service.read_frame()
        assert packet is not None
        assert isinstance(packet, FramePacket)
        assert packet.frame_index == 1
        assert packet.width == 1280
        assert packet.height == 720
        assert packet.monotonic_ts > 0.0
        assert service.state == CameraLifecycleState.STREAMING


def test_camera_service_read_frame_failure():
    """Test handling when cap.read() returns failure during streaming."""
    cfg = CameraConfig(index=0, width=1280, height=720)
    service = CameraService(cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

    with patch("cv2.VideoCapture", return_value=mock_cap):
        service.open()
        # Next read fails
        mock_cap.read.return_value = (False, None)
        packet = service.read_frame()
        assert packet is None


def test_camera_service_read_frame_when_not_opened():
    """Test read_frame() called on an unopened camera returns None immediately."""
    service = CameraService()
    assert service.is_opened is False
    assert service.read_frame() is None


def test_camera_service_release_idempotent(mock_frame):
    """Test that release() cleanly closes handle and can be called repeatedly without errors."""
    service = CameraService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        service.open()
        assert service.is_opened is True

        service.release()
        assert service.is_opened is False
        assert service.cap is None
        assert service.state == CameraLifecycleState.CLOSED
        mock_cap.release.assert_called_once()

        # Repeated release must be safe
        service.release()
        assert service.state == CameraLifecycleState.CLOSED


def test_camera_service_repeated_open_attempts(mock_frame):
    """Test that open() safely closes previous capture before opening a new one."""
    service = CameraService()
    cap1 = MagicMock()
    cap1.isOpened.return_value = True
    cap1.read.return_value = (True, mock_frame)

    cap2 = MagicMock()
    cap2.isOpened.return_value = True
    cap2.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", side_effect=[cap1, cap2]):
        assert service.open() is True
        assert service.open() is True
        cap1.release.assert_called()


# ==============================================================================
# 2. CameraCaptureThread Tests
# ==============================================================================

def test_camera_capture_thread_lifecycle(qapp, temp_db, mock_frame):
    """Test CameraCaptureThread startup, signal emission, frame buffering, and clean stop."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    thread = CameraCaptureThread(cfg, sess)
    started_signals = []
    acquired_signals = []
    stopped_signals = []

    thread.camera_started.connect(lambda w, h, b: started_signals.append((w, h, b)))
    thread.frame_acquired.connect(lambda pkt: acquired_signals.append(pkt))
    thread.camera_stopped.connect(lambda: stopped_signals.append(True))

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        thread.start()

        # Wait for camera_started
        for _ in range(50):
            if thread.is_camera_ready and started_signals:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert len(started_signals) == 1
        assert started_signals[0][0] == 1280
        assert started_signals[0][1] == 720

        # Wait for at least 2 frames
        for _ in range(50):
            if len(acquired_signals) >= 2:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert len(acquired_signals) >= 2
        pkt = thread.get_latest_packet()
        assert pkt is not None
        assert pkt.width == 1280

        thread.stop()
        qapp.processEvents()
        for _ in range(50):
            if not thread.isRunning() and stopped_signals:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert not thread.isRunning()
        assert len(stopped_signals) == 1


def test_camera_capture_thread_open_failure_emits_error(qapp, temp_db):
    """Test CameraCaptureThread emits camera_error when open() fails."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    thread = CameraCaptureThread(cfg, sess)
    errors = []
    thread.camera_error.connect(lambda err: errors.append(err))

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        thread.start()
        for _ in range(50):
            if errors or not thread.isRunning():
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert len(errors) == 1
        assert "Failed to open video source" in errors[0]
        assert not thread.isRunning()


# ==============================================================================
# 3. CameraInferenceWorker Tests
# ==============================================================================

def test_camera_inference_worker_warmup_and_resilience(qapp, temp_db, mock_frame):
    """Test that CameraInferenceWorker reports warmup status and processes frames from shared thread."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    cap_thread = CameraCaptureThread(cfg, sess)
    cap_thread.is_camera_ready = True
    cap_thread._latest_packet = FramePacket(
        frame=mock_frame,
        frame_index=1,
        wall_timestamp="2026-10-18T00:00:00Z",
        monotonic_ts=time.monotonic(),
        acquisition_latency_ms=10.0,
        width=1280,
        height=720,
    )

    worker = CameraInferenceWorker(cfg, sess, cap_thread)
    statuses = []
    inferences = []

    worker.pipeline_status.connect(lambda st: statuses.append(st))
    worker.inference_processed.connect(lambda res, afps, pfps, lat, sk: inferences.append((pfps, lat)))

    mock_pipeline = MagicMock()
    mock_pipeline.warmup.return_value = None
    mock_result = MagicMock()
    mock_result.tracked_persons = {}
    mock_pipeline.process_frame.return_value = mock_result

    with patch("lecture_analyzer.ui.live_monitor.VisionPipeline", return_value=mock_pipeline):
        worker.start()

        # Wait for ready status
        for _ in range(50):
            if "ready" in statuses:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert "warming_up" in statuses
        assert "ready" in statuses

        # Wait for inference
        for _ in range(50):
            if inferences:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert len(inferences) >= 1
        worker.stop()


def test_camera_inference_worker_missing_model_resilience(qapp, temp_db):
    """Test that when VisionPipeline raises error, inference_error is emitted without crashing."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    cap_thread = CameraCaptureThread(cfg, sess)
    cap_thread.is_camera_ready = True

    worker = CameraInferenceWorker(cfg, sess, cap_thread)
    errors = []
    statuses = []

    worker.inference_error.connect(lambda e: errors.append(e))
    worker.pipeline_status.connect(lambda s: statuses.append(s))

    with patch("lecture_analyzer.ui.live_monitor.VisionPipeline", side_effect=RuntimeError("Model file missing")):
        worker.start()
        for _ in range(50):
            if "error" in statuses:
                break
            time.sleep(0.02)
            qapp.processEvents()

        assert "error" in statuses
        assert len(errors) == 1
        assert "Model file missing" in errors[0]
        worker.stop()


# ==============================================================================
# 4. VideoDisplayWidget & LiveMonitorTab UI Tests
# ==============================================================================

def test_video_display_widget_rendering(qapp, mock_frame):
    """Test VideoDisplayWidget pixmap generation, aspect scaling, and null reset."""
    widget = VideoDisplayWidget(mirror_display=True)
    assert widget._pixmap is None

    # Update with frame
    widget.update_frame(mock_frame, {}, is_warming_up=True)
    assert widget._pixmap is not None
    assert not widget._pixmap.isNull()
    assert widget._pixmap.width() == 1280
    assert widget._pixmap.height() == 720

    # Reset with None
    widget.update_frame(None, {}, is_warming_up=False)
    assert widget._pixmap is None


def test_live_monitor_camera_error_handling(qapp, temp_db):
    """Test that LiveMonitorTab resets UI cleanly when camera error occurs."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    tab = LiveMonitorTab(cfg, sess)

    with patch.object(tab, "stop_camera") as mock_stop:
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
            tab._on_camera_error("Webcam disconnected")
            mock_msg.assert_called_once()
            mock_stop.assert_called_once()


def test_camera_restart_after_failed_attempt(qapp, temp_db, mock_frame):
    """Test that after a failed camera initialization, restarting works when camera becomes available."""
    cfg = AppConfig()
    db = temp_db
    sess = SessionService(cfg, db)

    tab = LiveMonitorTab(cfg, sess)

    # First attempt: Camera fails to open
    mock_fail_cap = MagicMock()
    mock_fail_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_fail_cap):
        with patch("PySide6.QtWidgets.QMessageBox.critical"):
            tab.start_camera()
            for _ in range(50):
                if tab.capture_thread is None:
                    break
                time.sleep(0.02)
                qapp.processEvents()

    assert tab.capture_thread is None
    assert tab.btn_camera_toggle.text() == "Start Camera"

    # Second attempt: Camera is now available
    mock_ok_cap = MagicMock()
    mock_ok_cap.isOpened.return_value = True
    mock_ok_cap.read.return_value = (True, mock_frame)

    mock_pipeline = MagicMock()
    mock_pipeline.warmup.return_value = None
    mock_pipeline.process_frame.return_value = MagicMock(tracked_persons={})

    with patch("cv2.VideoCapture", return_value=mock_ok_cap):
        with patch("lecture_analyzer.ui.live_monitor.VisionPipeline", return_value=mock_pipeline):
            tab.start_camera()
            for _ in range(50):
                if tab._first_frame_displayed:
                    break
                time.sleep(0.02)
                qapp.processEvents()

            assert tab.capture_thread is not None
            assert tab.capture_thread.is_camera_ready is True
            assert tab._first_frame_displayed is True
            assert tab.btn_camera_toggle.text() == "Stop Camera"

            tab.stop_camera()
            assert tab.capture_thread is None


def test_camera_service_video_file_mode(mock_frame):
    """Test CameraService opening a recorded video file with pacing and looping."""
    cfg = CameraConfig()
    service = CameraService(cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else (100.0 if prop == 7 else 0.0)
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        opened = service.open("data/recordings/classroom_benchmark.mp4")

    assert opened is True
    assert service.is_file is True
    assert service.file_fps == 30.0
    assert service.total_file_frames == 100
    assert service.actual_backend == "FILE"

    pkt = service.read_frame(loop=True)
    assert pkt is not None
    assert pkt.width == 1280
    assert pkt.height == 720

    service.release()
    assert service.is_opened is False


def test_live_monitor_source_mode_toggle(qapp, temp_db):
    """Test that LiveMonitorTab toggles UI elements correctly between Webcam and Video File modes."""
    config = AppConfig()
    service = SessionService(config, temp_db)
    tab = LiveMonitorTab(config, service)


    # Initial state: Webcam mode
    assert tab.cmb_source_mode.currentIndex() == 0
    assert tab.video_container.isHidden() is True
    assert tab.btn_camera_toggle.text() == "Start Camera"
    assert tab.chk_mirror_preview.isChecked() is True

    # Switch to Video File mode
    tab.cmb_source_mode.setCurrentIndex(1)
    assert tab.video_container.isHidden() is False
    assert tab.btn_camera_toggle.text() == "Start Video Analysis"
    assert tab.chk_mirror_preview.isChecked() is False
    assert "classroom_benchmark.mp4" in tab.txt_video_path.text()

    # Preset selection test
    tab.cmb_presets.setCurrentIndex(1)
    assert "SES_" in tab.txt_video_path.text()

    # Switch back to Webcam mode
    tab.cmb_source_mode.setCurrentIndex(0)
    assert tab.video_container.isHidden() is True
    assert tab.btn_camera_toggle.text() == "Start Camera"



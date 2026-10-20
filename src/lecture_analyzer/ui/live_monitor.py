"""Live monitor tab featuring real-time camera inference, decoupled acquisition, visual overlays, and session controls."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from PySide6.QtCore import QMutex, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.behavior.feature_extraction import extract_features_from_state
from lecture_analyzer.behavior.labels import BehaviorCategory, is_concentrating, is_distracted
from lecture_analyzer.ui.video_audit_dialog import VideoAuditDialog, VideoAuditWorker

from lecture_analyzer.behavior.rules import BehaviorRuleEngine
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult, TemporalBehaviorSmoother
from lecture_analyzer.collection.session_service import SessionService
from lecture_analyzer.config import AppConfig
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.utils.timestamps import format_duration, monotonic_time
from lecture_analyzer.vision.camera import CameraService, FramePacket
from lecture_analyzer.vision.device_discovery import (
    CameraDeviceInfo,
    discover_cameras,
    get_recommended_camera_index,
)
from lecture_analyzer.vision.pipeline import FrameAnalysisResult, VisionPipeline

logger = logging.getLogger("lecture_analyzer.ui.live_monitor")


class ModelPrewarmWorker(QThread):
    """Loads and pre-warms neural models asynchronously in background on app startup."""
    pipeline_warmed = Signal(object)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            logger.info("Asynchronous background neural model pre-warming started...")
            pipeline = VisionPipeline(self.config)
            pipeline.warmup(640, 480)
            logger.info("Asynchronous background neural model pre-warming completed.")
            self.pipeline_warmed.emit(pipeline)
        except Exception as e:
            logger.warning(f"Background neural model pre-warming notice: {e}")

# Color palette for behavioral states
STATE_COLORS = {
    BehaviorCategory.ENGAGED_BEHAVIOR: QColor(46, 204, 113),     # Green
    BehaviorCategory.LOOKING_AWAY: QColor(241, 196, 15),        # Amber/Yellow
    BehaviorCategory.HEAD_DOWN: QColor(230, 126, 34),           # Orange
    BehaviorCategory.PHONE_USE: QColor(231, 76, 60),            # Red
    BehaviorCategory.WRITING_READING: QColor(52, 152, 219),     # Blue
    BehaviorCategory.DROWSY_SLEEP_LIKE: QColor(155, 89, 182),   # Purple
    BehaviorCategory.UNCERTAIN: QColor(149, 165, 166),          # Gray
}


class CameraCaptureThread(QThread):
    """Dedicated background thread acquiring physical camera frames at native hardware FPS.

    Decouples frame acquisition from neural network inference to eliminate buffer backlog
    and enable instantaneous live camera display.
    """
    frame_acquired = Signal(object)        # Emits FramePacket immediately upon acquisition
    camera_started = Signal(int, int, str) # width, height, backend
    camera_error = Signal(str)
    camera_stopped = Signal()

    def __init__(
        self,
        config: AppConfig,
        session_service: SessionService,
        source: Optional[int | str] = None,
        loop_video: bool = True,
    ):
        super().__init__()
        self.config = config
        self.session_service = session_service
        self.source = source
        self.loop_video = loop_video
        self.camera: Optional[CameraService] = None
        self._latest_packet: Optional[FramePacket] = None
        self._mutex = QMutex()
        self._running = False
        self.total_acquired = 0
        self.is_camera_ready = False

    def run(self):
        self._running = True
        self.total_acquired = 0
        logger.info(f"CameraCaptureThread started with source: {self.source}")

        try:
            self.camera = CameraService(self.config.camera)
            opened = self.camera.open(source=self.source)
            if not opened:
                err_msg = self.camera.last_error or f"Could not open video source {self.source}."
                self.camera_error.emit(err_msg)
                self._running = False
                return
            self.is_camera_ready = True
            logger.info(
                f"Camera opened successfully: {self.camera.actual_width}x{self.camera.actual_height} "
                f"via backend {self.camera.actual_backend}"
            )
            self.camera_started.emit(
                self.camera.actual_width,
                self.camera.actual_height,
                self.camera.actual_backend,
            )
        except Exception as e:
            logger.error(f"Camera open error in capture thread: {e}", exc_info=True)
            self.camera_error.emit(f"Camera open error: {e}")
            self._running = False
            return

        first_frame_logged = False
        consecutive_read_failures = 0
        max_read_failures = 15

        while self._running:
            packet = self.camera.read_frame(loop=self.loop_video)
            if packet is not None:

                consecutive_read_failures = 0
                if not first_frame_logged:
                    first_frame_logged = True
                    logger.info(
                        f"First frame received by CameraCaptureThread: "
                        f"{packet.width}x{packet.height} (latency={packet.acquisition_latency_ms:.1f}ms)"
                    )

                self._mutex.lock()
                self._latest_packet = packet
                self.total_acquired += 1
                self._mutex.unlock()

                # Immediately emit for live UI preview
                self.frame_acquired.emit(packet)

                # Record every acquired frame to session video if recording is active
                if self.session_service.is_session_active and self.session_service.recording_service.is_recording:
                    self.session_service.record_video_frame(packet)
            else:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    logger.error(f"Camera reached {max_read_failures} consecutive frame read failures.")
                    self.camera_error.emit("Camera device stopped delivering frames.")
                    break
                self.msleep(10)

        self.is_camera_ready = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_stopped.emit()
        logger.info("CameraCaptureThread stopped cleanly.")

    def get_latest_packet(self) -> Optional[FramePacket]:
        """Atomically retrieves the newest available frame packet."""
        self._mutex.lock()
        pkt = self._latest_packet
        self._mutex.unlock()
        return pkt

    @property
    def effective_fps(self) -> float:
        return self.camera.effective_fps if self.camera else 0.0

    def stop(self):
        self._running = False
        self.wait(2000)


class CameraInferenceWorker(QThread):
    """Inference thread consuming from latest-frame buffer without accumulating processing delay."""
    inference_processed = Signal(object, float, float, float, int)
    # Legacy / backward-compatibility signal: frame_bgr, results_dict, acq_fps, proc_fps, latency_ms, skipped_count
    frame_processed = Signal(object, object, float, float, float, int)
    pipeline_status = Signal(str)  # "warming_up", "ready", "error"
    inference_error = Signal(str)
    camera_error = Signal(str)

    def __init__(
        self,
        config: AppConfig,
        session_service: SessionService,
        capture_thread: Optional[CameraCaptureThread] = None,
        pipeline: Optional[VisionPipeline] = None,
        source: Optional[int | str] = None,
        loop_video: bool = True,
    ):
        super().__init__()
        self.config = config
        self.session_service = session_service
        self._owns_capture_thread = (capture_thread is None)
        self.capture_thread = capture_thread or CameraCaptureThread(
            config, session_service, source=source, loop_video=loop_video
        )
        self.pipeline: Optional[VisionPipeline] = pipeline
        self.rule_engine = BehaviorRuleEngine(config)
        self.smoother = TemporalBehaviorSmoother(config.behavior)
        self._running = False
        self.total_skipped = 0
        self.is_pipeline_ready = (pipeline is not None)


    def run(self):
        self._running = True
        logger.info("CameraInferenceWorker started.")

        if self._owns_capture_thread:
            logger.info("CameraInferenceWorker starting dedicated capture thread...")
            self.capture_thread.camera_error.connect(self.camera_error.emit)
            self.capture_thread.start()

        # Wait briefly for camera to open if not already open
        for _ in range(60):
            if not self._running:
                return
            if self.capture_thread.is_camera_ready or (
                self.capture_thread.camera and self.capture_thread.camera.is_opened
            ):
                break
            self.msleep(50)

        # Initialize pipeline if not passed in pre-warmed
        if self.pipeline is None:
            self.pipeline_status.emit("warming_up")
            logger.info("Initializing VisionPipeline and warming up neural perception models...")
            try:
                self.pipeline = VisionPipeline(self.config)
                self.pipeline.warmup(self.config.camera.width, self.config.camera.height)
                self.is_pipeline_ready = True
                self.pipeline_status.emit("ready")
                logger.info("VisionPipeline warmup completed successfully. Inference loop active.")
            except Exception as e:
                logger.error(f"Vision pipeline initialization error: {e}", exc_info=True)
                self.pipeline_status.emit("error")
                self.inference_error.emit(f"Vision pipeline initialization error: {e}")
                if self._owns_capture_thread:
                    self.camera_error.emit(f"Vision pipeline initialization error: {e}")
                    self._running = False
                    return
        else:
            self.is_pipeline_ready = True
            self.pipeline_status.emit("ready")
            logger.info("Using cached VisionPipeline. Inference loop active immediately.")

        last_processed_frame_index = -1
        last_proc_time: Optional[float] = None
        effective_proc_fps: float = 0.0

        while self._running:
            packet = self.capture_thread.get_latest_packet()
            if packet is None or packet.frame_index == last_processed_frame_index:
                self.msleep(10)
                continue

            if not self.is_pipeline_ready or self.pipeline is None:
                self.msleep(50)
                continue

            # Track frame skips
            if last_processed_frame_index != -1 and packet.frame_index > last_processed_frame_index + 1:
                self.total_skipped += (packet.frame_index - last_processed_frame_index - 1)
            last_processed_frame_index = packet.frame_index

            t_infer_start = monotonic_time()
            try:
                result = self.pipeline.process_frame(packet)
            except Exception as e:
                logger.error(f"Inference error in worker: {e}")
                continue

            t_infer_end = monotonic_time()
            proc_latency_ms = (t_infer_end - t_infer_start) * 1000.0

            # Calculate processing FPS with EMA smoothing
            if last_proc_time is not None:
                delta_proc = t_infer_end - last_proc_time
                if delta_proc > 0.001:
                    instant_proc_fps = 1.0 / delta_proc
                    if effective_proc_fps == 0.0:
                        effective_proc_fps = instant_proc_fps
                    else:
                        effective_proc_fps = 0.85 * effective_proc_fps + 0.15 * instant_proc_fps
            else:
                effective_proc_fps = (1000.0 / proc_latency_ms) if proc_latency_ms > 0 else 0.0
            last_proc_time = t_infer_end
            proc_fps = effective_proc_fps

            # Process behaviors for each active track
            smoothed_results: Dict[str, SmoothedBehaviorResult] = {}
            obs_to_record = []

            for tid, state in result.tracked_persons.items():
                feats = extract_features_from_state(state, packet.wall_timestamp, packet.monotonic_ts)
                rule_eval = self.rule_engine.evaluate(feats)
                smoothed = self.smoother.process(rule_eval)
                smoothed.bbox = state.bbox
                smoothed_results[tid] = smoothed


                if self.session_service.is_session_active:
                    obs_to_record.append((feats, smoothed, packet.frame_index))

            # Record to active session if active
            if obs_to_record and self.session_service.is_session_active:
                self.session_service.record_frame_observations(obs_to_record)

            acq_fps = self.capture_thread.effective_fps

            # Emit inference results
            self.inference_processed.emit(
                smoothed_results,
                acq_fps,
                proc_fps,
                proc_latency_ms,
                self.total_skipped,
            )
            # Emit legacy signal for backwards compatibility
            self.frame_processed.emit(
                packet.frame,
                smoothed_results,
                acq_fps,
                proc_fps,
                proc_latency_ms,
                self.total_skipped,
            )

        if self._owns_capture_thread:
            self.capture_thread.stop()
        logger.info("CameraInferenceWorker stopped cleanly.")

    def stop(self):
        self._running = False
        if self._owns_capture_thread:
            self.capture_thread.stop()
        self.wait(3000)


class VideoDisplayWidget(QLabel):
    """Custom QLabel that renders frame overlays with responsive aspect ratio scaling and safe buffers."""

    def __init__(self, mirror_display: bool = True):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #121212; border: 1px solid #2d2d2d; border-radius: 6px;")
        self.current_frame: Optional[np.ndarray] = None
        self.current_results: Dict[str, SmoothedBehaviorResult] = {}
        self.is_warming_up: bool = False
        self.status_message: str = "Camera inactive. Click 'Start Camera' to begin."
        self.mirror_display: bool = mirror_display
        self._pixmap: Optional[QPixmap] = None

    def set_status_message(self, msg: str):
        self.status_message = msg
        self.update()

    def set_mirror_display(self, mirror: bool):
        self.mirror_display = mirror

    def update_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        results: Dict[str, SmoothedBehaviorResult],
        is_warming_up: bool = False,
    ):
        self.current_frame = frame_bgr
        self.current_results = results
        self.is_warming_up = is_warming_up

        if frame_bgr is None:
            self._pixmap = None
            self.update()
            return

        h, w = frame_bgr.shape[:2]
        disp_bgr = frame_bgr.copy()

        # Optional user mirror comfort (disabled for recorded video files)
        if self.mirror_display:
            disp_bgr = cv2.flip(disp_bgr, 1)

        # 1. Aggregate Top Classroom Header Bar
        total_students = len(self.current_results)
        if total_students > 0:
            n_conc = sum(1 for r in self.current_results.values() if is_concentrating(r.smoothed_behavior))
            n_dist = sum(1 for r in self.current_results.values() if is_distracted(r.smoothed_behavior))
            pct_conc = (n_conc / total_students) * 100.0
            pct_dist = (n_dist / total_students) * 100.0

            header_text = f"STUDENTS: {total_students}  |  CONCENTRATING: {n_conc} ({pct_conc:.0f}%)  |  DISTRACTED: {n_dist} ({pct_dist:.0f}%)"
            # Draw dark semi-transparent header bar
            header_bar = disp_bgr[0:36, 0:w].copy()
            cv2.rectangle(disp_bgr, (0, 0), (w, 36), (18, 18, 18), -1)
            disp_bgr[0:36, 0:w] = cv2.addWeighted(disp_bgr[0:36, 0:w], 0.82, header_bar, 0.18, 0)
            status_text_col = (46, 204, 113) if pct_conc >= 50 else (60, 76, 231)
            cv2.putText(disp_bgr, header_text, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_text_col, 2, cv2.LINE_AA)

        # 2. Draw per-student bounding boxes and attention badges
        y_offset = 65 if total_students > 0 else 35

        for tid, res in self.current_results.items():
            beh = res.smoothed_behavior
            if is_concentrating(beh):
                category_label = "CONCENTRATING"
                bgr_col = (46, 204, 113)  # Green
            elif is_distracted(beh):
                category_label = "DISTRACTED"
                if beh == BehaviorCategory.LOOKING_AWAY:
                    bgr_col = (15, 196, 241)   # Amber
                elif beh == BehaviorCategory.PHONE_USE:
                    bgr_col = (60, 76, 231)    # Red
                elif beh == BehaviorCategory.HEAD_DOWN:
                    bgr_col = (34, 126, 230)   # Orange
                else:
                    bgr_col = (182, 89, 155)   # Purple
            else:
                category_label = "UNCERTAIN"
                bgr_col = (166, 165, 149)      # Gray

            if res.bbox is not None:
                x1, y1, x2, y2 = res.bbox
                if self.mirror_display:
                    bx1 = max(0, min(w - 1, w - 1 - x2))
                    bx2 = max(0, min(w - 1, w - 1 - x1))
                else:
                    bx1 = max(0, min(w - 1, x1))
                    bx2 = max(0, min(w - 1, x2))
                by1 = max(0, min(h - 1, y1))
                by2 = max(0, min(h - 1, y2))

                # Draw bounding box
                cv2.rectangle(disp_bgr, (bx1, by1), (bx2, by2), bgr_col, 2)

                # Draw person badge banner above box
                pill_text = f"{tid}: {category_label} ({beh.value})"
                (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                py1 = max(0, by1 - th - 8)
                py2 = by1
                px2 = min(w, bx1 + tw + 10)
                cv2.rectangle(disp_bgr, (bx1, py1), (px2, py2), bgr_col, -1)
                txt_color = (0, 0, 0) if bgr_col == (15, 196, 241) else (255, 255, 255)
                cv2.putText(disp_bgr, pill_text, (bx1 + 5, py2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, txt_color, 1, cv2.LINE_AA)
            else:
                # Fallback text if person has no bounding box
                label_text = f"{tid}: {category_label} - {res.smoothed_behavior.value} (score={res.smoothed_score:.2f})"
                cv2.putText(disp_bgr, label_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp_bgr, label_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr_col, 2, cv2.LINE_AA)
                y_offset += 28


        if self.is_warming_up:
            warmup_text = "AI Models Warming Up... Live Feed Active"
            cv2.putText(
                disp_bgr,
                warmup_text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                disp_bgr,
                warmup_text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        # Convert to QImage and make a deep copy to ensure memory safety
        rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, self.status_message)
            painter.end()
            return

        # Fast transformation prevents paint-time UI thread lag while preserving full-res pixmap
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()


class LiveMonitorTab(QWidget):
    """The Live Monitor tab containing video preview, telemetry, and collection controls."""

    def __init__(
        self,
        config: AppConfig,
        session_service: SessionService,
        auto_prewarm: bool = False,
    ):
        super().__init__()
        self.config = config
        self.session_service = session_service
        self.capture_thread: Optional[CameraCaptureThread] = None
        self.worker: Optional[CameraInferenceWorker] = None
        self._cached_pipeline: Optional[VisionPipeline] = None
        self._prewarm_worker: Optional[ModelPrewarmWorker] = None
        self._latest_results: Dict[str, SmoothedBehaviorResult] = {}
        self._is_warming_up: bool = False
        self._first_frame_displayed: bool = False
        self._is_updating_display: bool = False

        self._init_ui()

        if auto_prewarm:
            self.start_prewarm()

    def start_prewarm(self):
        """Starts asynchronous background neural model pre-warming."""
        if self._cached_pipeline is not None or self._prewarm_worker is not None:
            return
        self._prewarm_worker = ModelPrewarmWorker(self.config)
        self._prewarm_worker.pipeline_warmed.connect(self._on_pipeline_prewarmed)
        self._prewarm_worker.start()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left pane: Video Display
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.video_display = VideoDisplayWidget(mirror_display=self.config.vision.mirror_display)
        left_layout.addWidget(self.video_display)

        # Telemetry Status Bar under video
        status_bar = QFrame()
        status_bar.setStyleSheet("background-color: #1a1a1a; padding: 6px; border-radius: 4px;")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_acq_fps = QLabel("Acq: 0.0 FPS")
        self.lbl_proc_fps = QLabel("Proc: 0.0 FPS")
        self.lbl_latency = QLabel("Latency: 0 ms")
        self.lbl_skipped = QLabel("Skipped: 0")
        self.lbl_session_status = QLabel("Session: Inactive")
        self.lbl_session_status.setStyleSheet("color: #e74c3c; font-weight: bold;")

        sb_layout.addWidget(self.lbl_acq_fps)
        sb_layout.addWidget(QLabel("|"))
        sb_layout.addWidget(self.lbl_proc_fps)
        sb_layout.addWidget(QLabel("|"))
        sb_layout.addWidget(self.lbl_latency)
        sb_layout.addWidget(QLabel("|"))
        sb_layout.addWidget(self.lbl_skipped)
        sb_layout.addWidget(QLabel("|"))
        sb_layout.addWidget(self.lbl_session_status)
        sb_layout.addStretch()

        left_layout.addWidget(status_bar)
        splitter.addWidget(left_container)

        # Right pane: Controls & Telemetry
        right_panel = QWidget()
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Hardware & Video Input Control Box
        cam_box = QGroupBox("Video Input Source")
        cam_form = QVBoxLayout(cam_box)
        cam_form.setSpacing(6)

        # Source mode selector
        src_mode_row = QHBoxLayout()
        src_mode_row.addWidget(QLabel("Source Mode:"))
        self.cmb_source_mode = QComboBox()
        self.cmb_source_mode.addItems(["Live Webcam Device", "Recorded Video File"])
        self.cmb_source_mode.currentIndexChanged.connect(self._on_source_mode_changed)
        src_mode_row.addWidget(self.cmb_source_mode)
        cam_form.addLayout(src_mode_row)

        # Camera hardware device selection container (for Live Webcam mode)
        self.camera_device_container = QWidget()
        cam_dev_layout = QVBoxLayout(self.camera_device_container)
        cam_dev_layout.setContentsMargins(0, 2, 0, 2)
        cam_dev_layout.setSpacing(4)

        cam_sel_row = QHBoxLayout()
        cam_sel_row.addWidget(QLabel("Device:"))
        self.cmb_cameras = QComboBox()
        self.cmb_cameras.currentIndexChanged.connect(self._on_camera_device_selected)
        cam_sel_row.addWidget(self.cmb_cameras, 1)

        self.btn_refresh_cameras = QPushButton("Scan")
        self.btn_refresh_cameras.setToolTip("Scan system for connected cameras")
        self.btn_refresh_cameras.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        self.btn_refresh_cameras.clicked.connect(self._refresh_camera_devices)
        cam_sel_row.addWidget(self.btn_refresh_cameras)
        cam_dev_layout.addLayout(cam_sel_row)
        cam_form.addWidget(self.camera_device_container)

        # Video file selection container (for Recorded Video File mode)
        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0, 2, 0, 2)
        v_layout.setSpacing(5)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.cmb_presets = QComboBox()
        self.cmb_presets.addItem("Intel HD Classroom (4 Students)", "data/recordings/classroom_benchmark.mp4")
        self.cmb_presets.addItem("User Session 2 (Full Interaction)", "data/recordings/SES_20261019_054258_f420de.mp4")
        self.cmb_presets.addItem("User Session 1 (Initial Test)", "data/recordings/SES_20261019_054012_60551d.mp4")
        self.cmb_presets.addItem("Custom File...", "")
        self.cmb_presets.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.cmb_presets)
        v_layout.addLayout(preset_row)

        path_row = QHBoxLayout()
        self.txt_video_path = QLineEdit("data/recordings/classroom_benchmark.mp4")
        self.btn_browse_video = QPushButton("Browse...")
        self.btn_browse_video.clicked.connect(self._browse_video_file)
        path_row.addWidget(self.txt_video_path)
        path_row.addWidget(self.btn_browse_video)
        v_layout.addLayout(path_row)

        self.chk_loop_video = QCheckBox("Loop Video Playback Continuously")
        self.chk_loop_video.setChecked(True)
        v_layout.addWidget(self.chk_loop_video)

        self.btn_run_video_audit = QPushButton("Run Full Video Attention Audit")
        self.btn_run_video_audit.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; padding: 7px; border-radius: 4px;"
        )
        self.btn_run_video_audit.setToolTip("Process entire 2-5 min video to evaluate classroom concentration, student roster, and generate formal audit report.")
        self.btn_run_video_audit.clicked.connect(self._run_full_video_audit)
        v_layout.addWidget(self.btn_run_video_audit)

        self.video_container.setVisible(False)
        cam_form.addWidget(self.video_container)

        # Mirror display checkbox
        self.chk_mirror_preview = QCheckBox("Mirror Video Display")
        self.chk_mirror_preview.setChecked(self.config.vision.mirror_display)
        self.chk_mirror_preview.toggled.connect(self._on_mirror_toggled)
        cam_form.addWidget(self.chk_mirror_preview)

        self.lbl_cam_details = QLabel(f"Device: Index {self.config.camera.index} (Offline)")
        self.lbl_cam_details.setStyleSheet("color: #95a5a6; font-size: 11px;")
        cam_form.addWidget(self.lbl_cam_details)

        self.btn_camera_toggle = QPushButton("Start Camera")
        self.btn_camera_toggle.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
        self.btn_camera_toggle.clicked.connect(self._toggle_camera)
        cam_form.addWidget(self.btn_camera_toggle)

        right_layout.addWidget(cam_box)

        # Discover available cameras and auto-select laptop webcam
        self._refresh_camera_devices()

        # 2. Classroom Concentration Analytics Box
        class_box = QGroupBox("Classroom Attention Analytics")
        class_layout = QVBoxLayout(class_box)
        class_layout.setSpacing(6)

        # Metric Badges
        metrics_layout = QHBoxLayout()
        self.lbl_total_students = QLabel("Students: 0")
        self.lbl_total_students.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-weight: bold; padding: 4px 6px; border-radius: 4px;")
        self.lbl_concentrating_count = QLabel("Concentrating: 0 (0%)")
        self.lbl_concentrating_count.setStyleSheet("background-color: #1e4620; color: #2ecc71; font-weight: bold; padding: 4px 6px; border-radius: 4px;")
        self.lbl_distracted_count = QLabel("Distracted: 0 (0%)")
        self.lbl_distracted_count.setStyleSheet("background-color: #4a1c1a; color: #e74c3c; font-weight: bold; padding: 4px 6px; border-radius: 4px;")

        metrics_layout.addWidget(self.lbl_total_students)
        metrics_layout.addWidget(self.lbl_concentrating_count)
        metrics_layout.addWidget(self.lbl_distracted_count)
        class_layout.addLayout(metrics_layout)

        # Student Roster Table
        self.tbl_students = QTableWidget(0, 4)
        self.tbl_students.setHorizontalHeaderLabels(["ID", "Status", "Observable Action", "Score"])
        self.tbl_students.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_students.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_students.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_students.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl_students.setFixedHeight(115)
        self.tbl_students.setStyleSheet("background-color: #1a1a1a; gridline-color: #2d2d2d; color: #ecf0f1; font-size: 11px;")
        class_layout.addWidget(self.tbl_students)

        right_layout.addWidget(class_box)

        # 3. Research Data Collection Box
        coll_box = QGroupBox("Research Session Collection")
        coll_layout = QVBoxLayout(coll_box)

        self.chk_research_mode = QCheckBox("Enable Research Collection Mode")
        self.chk_research_mode.setChecked(True)
        coll_layout.addWidget(self.chk_research_mode)

        self.txt_participant_id = QLineEdit()
        self.txt_participant_id.setPlaceholderText("e.g. P001 (Anonymous ID)")
        coll_layout.addWidget(QLabel("Participant Code:"))
        coll_layout.addWidget(self.txt_participant_id)

        self.chk_consent = QCheckBox("Participant Consent Confirmed")
        self.chk_consent.setStyleSheet("color: #f39c12; font-weight: bold;")
        coll_layout.addWidget(self.chk_consent)

        self.chk_record_video = QCheckBox("Save Local Video (.mp4)")
        coll_layout.addWidget(self.chk_record_video)

        self.btn_session_toggle = QPushButton("Start Recording Session")
        self.btn_session_toggle.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 8px;")
        self.btn_session_toggle.clicked.connect(self._toggle_session)
        coll_layout.addWidget(self.btn_session_toggle)

        right_layout.addWidget(coll_box)

        # 4. Observability & Focus Track Evidence Box
        obs_box = QGroupBox("Focus Track Evidence")
        obs_layout = QVBoxLayout(obs_box)

        self.lbl_active_track = QLabel("Focus Track: None")
        self.lbl_behavior = QLabel("Behavior: --")
        self.lbl_behavior.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.lbl_score = QLabel("Heuristic Score: --")
        self.lbl_explanation = QLabel("Explanation: Visual perception idle.")
        self.lbl_explanation.setWordWrap(True)
        self.lbl_explanation.setStyleSheet("color: #bdc3c7; font-size: 11px;")

        obs_layout.addWidget(self.lbl_active_track)
        obs_layout.addWidget(self.lbl_behavior)
        obs_layout.addWidget(self.lbl_score)
        obs_layout.addWidget(self.lbl_explanation)

        right_layout.addWidget(obs_box)
        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setSizes([680, 380])
        main_layout.addWidget(splitter)


    @Slot(object)
    def _on_pipeline_prewarmed(self, pipeline: VisionPipeline):
        """Receives pre-warmed VisionPipeline from background worker."""
        self._cached_pipeline = pipeline
        self._is_warming_up = False
        logger.info("LiveMonitorTab received pre-warmed VisionPipeline. Ready for instant start.")

    def _refresh_camera_devices(self):
        """Discovers camera hardware and selects the built-in laptop webcam by default."""
        try:
            cams = discover_cameras()
            rec_idx = get_recommended_camera_index(cams)
            self.cmb_cameras.blockSignals(True)
            self.cmb_cameras.clear()
            selected_row = 0
            for row, cam in enumerate(cams):
                self.cmb_cameras.addItem(cam.display_name, cam.index)
                if cam.index == rec_idx:
                    selected_row = row
            if cams:
                self.cmb_cameras.setCurrentIndex(selected_row)
                self.config.camera.index = rec_idx
                self.lbl_cam_details.setText(f"Active Device: Camera {rec_idx} (Offline)")
            else:
                self.cmb_cameras.addItem("Default Camera (Index 0)", 0)
                self.lbl_cam_details.setText("No cameras detected. Fallback Index 0.")
            self.cmb_cameras.blockSignals(False)
        except Exception as e:
            logger.warning(f"Error enumerating cameras: {e}")

    @Slot(int)
    def _on_camera_device_selected(self, row: int):
        idx = self.cmb_cameras.currentData()
        if idx is not None:
            self.config.camera.index = int(idx)
            self.lbl_cam_details.setText(f"Selected Device: Camera {idx} (Offline)")

    @Slot(int)
    def _on_source_mode_changed(self, idx: int):
        is_video = (idx == 1)
        self.camera_device_container.setVisible(not is_video)
        self.video_container.setVisible(is_video)
        if is_video:
            self.chk_mirror_preview.setChecked(False)
            self.video_display.set_mirror_display(False)
            if self.capture_thread is None or not self.capture_thread.isRunning():
                self.btn_camera_toggle.setText("Start Video Analysis")
                self.lbl_cam_details.setText("Mode: Recorded Video File (Idle)")
        else:
            self.chk_mirror_preview.setChecked(self.config.vision.mirror_display)
            self.video_display.set_mirror_display(self.config.vision.mirror_display)
            if self.capture_thread is None or not self.capture_thread.isRunning():
                self.btn_camera_toggle.setText("Start Camera")
                cam_idx = self.cmb_cameras.currentData()
                cur = cam_idx if cam_idx is not None else self.config.camera.index
                self.lbl_cam_details.setText(f"Selected Device: Camera {cur} (Offline)")

    @Slot(int)
    def _on_preset_changed(self, idx: int):
        path = self.cmb_presets.currentData()
        if path:
            self.txt_video_path.setText(str(path))

    @Slot()
    def _browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File for Attention Analysis",
            "data/recordings",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if file_path:
            self.txt_video_path.setText(file_path)
            self.cmb_presets.setCurrentIndex(self.cmb_presets.count() - 1)

    @Slot()
    def _run_full_video_audit(self):
        path_str = self.txt_video_path.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Video Required", "Please select or browse for a valid video file.")
            return

        p = Path(path_str)
        if not p.is_file():
            QMessageBox.warning(self, "File Not Found", f"Specified video file does not exist:\n{path_str}")
            return

        # If live camera or video playback is active, stop it cleanly
        if self.capture_thread is not None and self.capture_thread.isRunning():
            self.stop_camera()

        # Initialize progress dialog
        self.audit_progress_dialog = QProgressDialog(
            f"Initializing perception models to audit {p.name}...",
            "Cancel",
            0,
            100,
            self,
        )
        self.audit_progress_dialog.setWindowTitle("Classroom Video Attention Audit")
        self.audit_progress_dialog.setWindowModality(Qt.WindowModal)
        self.audit_progress_dialog.setMinimumDuration(0)
        self.audit_progress_dialog.setValue(0)

        # Launch background audit worker
        self._audit_worker = VideoAuditWorker(
            video_path=str(p.resolve()),
            config=self.config,
            generate_annotated_video=True,
            frame_stride=2,
        )
        self._audit_worker.progress.connect(self._on_audit_progress)
        self._audit_worker.audit_completed.connect(self._on_audit_completed)
        self._audit_worker.audit_failed.connect(self._on_audit_failed)
        self.audit_progress_dialog.canceled.connect(self._cancel_audit)
        self._audit_worker.start()

    @Slot(int, int, float, str)
    def _on_audit_progress(self, cur: int, total: int, fps: float, msg: str):
        if hasattr(self, "audit_progress_dialog") and self.audit_progress_dialog is not None:
            pct = int((cur / total) * 100.0) if total > 0 else 0
            self.audit_progress_dialog.setLabelText(f"{msg}\nProgress: {pct}% | Speed: {fps:.1f} FPS")
            self.audit_progress_dialog.setValue(pct)

    @Slot(object)
    def _on_audit_completed(self, result):
        if hasattr(self, "audit_progress_dialog") and self.audit_progress_dialog is not None:
            self.audit_progress_dialog.setValue(100)
            self.audit_progress_dialog.close()

        # Update classroom badges with audit metrics
        self.lbl_total_students.setText(f"Students: {result.total_unique_students}")
        self.lbl_concentrating_count.setText(f"Concentrating: {result.overall_concentration_percentage:.1f}%")
        self.lbl_distracted_count.setText(f"Distracted: {result.overall_distraction_percentage:.1f}%")

        # Open comprehensive audit report dialog
        dlg = VideoAuditDialog(self, result)
        dlg.exec_()

    @Slot(str)
    def _on_audit_failed(self, error_msg: str):
        if hasattr(self, "audit_progress_dialog") and self.audit_progress_dialog is not None:
            self.audit_progress_dialog.close()
        QMessageBox.critical(self, "Audit Execution Error", f"Video attention audit encountered an error:\n{error_msg}")

    @Slot()
    def _cancel_audit(self):
        if hasattr(self, "_audit_worker") and self._audit_worker is not None and self._audit_worker.isRunning():
            self._audit_worker.terminate()
            self._audit_worker.wait(1000)
            logger.info("Video attention audit was cancelled by the user.")

    @Slot(bool)
    def _on_mirror_toggled(self, checked: bool):
        self.video_display.set_mirror_display(checked)

    def start_camera(self) -> bool:
        """Starts physical camera or video file capture thread and background inference worker cleanly."""
        if self.capture_thread is not None and self.capture_thread.isRunning():
            return True

        self._first_frame_displayed = False
        self._latest_results = {}
        self._is_warming_up = (self._cached_pipeline is None)

        is_video = (self.cmb_source_mode.currentIndex() == 1)
        if is_video:
            path_str = self.txt_video_path.text().strip()
            if not path_str:
                QMessageBox.warning(self, "Video File Required", "Please provide a valid video file path.")
                return False
            p = Path(path_str)
            if not p.is_file():
                QMessageBox.warning(self, "File Not Found", f"Video file not found:\n{path_str}")
                return False
            cam_source = str(p.resolve())
            loop_video = self.chk_loop_video.isChecked()
            self.video_display.set_mirror_display(self.chk_mirror_preview.isChecked())
            self.video_display.set_status_message(f"Loading video: {p.name}...")
            self.btn_camera_toggle.setText("Opening Video...")
        else:
            cam_idx = self.cmb_cameras.currentData()
            cam_source = int(cam_idx) if cam_idx is not None else self.config.camera.index
            loop_video = False
            self.video_display.set_mirror_display(self.chk_mirror_preview.isChecked())
            self.video_display.set_status_message(f"Opening Camera {cam_source}...")
            self.btn_camera_toggle.setText("Opening...")

        self.btn_camera_toggle.setEnabled(False)

        # 1. Start dedicated capture thread (sole hardware owner)
        self.capture_thread = CameraCaptureThread(
            self.config,
            self.session_service,
            source=cam_source,
            loop_video=loop_video,
        )
        self.capture_thread.frame_acquired.connect(self._on_frame_acquired)
        self.capture_thread.camera_started.connect(self._on_camera_started)
        self.capture_thread.camera_error.connect(self._on_camera_error)
        self.capture_thread.camera_stopped.connect(self._on_camera_stopped)
        self.capture_thread.start()

        # 2. Start inference worker using shared capture thread and cached pipeline if available
        self.worker = CameraInferenceWorker(
            self.config,
            self.session_service,
            self.capture_thread,
            pipeline=self._cached_pipeline,
        )
        self.worker.inference_processed.connect(self._on_inference_processed)
        self.worker.pipeline_status.connect(self._on_pipeline_status)
        self.worker.inference_error.connect(self._on_inference_error)
        self.worker.start()

        toggle_text = "Stop Video Analysis" if is_video else "Stop Camera"
        self.btn_camera_toggle.setText(toggle_text)
        self.btn_camera_toggle.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
        self.btn_camera_toggle.setEnabled(True)
        logger.info(f"LiveMonitorTab initiated capture thread (source={cam_source}) and inference worker.")
        return True

    def stop_camera(self):
        """Safely stops active workers, stops session if active, and resets UI."""
        if self.session_service.is_session_active:
            self._stop_session()

        if self.worker is not None:
            if self.worker.is_pipeline_ready and self.worker.pipeline is not None:
                self._cached_pipeline = self.worker.pipeline
            self.worker.stop()
            self.worker = None

        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None

        self._latest_results = {}
        self._is_warming_up = False
        self._first_frame_displayed = False

        is_video = (self.cmb_source_mode.currentIndex() == 1)
        toggle_text = "Start Video Analysis" if is_video else "Start Camera"
        self.btn_camera_toggle.setText(toggle_text)
        self.btn_camera_toggle.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
        self.btn_camera_toggle.setEnabled(True)
        if is_video:
            self.lbl_cam_details.setText("Mode: Recorded Video File (Offline)")
        else:
            cam_idx = self.cmb_cameras.currentData()
            cur = cam_idx if cam_idx is not None else self.config.camera.index
            self.lbl_cam_details.setText(f"Device: Camera {cur} (Offline)")
        self.lbl_cam_details.setStyleSheet("color: #95a5a6; font-size: 11px;")
        self.video_display.update_frame(None, {}, is_warming_up=False)
        self.video_display.set_status_message("Camera inactive. Click 'Start Camera' or 'Start Video Analysis' to begin.")
        self.lbl_acq_fps.setText("Acq: 0.0 FPS")
        self.lbl_proc_fps.setText("Proc: 0.0 FPS")
        self.lbl_latency.setText("Latency: 0 ms")
        self.lbl_skipped.setText("Skipped: 0")

        # Reset classroom analytics
        self.lbl_total_students.setText("Students: 0")
        self.lbl_concentrating_count.setText("Concentrating: 0 (0%)")
        self.lbl_distracted_count.setText("Distracted: 0 (0%)")
        self.tbl_students.setRowCount(0)
        logger.info("LiveMonitorTab stopped camera and reset preview.")

    @Slot()
    def _toggle_camera(self):
        if self.capture_thread is None or not self.capture_thread.isRunning():
            self.start_camera()
        else:
            self.stop_camera()

    @Slot(object)
    def _on_frame_acquired(self, packet: FramePacket):
        """Immediately renders camera frames without event-loop backlogging."""
        if not self._first_frame_displayed:
            self._first_frame_displayed = True
            logger.info(f"First frame displayed in UI: {packet.width}x{packet.height}")

        if self._is_updating_display:
            # Drop paint event if previous paint is still executing to guarantee zero UI lag
            return
        self._is_updating_display = True
        try:
            self.video_display.update_frame(
                packet.frame,
                self._latest_results,
                is_warming_up=self._is_warming_up,
            )
        finally:
            self._is_updating_display = False

        if self.capture_thread:
            self.lbl_acq_fps.setText(f"Acq: {self.capture_thread.effective_fps:.1f} FPS")

    @Slot(int, int, str)
    def _on_camera_started(self, width: int, height: int, backend: str):
        if backend == "FILE":
            src_name = Path(str(self.capture_thread.source)).name if self.capture_thread and self.capture_thread.source else "Video File"
            self.lbl_cam_details.setText(f"Active Video: {src_name} ({width}x{height})")
        else:
            self.lbl_cam_details.setText(f"Active Camera: {width}x{height} ({backend})")
        self.lbl_cam_details.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold;")
        logger.info(f"Camera/Video started: {width}x{height} via {backend}")

    @Slot()
    def _on_camera_stopped(self):
        is_video = (self.cmb_source_mode.currentIndex() == 1)
        if is_video:
            self.lbl_cam_details.setText("Mode: Recorded Video File (Offline)")
        else:
            self.lbl_cam_details.setText(f"Device: Index {self.config.camera.index} (Offline)")
        self.lbl_cam_details.setStyleSheet("color: #95a5a6; font-size: 11px;")
        logger.info("Camera stopped notification received.")

    @Slot(str)
    def _on_camera_error(self, err_msg: str):
        logger.error(f"LiveMonitor received camera error: {err_msg}")
        QMessageBox.critical(self, "Camera Error", err_msg)
        self.stop_camera()

    @Slot(str)
    def _on_pipeline_status(self, status: str):
        if status == "ready":
            self._is_warming_up = False
            logger.info("AI perception pipeline ready. Overlays active.")
        elif status == "warming_up":
            self._is_warming_up = True
            logger.info("AI models warming up in background. Live video remains active.")
        elif status == "error":
            self._is_warming_up = False
            logger.warning("AI models encountered error. Live video remains active.")

    @Slot(str)
    def _on_inference_error(self, err_msg: str):
        logger.warning(f"Inference error: {err_msg}")

    @Slot(object, float, float, float, int)
    def _on_inference_processed(
        self,
        results: Dict[str, SmoothedBehaviorResult],
        acq_fps: float,
        proc_fps: float,
        latency: float,
        skipped: int,
    ):
        self._latest_results = results
        self.lbl_proc_fps.setText(f"Proc: {proc_fps:.1f} FPS")
        self.lbl_latency.setText(f"Latency: {latency:.1f} ms")
        self.lbl_skipped.setText(f"Skipped: {skipped}")

        # Update Classroom Concentration Analytics
        total = len(results)
        self.lbl_total_students.setText(f"Students: {total}")
        if total > 0:
            n_conc = sum(1 for r in results.values() if is_concentrating(r.smoothed_behavior))
            n_dist = sum(1 for r in results.values() if is_distracted(r.smoothed_behavior))
            p_conc = (n_conc / total) * 100.0
            p_dist = (n_dist / total) * 100.0
            self.lbl_concentrating_count.setText(f"Concentrating: {n_conc} ({p_conc:.0f}%)")
            self.lbl_distracted_count.setText(f"Distracted: {n_dist} ({p_dist:.0f}%)")
        else:
            self.lbl_concentrating_count.setText("Concentrating: 0 (0%)")
            self.lbl_distracted_count.setText("Distracted: 0 (0%)")

        # Update Live Student Roster Table
        self.tbl_students.setRowCount(total)
        for row, (tid, res) in enumerate(results.items()):
            beh = res.smoothed_behavior
            if is_concentrating(beh):
                stat_text = "CONCENTRATING"
                fg_col = QColor(46, 204, 113)
            elif is_distracted(beh):
                stat_text = "DISTRACTED"
                fg_col = QColor(231, 76, 60) if beh == BehaviorCategory.PHONE_USE else QColor(241, 196, 15)
            else:
                stat_text = "UNCERTAIN"
                fg_col = QColor(149, 165, 166)

            item_id = QTableWidgetItem(tid)
            item_stat = QTableWidgetItem(stat_text)
            item_stat.setForeground(fg_col)
            item_stat.setFont(QFont("Segoe UI", 9, QFont.Bold))
            item_act = QTableWidgetItem(beh.value)
            item_score = QTableWidgetItem(f"{res.smoothed_score:.2f}")

            self.tbl_students.setItem(row, 0, item_id)
            self.tbl_students.setItem(row, 1, item_stat)
            self.tbl_students.setItem(row, 2, item_act)
            self.tbl_students.setItem(row, 3, item_score)

        # Update Focus Track Detail Telemetry
        if results:
            first_tid = next(iter(results.keys()))
            res = results[first_tid]
            self.lbl_active_track.setText(f"Focus Track: {first_tid}")
            color = STATE_COLORS.get(res.smoothed_behavior, QColor(200, 200, 200)).name()
            self.lbl_behavior.setText(f"Behavior: {res.smoothed_behavior.value}")
            self.lbl_behavior.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
            self.lbl_score.setText(f"Heuristic Rule Score: {res.smoothed_score:.2f}")
            self.lbl_explanation.setText(f"Explanation: {res.explanation}")
        else:
            self.lbl_active_track.setText("Focus Track: None")
            self.lbl_behavior.setText("Behavior: --")
            self.lbl_score.setText("Heuristic Rule Score: --")
            self.lbl_explanation.setText("Explanation: No person currently detected in frame.")


    @Slot(object, object, float, float, float, int)
    def _on_frame_processed(
        self,
        frame_bgr: np.ndarray,
        results: Dict[str, SmoothedBehaviorResult],
        acq_fps: float,
        proc_fps: float,
        latency: float,
        skipped: int,
    ):
        """Backward-compatibility slot for direct worker connections."""
        self._on_inference_processed(results, acq_fps, proc_fps, latency, skipped)

    @Slot()
    def _toggle_session(self):
        if not self.session_service.is_session_active:
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self):
        if self.capture_thread is None or not self.capture_thread.is_camera_ready:
            if self.capture_thread is not None and self.capture_thread.isRunning():
                QMessageBox.information(
                    self,
                    "Camera Initializing",
                    "The camera is currently initializing. Please wait until the live feed appears before starting a session."
                )
                return

            reply = QMessageBox.question(
                self,
                "Camera Required",
                "Live camera feed is not running. Would you like to start the camera now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.start_camera()
                QMessageBox.information(
                    self,
                    "Camera Starting",
                    "Camera is starting up. Once the live preview is visible, click 'Start Recording Session' again."
                )
            return

        is_research = self.chk_research_mode.isChecked()
        participant_id = self.txt_participant_id.text().strip()
        consent_ok = self.chk_consent.isChecked()
        record_video = self.chk_record_video.isChecked()

        is_video = (self.cmb_source_mode.currentIndex() == 1)

        if is_research:
            if is_video:
                QMessageBox.warning(
                    self,
                    "Live Camera Required",
                    "Scientific research data collection requires a genuine live camera feed with confirmed participant consent."
                )
                return
            if not consent_ok:
                QMessageBox.warning(self, "Consent Required", "Participant consent must be confirmed to start research collection.")
                return
            if not participant_id:
                QMessageBox.warning(self, "Participant ID Required", "Please provide an anonymous participant code (e.g. P001).")
                return
            prov = DataProvenance.REAL_HUMAN
            coll_mode = "LIVE_CAMERA"
            coll_src = "PHYSICAL_WEBCAM"
        else:
            prov = DataProvenance.PIPELINE_VALIDATION if is_video else DataProvenance.SIMULATED
            coll_mode = "VIDEO_FILE_BENCHMARK" if is_video else "SYNTHETIC_DEMO"
            coll_src = "RECORDED_VIDEO_FILE" if is_video else "PHYSICAL_WEBCAM"

        actual_w = self.capture_thread.camera.actual_width if (self.capture_thread and self.capture_thread.camera) else self.config.camera.width
        actual_h = self.capture_thread.camera.actual_height if (self.capture_thread and self.capture_thread.camera) else self.config.camera.height

        try:
            sid = self.session_service.start_session(
                collection_source=coll_src,
                provenance=prov,
                collection_mode=coll_mode,
                participant_code=participant_id if is_research else None,
                consent_confirmed=consent_ok,
                record_video=record_video,
                frame_width=actual_w,
                frame_height=actual_h,
            )

            self.lbl_session_status.setText(f"Session: ACTIVE ({sid})")
            self.lbl_session_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.btn_session_toggle.setText("Stop Recording Session")
            self.btn_session_toggle.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
            self.txt_participant_id.setEnabled(False)
            self.chk_consent.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Session Start Error", str(e))

    def _stop_session(self):
        if self.session_service.is_session_active:
            completed = self.session_service.stop_session()
            self.lbl_session_status.setText("Session: Inactive")
            self.lbl_session_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.btn_session_toggle.setText("Start Recording Session")
            self.btn_session_toggle.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 8px;")
            self.txt_participant_id.setEnabled(True)
            self.chk_consent.setEnabled(True)

            if completed:
                QMessageBox.information(
                    self,
                    "Session Finished",
                    f"Session {completed.session_id} successfully saved.\n"
                    f"Duration: {completed.duration_seconds:.1f}s\n"
                    f"Provenance: {completed.provenance}\n"
                    f"Recorded Observations: {self.session_service.obs_repo.count(completed.session_id)}"
                )

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

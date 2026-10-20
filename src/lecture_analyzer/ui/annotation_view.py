"""Independent human annotation tab featuring video playback, timeline seeking, and interval labeling."""

import logging
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.annotation.annotation_service import AnnotationService
from lecture_analyzer.behavior.labels import ALL_BEHAVIOR_LABELS
from lecture_analyzer.collection.recording_service import (
    frame_idx_for_time,
    load_timestamp_sidecar,
    time_for_frame_idx,
)
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import SessionRepository
from lecture_analyzer.utils.paths import resolve_path

logger = logging.getLogger("lecture_analyzer.ui.annotation_view")


def format_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS.S display."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


class AnnotationTab(QWidget):
    """Tab providing independent ground-truth human labeling decoupled from automated inference."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.session_repo = SessionRepository(db)
        self.annot_service = AnnotationService(db)
        self.current_session_id: Optional[str] = None

        # Video playback state
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[Path] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._duration_s: float = 0.0
        self._current_pos_s: float = 0.0
        self._is_playing: bool = False

        # Precision timestamp sidecar state
        self._sidecar: Optional[dict] = None
        self._frame_elapsed_times: list[float] = []

        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._init_ui()
        self.refresh_sessions()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # ==============================================================================
        # Left Panel: Session Selector, Form Inputs, Coverage Summary
        # ==============================================================================
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. Target Session Selector Box
        sess_box = QGroupBox("Target Session")
        sess_layout = QVBoxLayout(sess_box)

        self.cmb_sessions = QComboBox()
        self.cmb_sessions.currentIndexChanged.connect(self._on_session_changed)
        sess_layout.addWidget(QLabel("Select Recorded Session:"))
        sess_layout.addWidget(self.cmb_sessions)

        self.lbl_session_info = QLabel("Duration: -- | Video: None")
        self.lbl_session_info.setStyleSheet("color: #95a5a6; font-size: 11px;")
        sess_layout.addWidget(self.lbl_session_info)
        left_layout.addWidget(sess_box)

        # 2. Human Ground-Truth Form
        input_box = QGroupBox("Record Human Ground-Truth")
        form = QFormLayout(input_box)

        # Start Time with quick-sync button
        start_row = QHBoxLayout()
        self.spn_start = QDoubleSpinBox()
        self.spn_start.setRange(0.0, 36000.0)
        self.spn_start.setSingleStep(0.5)
        self.spn_start.setSuffix(" s")
        self.btn_set_start = QPushButton("Use Playhead")
        self.btn_set_start.setToolTip("Set start time from current video playback timestamp")
        self.btn_set_start.clicked.connect(self._set_start_from_playhead)
        start_row.addWidget(self.spn_start)
        start_row.addWidget(self.btn_set_start)
        form.addRow("Start Timestamp:", start_row)

        # End Time with quick-sync button
        end_row = QHBoxLayout()
        self.spn_end = QDoubleSpinBox()
        self.spn_end.setRange(0.0, 36000.0)
        self.spn_end.setSingleStep(0.5)
        self.spn_end.setSuffix(" s")
        self.btn_set_end = QPushButton("Use Playhead")
        self.btn_set_end.setToolTip("Set end time from current video playback timestamp")
        self.btn_set_end.clicked.connect(self._set_end_from_playhead)
        end_row.addWidget(self.spn_end)
        end_row.addWidget(self.btn_set_end)
        form.addRow("End Timestamp:", end_row)

        self.cmb_label = QComboBox()
        self.cmb_label.addItems(ALL_BEHAVIOR_LABELS)

        self.txt_annotator = QLineEdit()
        self.txt_annotator.setPlaceholderText("e.g. ANNOTATOR_1")

        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText("Optional observations context")

        form.addRow("Observed Behavior:", self.cmb_label)
        form.addRow("Annotator ID:", self.txt_annotator)
        form.addRow("Notes:", self.txt_notes)

        self.btn_save = QPushButton("Save Ground-Truth Annotation")
        self.btn_save.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
        self.btn_save.clicked.connect(self._save_annotation)
        form.addRow(self.btn_save)

        left_layout.addWidget(input_box)

        # 3. Coverage Summary Box
        cov_box = QGroupBox("Annotation Coverage")
        cov_layout = QVBoxLayout(cov_box)
        self.lbl_coverage = QLabel("Coverage: 0.0% (0.0s / 0.0s)")
        self.lbl_overlap_warn = QLabel("")
        self.lbl_overlap_warn.setStyleSheet("color: #e74c3c; font-weight: bold;")
        cov_layout.addWidget(self.lbl_coverage)
        cov_layout.addWidget(self.lbl_overlap_warn)
        left_layout.addWidget(cov_box)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ==============================================================================
        # Right Panel: Video Player, Timeline Slider, Stored Annotations Table
        # ==============================================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Video Player Box
        video_box = QGroupBox("Session Video Review")
        video_layout = QVBoxLayout(video_box)
        video_layout.setContentsMargins(8, 8, 8, 8)
        video_layout.setSpacing(6)

        self.lbl_video_display = QLabel("No video recording loaded.")
        self.lbl_video_display.setAlignment(Qt.AlignCenter)
        self.lbl_video_display.setMinimumHeight(240)
        self.lbl_video_display.setStyleSheet("background-color: #1a1a1a; color: #7f8c8d; font-size: 13px; border-radius: 4px;")
        video_layout.addWidget(self.lbl_video_display)

        # Playback Controls Bar
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self._toggle_playback)

        self.btn_step_back = QPushButton("⏪ -1s")
        self.btn_step_back.setFixedWidth(65)
        self.btn_step_back.clicked.connect(lambda: self._seek_relative(-1.0))

        self.btn_step_fwd = QPushButton("+1s ⏩")
        self.btn_step_fwd.setFixedWidth(65)
        self.btn_step_fwd.clicked.connect(lambda: self._seek_relative(1.0))

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 1000)
        self.slider_timeline.sliderMoved.connect(self._on_slider_moved)

        self.lbl_playback_time = QLabel("00:00.0 / 00:00.0 (0.0s)")
        self.lbl_playback_time.setStyleSheet("font-family: monospace; font-size: 11px; min-width: 140px;")

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_step_back)
        ctrl_layout.addWidget(self.btn_step_fwd)
        ctrl_layout.addWidget(self.slider_timeline)
        ctrl_layout.addWidget(self.lbl_playback_time)
        video_layout.addLayout(ctrl_layout)

        right_layout.addWidget(video_box)

        # Annotations Table Header & Actions
        tbl_header = QHBoxLayout()
        tbl_header.addWidget(QLabel("Stored Annotations for Selected Session:"))
        tbl_header.addStretch()
        self.btn_delete_annot = QPushButton("Delete Selected")
        self.btn_delete_annot.setStyleSheet("background-color: #c0392b; color: white;")
        self.btn_delete_annot.clicked.connect(self._delete_annotation)
        tbl_header.addWidget(self.btn_delete_annot)
        right_layout.addLayout(tbl_header)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID", "Start (s)", "End (s)", "Behavior Label", "Annotator", "Notes"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        right_layout.addWidget(self.table)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)

    # ==============================================================================
    # Video Playback Engine
    # ==============================================================================

    def _close_current_video(self):
        """Releases active video capture and stops playback timer."""
        self._playback_timer.stop()
        self._is_playing = False
        self.btn_play.setText("▶ Play")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = None
        self._total_frames = 0
        self._duration_s = 0.0
        self._current_pos_s = 0.0
        self._sidecar = None
        self._frame_elapsed_times = []

    def _load_video_for_session(self, video_path_str: Optional[str], session_duration: float):
        """Initializes video stream and timeline slider for the session recording."""
        self._close_current_video()

        if not video_path_str:
            self.lbl_video_display.setText("No video recording file recorded for this session.")
            self.slider_timeline.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.lbl_playback_time.setText(f"00:00.0 / {format_seconds(session_duration)}")
            return

        resolved = resolve_path(video_path_str)
        if not resolved.exists():
            self.lbl_video_display.setText(f"Video file not found at: {resolved.name}")
            self.slider_timeline.setEnabled(False)
            self.btn_play.setEnabled(False)
            return

        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            self.lbl_video_display.setText(f"Could not open video file: {resolved.name}")
            self.slider_timeline.setEnabled(False)
            self.btn_play.setEnabled(False)
            return

        # Check for precision timestamp sidecar
        sidecar = load_timestamp_sidecar(resolved)
        if sidecar:
            self._sidecar = sidecar
            self._frame_elapsed_times = [float(f["elapsed_s"]) for f in sidecar.get("frames", [])]
            logger.info(f"Loaded timestamp sidecar with {len(self._frame_elapsed_times)} frames for {resolved.name}")
        else:
            self._sidecar = None
            self._frame_elapsed_times = []

        self._cap = cap
        self._video_path = resolved
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1.0 else 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._sidecar and self._sidecar.get("session_duration_s"):
            self._duration_s = float(self._sidecar["session_duration_s"])
        elif self._frame_elapsed_times:
            self._duration_s = self._frame_elapsed_times[-1]
        elif self._fps > 0:
            self._duration_s = self._total_frames / self._fps
        else:
            self._duration_s = session_duration

        if self._duration_s <= 0.0:
            self._duration_s = max(session_duration, 1.0)

        self.slider_timeline.setEnabled(True)
        self.slider_timeline.setRange(0, max(1, self._total_frames - 1))
        self.slider_timeline.setValue(0)
        self.btn_play.setEnabled(True)

        # Show first frame
        self._show_frame_at_index(0)

    def _show_frame_at_index(self, frame_idx: int):
        """Seeks to the given frame and renders it in the QLabel."""
        if self._cap is None:
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if ret and frame is not None:
            if self._frame_elapsed_times and 0 <= frame_idx < len(self._frame_elapsed_times):
                self._current_pos_s = self._frame_elapsed_times[frame_idx]
            else:
                self._current_pos_s = frame_idx / self._fps
            self._render_frame(frame)
            self._update_time_label()

    def _render_frame(self, frame):
        """Converts OpenCV BGR image to QPixmap and displays it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        self.lbl_video_display.setPixmap(
            pix.scaled(
                self.lbl_video_display.width(),
                self.lbl_video_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _update_time_label(self):
        cur_str = format_seconds(self._current_pos_s)
        dur_str = format_seconds(self._duration_s)
        self.lbl_playback_time.setText(f"{cur_str} / {dur_str} ({self._current_pos_s:.1f}s)")

    @Slot()
    def _toggle_playback(self):
        if self._cap is None:
            return

        if self._is_playing:
            self._playback_timer.stop()
            self._is_playing = False
            self.btn_play.setText("▶ Play")
        else:
            interval_ms = max(10, int(1000.0 / self._fps))
            self._playback_timer.start(interval_ms)
            self._is_playing = True
            self.btn_play.setText("⏸ Pause")

    def _on_playback_tick(self):
        if self._cap is None:
            self._playback_timer.stop()
            return

        ret, frame = self._cap.read()
        if not ret or frame is None:
            # EOF reached
            self._playback_timer.stop()
            self._is_playing = False
            self.btn_play.setText("▶ Play")
            return

        cur_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self._frame_elapsed_times and 0 <= cur_frame < len(self._frame_elapsed_times):
            self._current_pos_s = self._frame_elapsed_times[cur_frame]
        else:
            self._current_pos_s = cur_frame / self._fps
        self._render_frame(frame)
        self._update_time_label()

        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(cur_frame)
        self.slider_timeline.blockSignals(False)

    @Slot(int)
    def _on_slider_moved(self, value: int):
        if self._cap is None:
            return
        self._show_frame_at_index(value)

    def _seek_relative(self, delta_s: float):
        if self._cap is None:
            return
        target_s = max(0.0, min(self._duration_s, self._current_pos_s + delta_s))
        self._seek_to_time(target_s)

    def _seek_to_time(self, target_s: float):
        """Seeks the video player to the closest frame for a given session elapsed timestamp."""
        if self._cap is None:
            return
        clamped_s = max(0.0, min(self._duration_s, target_s))
        if self._sidecar:
            frame_idx = frame_idx_for_time(self._sidecar, clamped_s)
        else:
            frame_idx = int(clamped_s * self._fps)
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(frame_idx)
        self.slider_timeline.blockSignals(False)
        self._show_frame_at_index(frame_idx)

    @Slot(int, int)
    def _on_table_double_clicked(self, row: int, column: int):
        """Double-clicking an annotation row seeks the video to its start timestamp."""
        start_item = self.table.item(row, 1)
        if start_item:
            try:
                start_s = float(start_item.text())
                self._seek_to_time(start_s)
            except ValueError:
                pass

    @Slot()
    def _set_start_from_playhead(self):
        self.spn_start.setValue(round(self._current_pos_s, 2))

    @Slot()
    def _set_end_from_playhead(self):
        self.spn_end.setValue(round(self._current_pos_s, 2))

    # ==============================================================================
    # Session Management & Table Updates
    # ==============================================================================

    def refresh_sessions(self):
        """Populates the session selector with sessions from SQLite."""
        self.cmb_sessions.clear()
        sessions = self.session_repo.list_all()
        for s in sessions:
            self.cmb_sessions.addItem(f"{s.session_id} ({s.participant_code or 'ANON'}, {s.provenance})", s.session_id)

    @Slot(int)
    def _on_session_changed(self, index: int):
        if index < 0:
            self.current_session_id = None
            self._close_current_video()
            return

        sid = self.cmb_sessions.currentData()
        self.current_session_id = sid

        sess = self.session_repo.get(sid)
        if sess:
            dur = sess.duration_seconds or 0.0
            rec = sess.recording_path or "No video"
            self.lbl_session_info.setText(f"Duration: {dur:.1f}s | Video: {rec}")
            self.spn_start.setValue(0.0)
            self.spn_end.setValue(min(5.0, dur if dur > 0 else 5.0))
            self._load_video_for_session(sess.recording_path, dur)
        else:
            self._close_current_video()

        self.refresh_annotations()

    def refresh_annotations(self):
        """Loads and displays annotations for the active session."""
        if not self.current_session_id:
            self.table.setRowCount(0)
            return

        annots = self.annot_service.list_annotations(self.current_session_id)
        self.table.setRowCount(len(annots))

        for row, a in enumerate(annots):
            self.table.setItem(row, 0, QTableWidgetItem(str(a.id)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{a.start_seconds:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{a.end_seconds:.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(a.behavior_label))
            self.table.setItem(row, 4, QTableWidgetItem(a.annotator_id))
            self.table.setItem(row, 5, QTableWidgetItem(a.notes or ""))

        # Update coverage
        cov = self.annot_service.get_coverage(self.current_session_id)
        if cov:
            self.lbl_coverage.setText(
                f"Coverage: {cov.coverage_fraction * 100.0:.1f}% ({cov.total_annotated_seconds:.1f}s / {cov.total_session_seconds:.1f}s)"
            )
            if cov.has_overlaps:
                self.lbl_overlap_warn.setText(f"Warning: {len(cov.overlapping_intervals)} overlapping interval(s) detected!")
            else:
                self.lbl_overlap_warn.setText("")

    @Slot()
    def _save_annotation(self):
        if not self.current_session_id:
            QMessageBox.warning(self, "No Session", "Please select a target session.")
            return

        start_s = self.spn_start.value()
        end_s = self.spn_end.value()
        label = self.cmb_label.currentText()
        annotator = self.txt_annotator.text().strip()
        notes = self.txt_notes.text().strip()

        if not annotator:
            QMessageBox.warning(self, "Annotator ID Required", "Please enter an Annotator ID.")
            return

        if end_s <= start_s:
            QMessageBox.warning(self, "Invalid Range", "End time must be strictly greater than start time.")
            return

        try:
            self.annot_service.add_annotation(
                session_id=self.current_session_id,
                start_seconds=start_s,
                end_seconds=end_s,
                behavior_label=label,
                annotator_id=annotator,
                notes=notes,
            )
            # Advance start time to previous end time for convenient sequential entry
            self.spn_start.setValue(end_s)
            self.spn_end.setValue(end_s + 5.0)
            self.refresh_annotations()
        except Exception as e:
            QMessageBox.critical(self, "Annotation Error", str(e))

    @Slot()
    def _delete_annotation(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        annot_id_item = self.table.item(row, 0)
        if not annot_id_item:
            return

        annot_id = int(annot_id_item.text())
        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete annotation #{annot_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.annot_service.delete_annotation(annot_id)
            self.refresh_annotations()

    def closeEvent(self, event):
        """Cleanup on close."""
        self._close_current_video()
        super().closeEvent(event)

import json
import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.analytics.session_metrics import compute_session_metrics
from lecture_analyzer.config import AppConfig
from lecture_analyzer.dataset.quality_audit import QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import SessionRecord, SessionRepository
from lecture_analyzer.ui.video_audit_dialog import VideoAuditDialog, VideoAuditWorker

logger = logging.getLogger("lecture_analyzer.ui.session_browser")


class SessionBrowserTab(QWidget):
    """Tab for inspecting stored sessions, behavioral metrics, and running quality audits."""

    def __init__(self, db: Database, config: Optional[AppConfig] = None):
        super().__init__()
        self.db = db
        self.config = config or AppConfig()
        self.session_repo = SessionRepository(db)
        self.auditor = QualityAuditor(db)
        self.selected_session_id: Optional[str] = None

        self._init_ui()
        self.refresh_sessions()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_sessions)
        self.btn_audit = QPushButton("Run Quality Audit")
        self.btn_audit.clicked.connect(self._run_audit)

        self.btn_audit_video = QPushButton("Audit Video File...")
        self.btn_audit_video.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        self.btn_audit_video.setToolTip("Select any 2-5 min recorded video to analyze concentration, student roster, and generate audit report.")
        self.btn_audit_video.clicked.connect(self._audit_external_video)

        self.btn_delete = QPushButton("Delete Session")
        self.btn_delete.setStyleSheet("background-color: #c0392b; color: white;")
        self.btn_delete.clicked.connect(self._delete_session)

        toolbar.addWidget(self.btn_refresh)
        toolbar.addWidget(self.btn_audit)
        toolbar.addWidget(self.btn_audit_video)
        toolbar.addWidget(self.btn_delete)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Splitter: Table on left, details on right
        splitter = QSplitter(Qt.Horizontal)

        # Left: Sessions Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Session ID", "Participant", "Provenance", "Mode", "Duration", "Observations", "Quality"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_session_selected)
        splitter.addWidget(self.table)

        # Right: Detail & Analytics Viewer
        self.txt_details = QTextEdit()
        self.txt_details.setReadOnly(True)
        self.txt_details.setFontFamily("Consolas")
        self.txt_details.setStyleSheet("background-color: #1a1a1a; color: #ecf0f1; padding: 8px;")
        splitter.addWidget(self.txt_details)

        splitter.setSizes([600, 400])
        main_layout.addWidget(splitter)

    def refresh_sessions(self):
        """Reloads all sessions from the SQLite database."""
        sessions = self.session_repo.list_all()
        self.table.setRowCount(len(sessions))

        for row, s in enumerate(sessions):
            obs_count = self.session_repo.db.execute_query(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?;", (s.session_id,)
            )[0][0]

            self.table.setItem(row, 0, QTableWidgetItem(s.session_id))
            self.table.setItem(row, 1, QTableWidgetItem(s.participant_code or "ANONYMOUS"))
            self.table.setItem(row, 2, QTableWidgetItem(s.provenance))
            self.table.setItem(row, 3, QTableWidgetItem(s.collection_mode))
            self.table.setItem(row, 4, QTableWidgetItem(f"{s.duration_seconds or 0.0:.1f}s"))
            self.table.setItem(row, 5, QTableWidgetItem(str(obs_count)))
            self.table.setItem(row, 6, QTableWidgetItem(s.quality_status))

        if sessions:
            self.table.selectRow(0)
        else:
            self.txt_details.setText("No sessions stored in database yet.")

    @Slot()
    def _on_session_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            self.selected_session_id = None
            return

        row = selected_rows[0].row()
        sid_item = self.table.item(row, 0)
        if not sid_item:
            return

        sid = sid_item.text()
        self.selected_session_id = sid

        # Load metrics and render detail view
        metrics = compute_session_metrics(self.db, sid)
        sess = self.session_repo.get(sid)

        if not sess or not metrics:
            self.txt_details.setText("Unable to load session details.")
            return

        lines = [
            f"=== SESSION DETAILS: {sid} ===",
            f"Participant Code:     {sess.participant_code or 'ANONYMOUS'}",
            f"Provenance:           {sess.provenance}",
            f"Collection Mode:      {sess.collection_mode}",
            f"Consent Confirmed:    {sess.consent_confirmed}",
            f"Start Time (UTC):     {sess.start_time}",
            f"End Time (UTC):       {sess.end_time or 'In progress'}",
            f"Duration:             {metrics.duration_seconds:.1f} seconds",
            f"Total Observations:   {metrics.total_observations}",
            f"Unique Tracks:        {metrics.track_count}",
            f"Effective Rate:       {metrics.timing.effective_fps:.1f} FPS (mean {metrics.timing.mean_interval_ms:.1f} ms, std {metrics.timing.std_interval_ms:.1f} ms)",
            f"Mean Face Quality:    {metrics.mean_face_quality:.2f}",
            "",
            "--- OBSERVABILITY DISTRIBUTION ---",
        ]
        for obs_lvl, pct in metrics.observability_distribution.items():
            lines.append(f"  {obs_lvl:20s}: {pct:5.1f}%")

        lines.extend([
            "",
            "--- SMOOTHED BEHAVIORAL STATES ---",
        ])
        for beh, pct in metrics.smoothed_behavior_distribution.items():
            lines.append(f"  {beh:20s}: {pct:5.1f}%")

        lines.extend([
            "",
            "--- RAW FRAME BEHAVIORAL STATES ---",
        ])
        for beh, pct in metrics.raw_behavior_distribution.items():
            lines.append(f"  {beh:20s}: {pct:5.1f}%")

        self.txt_details.setText("\n".join(lines))

    @Slot()
    def _run_audit(self):
        if not self.selected_session_id:
            QMessageBox.warning(self, "Select Session", "Please select a session to audit.")
            return

        report = self.auditor.audit_session(self.selected_session_id, persist_result=True)
        self.refresh_sessions()

        QMessageBox.information(
            self,
            "Quality Audit Completed",
            f"Audit Verdict: {report.verdict.value}\n\n"
            f"Effective FPS: {report.effective_fps:.1f}\n"
            f"Mean Interval: {report.mean_interval_ms:.1f} ms (std {report.std_interval_ms:.1f} ms)\n"
            f"Annotation Coverage: {report.annotation_coverage_fraction * 100.0:.1f}%\n"
            f"Warnings ({len(report.warnings)}):\n  - " + "\n  - ".join(report.warnings or ["None"])
        )

    @Slot()
    def _delete_session(self):
        if not self.selected_session_id:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete session {self.selected_session_id} and all its observations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.session_repo.delete(self.selected_session_id)
            self.selected_session_id = None
            self.refresh_sessions()

    @Slot()
    def _audit_external_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 2-5 Minute Classroom Video to Audit",
            "data/recordings",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if not file_path:
            return

        p = Path(file_path)
        if not p.is_file():
            QMessageBox.warning(self, "File Not Found", f"Video file not found:\n{file_path}")
            return

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

        dlg = VideoAuditDialog(self, result)
        dlg.exec_()

    @Slot(str)
    def _on_audit_failed(self, error_msg: str):
        if hasattr(self, "audit_progress_dialog") and self.audit_progress_dialog is not None:
            self.audit_progress_dialog.close()
        QMessageBox.critical(self, "Audit Error", f"Video attention audit encountered an error:\n{error_msg}")

    @Slot()
    def _cancel_audit(self):
        if hasattr(self, "_audit_worker") and self._audit_worker is not None and self._audit_worker.isRunning():
            self._audit_worker.terminate()
            self._audit_worker.wait(1000)
            logger.info("Video attention audit was cancelled by the user.")

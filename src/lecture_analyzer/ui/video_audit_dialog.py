"""Interactive GUI dialog for inspecting classroom attention audit results."""

import logging
import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.analytics.video_audit_service import (
    VideoAttentionAuditService,
    VideoAuditResult,
)
from lecture_analyzer.config import AppConfig

logger = logging.getLogger("lecture_analyzer.ui.video_audit_dialog")


class VideoAuditWorker(QThread):
    """Background worker for non-blocking execution of video attention audit."""
    progress = Signal(int, int, float, str)  # current, total, fps, message
    audit_completed = Signal(object)        # VideoAuditResult
    audit_failed = Signal(str)              # error message

    def __init__(
        self,
        video_path: str,
        config: AppConfig,
        generate_annotated_video: bool = True,
        frame_stride: int = 1,
    ):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.generate_annotated_video = generate_annotated_video
        self.frame_stride = frame_stride

    def run(self):
        try:
            service = VideoAttentionAuditService(self.config)

            def on_progress(cur, total, fps, msg):
                self.progress.emit(cur, total, fps, msg)

            result = service.audit_video(
                video_path=self.video_path,
                generate_annotated_video=self.generate_annotated_video,
                frame_stride=self.frame_stride,
                progress_callback=on_progress,
            )
            self.audit_completed.emit(result)
        except Exception as e:
            logger.error(f"Video audit background task failed: {e}", exc_info=True)
            self.audit_failed.emit(str(e))


class VideoAuditDialog(QDialog):
    """Comprehensive modal dialog for reviewing video attention audits and student breakdowns."""

    def __init__(self, parent: Optional[QWidget], result: VideoAuditResult):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle(f"Classroom Attention Audit — {result.video_filename}")
        self.resize(1100, 750)
        self.setMinimumSize(850, 550)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Header Title and Summary Cards
        header_box = QGroupBox("Audit Summary: " + self.result.video_filename)
        header_layout = QVBoxLayout(header_box)
        header_layout.setSpacing(8)

        # Metric Badges Row
        badge_row = QHBoxLayout()
        badge_row.setSpacing(10)

        dur_m, dur_s = divmod(int(self.result.video_duration_seconds), 60)
        lbl_dur = QLabel(f"Duration: {dur_m:02d}:{dur_s:02d} ({self.result.video_duration_seconds:.1f}s)")
        lbl_dur.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

        lbl_students = QLabel(f"Total Students: {self.result.total_unique_students}")
        lbl_students.setStyleSheet("background-color: #34495e; color: #ecf0f1; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

        conc_pct = self.result.overall_concentration_percentage
        lbl_conc = QLabel(f"Concentrated: {conc_pct:.1f}%")
        lbl_conc.setStyleSheet("background-color: #1e4620; color: #2ecc71; font-weight: bold; padding: 6px 12px; border-radius: 4px; font-size: 13px;")

        dist_pct = self.result.overall_distraction_percentage
        lbl_dist = QLabel(f"Distracted: {dist_pct:.1f}%")
        lbl_dist.setStyleSheet("background-color: #4a1c1a; color: #e74c3c; font-weight: bold; padding: 6px 12px; border-radius: 4px; font-size: 13px;")

        unc_pct = self.result.overall_uncertain_percentage
        lbl_unc = QLabel(f"Uncertain: {unc_pct:.1f}%")
        lbl_unc.setStyleSheet("background-color: #2b2b2b; color: #95a5a6; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

        badge_row.addWidget(lbl_dur)
        badge_row.addWidget(lbl_students)
        badge_row.addWidget(lbl_conc)
        badge_row.addWidget(lbl_dist)
        badge_row.addWidget(lbl_unc)
        badge_row.addStretch()
        header_layout.addLayout(badge_row)

        layout.addWidget(header_box)

        # Tab Widget for detailed tabs
        tabs = QTabWidget()

        # Tab 1: Formatted Markdown Report
        tab_report = QWidget()
        tab_report_layout = QVBoxLayout(tab_report)
        tab_report_layout.setContentsMargins(6, 6, 6, 6)
        self.txt_markdown = QTextEdit()
        self.txt_markdown.setReadOnly(True)
        self.txt_markdown.setFontFamily("Consolas")
        self.txt_markdown.setStyleSheet("background-color: #1a1a1a; color: #ecf0f1; padding: 10px; font-size: 12px;")
        self.txt_markdown.setMarkdown(self.result.summary_markdown)
        tab_report_layout.addWidget(self.txt_markdown)
        tabs.addTab(tab_report, "Formal Audit Report")

        # Tab 2: Individual Student Roster
        tab_students = QWidget()
        tab_students_layout = QVBoxLayout(tab_students)
        tab_students_layout.setContentsMargins(6, 6, 6, 6)

        table_students = QTableWidget()
        table_students.setColumnCount(8)
        table_students.setHorizontalHeaderLabels([
            "Student ID", "Visible Time", "Concentrating %", "Distracted %", "Uncertain %",
            "Dominant Action", "Distraction Events", "Rating"
        ])
        table_students.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_students.setRowCount(len(self.result.students))

        for row, s in enumerate(self.result.students):
            vm, vs = divmod(int(s.total_visible_seconds), 60)
            vis_text = f"{vm:02d}:{vs:02d} ({s.visibility_percentage:.0f}%)"

            item_id = QTableWidgetItem(s.track_id)
            item_id.setTextAlignment(Qt.AlignCenter)
            item_vis = QTableWidgetItem(vis_text)
            item_vis.setTextAlignment(Qt.AlignCenter)

            item_conc = QTableWidgetItem(f"{s.concentrating_percentage:.1f}% ({s.concentrating_seconds:.1f}s)")
            item_conc.setTextAlignment(Qt.AlignCenter)
            item_conc.setForeground(Qt.green if s.concentrating_percentage >= 50.0 else Qt.yellow)

            item_dist = QTableWidgetItem(f"{s.distracted_percentage:.1f}% ({s.distracted_seconds:.1f}s)")
            item_dist.setTextAlignment(Qt.AlignCenter)
            item_dist.setForeground(Qt.red if s.distracted_percentage >= 30.0 else Qt.lightGray)

            item_unc = QTableWidgetItem(f"{s.uncertain_percentage:.1f}%")
            item_unc.setTextAlignment(Qt.AlignCenter)

            item_dom = QTableWidgetItem(s.dominant_behavior)
            item_dom.setTextAlignment(Qt.AlignCenter)

            item_events = QTableWidgetItem(str(s.distraction_event_count))
            item_events.setTextAlignment(Qt.AlignCenter)

            item_rating = QTableWidgetItem(s.attention_rating)
            item_rating.setTextAlignment(Qt.AlignCenter)

            table_students.setItem(row, 0, item_id)
            table_students.setItem(row, 1, item_vis)
            table_students.setItem(row, 2, item_conc)
            table_students.setItem(row, 3, item_dist)
            table_students.setItem(row, 4, item_unc)
            table_students.setItem(row, 5, item_dom)
            table_students.setItem(row, 6, item_events)
            table_students.setItem(row, 7, item_rating)

        tab_students_layout.addWidget(table_students)
        tabs.addTab(tab_students, "Student Roster Breakdown")

        # Tab 3: Timeline Shifts
        tab_timeline = QWidget()
        tab_timeline_layout = QVBoxLayout(tab_timeline)
        tab_timeline_layout.setContentsMargins(6, 6, 6, 6)

        table_timeline = QTableWidget()
        table_timeline.setColumnCount(7)
        table_timeline.setHorizontalHeaderLabels([
            "Time Window", "Students Present", "Concentrating", "Distracted",
            "Uncertain", "Attention %", "Status"
        ])
        table_timeline.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_timeline.setRowCount(len(self.result.timeline))

        for row, t in enumerate(self.result.timeline):
            item_w = QTableWidgetItem(t.time_label)
            item_w.setTextAlignment(Qt.AlignCenter)
            item_p = QTableWidgetItem(str(t.students_present))
            item_p.setTextAlignment(Qt.AlignCenter)
            item_c = QTableWidgetItem(str(t.students_concentrating))
            item_c.setTextAlignment(Qt.AlignCenter)
            item_d = QTableWidgetItem(str(t.students_distracted))
            item_d.setTextAlignment(Qt.AlignCenter)
            item_u = QTableWidgetItem(str(t.students_uncertain))
            item_u.setTextAlignment(Qt.AlignCenter)

            item_pct = QTableWidgetItem(f"{t.attention_percentage:.1f}%")
            item_pct.setTextAlignment(Qt.AlignCenter)
            item_pct.setForeground(Qt.green if t.attention_percentage >= 60.0 else Qt.red)

            item_st = QTableWidgetItem(t.attention_status)
            item_st.setTextAlignment(Qt.AlignCenter)

            table_timeline.setItem(row, 0, item_w)
            table_timeline.setItem(row, 1, item_p)
            table_timeline.setItem(row, 2, item_c)
            table_timeline.setItem(row, 3, item_d)
            table_timeline.setItem(row, 4, item_u)
            table_timeline.setItem(row, 5, item_pct)
            table_timeline.setItem(row, 6, item_st)

        tab_timeline_layout.addWidget(table_timeline)
        tabs.addTab(tab_timeline, "Timeline Attention Shifts")

        layout.addWidget(tabs)

        # Action Buttons Row
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.btn_copy_markdown = QPushButton("Copy Report Text")
        self.btn_copy_markdown.clicked.connect(self._copy_markdown)

        self.btn_open_report_file = QPushButton("Open Markdown File")
        self.btn_open_report_file.clicked.connect(self._open_markdown_file)

        self.btn_open_video = QPushButton("Open Annotated Video")
        self.btn_open_video.clicked.connect(self._open_annotated_video)
        if not self.result.annotated_video_path or not os.path.exists(self.result.annotated_video_path):
            self.btn_open_video.setEnabled(False)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        action_row.addWidget(self.btn_copy_markdown)
        action_row.addWidget(self.btn_open_report_file)
        action_row.addWidget(self.btn_open_video)
        action_row.addStretch()
        action_row.addWidget(self.btn_close)

        layout.addLayout(action_row)

    def _copy_markdown(self):
        from PySide6.QtGui import QGuiApplication
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.result.summary_markdown)
        QMessageBox.information(self, "Copied", "Full Audit Report copied to clipboard.")

    def _open_markdown_file(self):
        if self.result.report_markdown_path and os.path.exists(self.result.report_markdown_path):
            _open_file_in_os(self.result.report_markdown_path)
        else:
            QMessageBox.warning(self, "Not Found", "Markdown report file does not exist on disk.")

    def _open_annotated_video(self):
        if self.result.annotated_video_path and os.path.exists(self.result.annotated_video_path):
            _open_file_in_os(self.result.annotated_video_path)
        else:
            QMessageBox.warning(self, "Not Found", "Annotated video file does not exist on disk.")


def _open_file_in_os(path: str):
    """Safely opens a file or directory with system default application across OS."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        logger.error(f"Failed to launch OS file opener for {path}: {e}")

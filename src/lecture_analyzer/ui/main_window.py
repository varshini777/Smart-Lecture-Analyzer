"""Main desktop application window for Lecture Engagement Pattern Analyzer."""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.collection.session_service import SessionService
from lecture_analyzer.config import AppConfig
from lecture_analyzer.storage.database import Database
from lecture_analyzer.ui.annotation_view import AnnotationTab
from lecture_analyzer.ui.dataset_view import DatasetViewTab
from lecture_analyzer.ui.live_monitor import LiveMonitorTab
from lecture_analyzer.ui.report_view import ModelEvaluationTab
from lecture_analyzer.ui.session_browser import SessionBrowserTab

logger = logging.getLogger("lecture_analyzer.ui.main_window")

DARK_STYLE = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #333333;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #3498db;
}
QTabWidget::pane {
    border: 1px solid #2d2d2d;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background: #181818;
    color: #95a5a6;
    padding: 10px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #2c3e50;
    color: #ffffff;
    border-bottom: 2px solid #3498db;
}
QPushButton {
    background-color: #2c3e50;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #34495e;
}
QPushButton:pressed {
    background-color: #1a252f;
}
QPushButton:disabled {
    background-color: #2b2b2b;
    color: #666666;
}
QLineEdit, QComboBox, QDoubleSpinBox {
    background-color: #252525;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
}
QTableWidget {
    background-color: #181818;
    gridline-color: #2a2a2a;
    border: 1px solid #2d2d2d;
}
QHeaderView::section {
    background-color: #252525;
    color: #ecf0f1;
    padding: 4px;
    border: 1px solid #333333;
    font-weight: bold;
}
"""


class MainWindow(QMainWindow):
    """Primary application window coordinating all functional subsystem tabs."""

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig()
        self.db = Database(self.config.storage.database_path)
        self.session_service = SessionService(self.config, self.db)

        self.setWindowTitle(f"{self.config.system.app_name} v{self.config.system.version}")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 680)

        self._init_ui()
        self.setStyleSheet(DARK_STYLE)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tab Widget
        self.tabs = QTabWidget()

        # 1. Live Monitor
        self.tab_live = LiveMonitorTab(self.config, self.session_service)
        self.tabs.addTab(self.tab_live, "Live Monitor")

        # 2. Sessions and Reports
        self.tab_sessions = SessionBrowserTab(self.db, self.config)
        self.tabs.addTab(self.tab_sessions, "Sessions and Reports")

        # 3. Annotation
        self.tab_annotations = AnnotationTab(self.db)
        self.tabs.addTab(self.tab_annotations, "Human Annotation")

        # 4. Dataset and Quality Audit
        self.tab_dataset = DatasetViewTab(self.db)
        self.tabs.addTab(self.tab_dataset, "Dataset & Quality Audit")

        # 5. Model and Evaluation
        self.tab_model = ModelEvaluationTab(self.db)
        self.tabs.addTab(self.tab_model, "ML Readiness & Evaluation")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        # Status Bar
        self.statusBar().showMessage("Ready. Operates in primary Explainable Rule Engine mode.")

    def _on_tab_changed(self, index: int):
        # Refresh target tab data dynamically on switch
        current_widget = self.tabs.widget(index)
        if isinstance(current_widget, SessionBrowserTab):
            current_widget.refresh_sessions()
        elif isinstance(current_widget, AnnotationTab):
            current_widget.refresh_sessions()
        elif isinstance(current_widget, DatasetViewTab):
            current_widget.refresh_summary()
        elif isinstance(current_widget, ModelEvaluationTab):
            current_widget.refresh_readiness()

    def closeEvent(self, event):
        """Ensures active camera workers and recording sessions stop cleanly."""
        logger.info("Closing application: cleaning up resources...")
        if self.session_service.is_session_active:
            self.session_service.stop_session()

        if hasattr(self.tab_live, "stop_camera"):
            self.tab_live.stop_camera()
        elif hasattr(self.tab_live, "worker") and self.tab_live.worker is not None:
            self.tab_live.worker.stop()

        if hasattr(self.tab_live, "_prewarm_worker") and self.tab_live._prewarm_worker is not None:
            self.tab_live._prewarm_worker.wait(1000)

        event.accept()


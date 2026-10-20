"""Machine learning readiness gatekeeper and baseline model evaluation tab."""

import json
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.ml.readiness import MLReadinessChecker
from lecture_analyzer.ml.train import MLTrainingOrchestrator, ReadinessGateViolationError
from lecture_analyzer.storage.database import Database
from lecture_analyzer.utils.paths import resolve_path

logger = logging.getLogger("lecture_analyzer.ui.report_view")


class ModelEvaluationTab(QWidget):
    """Tab presenting ML readiness gating status, baseline training controls, and honest evaluation reports."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.checker = MLReadinessChecker()
        self.trainer = MLTrainingOrchestrator(db)

        self._init_ui()
        self.refresh_readiness()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Readiness Gating & Training Controls
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Gate Status Box
        gate_box = QGroupBox("Scientific ML Readiness Gate")
        gate_layout = QVBoxLayout(gate_box)

        self.lbl_gate_status = QLabel("Evaluating Readiness...")
        self.lbl_gate_status.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px; border-radius: 4px;")
        gate_layout.addWidget(self.lbl_gate_status)

        self.txt_gate_reasons = QTextEdit()
        self.txt_gate_reasons.setReadOnly(True)
        self.txt_gate_reasons.setStyleSheet("background-color: #1a1a1a; font-size: 11px;")
        gate_layout.addWidget(self.txt_gate_reasons)

        self.btn_recheck = QPushButton("Re-evaluate Readiness")
        self.btn_recheck.clicked.connect(self.refresh_readiness)
        gate_layout.addWidget(self.btn_recheck)

        left_layout.addWidget(gate_box)

        # Training Controls Box
        train_box = QGroupBox("Baseline ML Training (Participant-Held-Out)")
        train_layout = QVBoxLayout(train_box)

        self.cmb_model_type = QComboBox()
        self.cmb_model_type.addItems([
            "Random Forest",
            "Logistic Regression",
            "Decision Tree",
            "Majority-Class Baseline",
        ])
        train_layout.addWidget(QLabel("Select Baseline Model:"))
        train_layout.addWidget(self.cmb_model_type)

        self.btn_train = QPushButton("Train Baseline Model")
        self.btn_train.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.btn_train.clicked.connect(self._train_model)
        train_layout.addWidget(self.btn_train)

        left_layout.addWidget(train_box)
        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right: Evaluation Report & Confusion Matrix Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_layout.addWidget(QLabel("Evaluation Report & Scientific Diagnostics:"))
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setFontFamily("Consolas")
        self.txt_report.setStyleSheet("background-color: #1a1a1a; color: #ecf0f1; padding: 8px;")
        right_layout.addWidget(self.txt_report)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)

    def refresh_readiness(self):
        """Assesses readiness of current SQLite database records for ML."""
        report = self.checker.check_database_readiness(self.db)

        if report.is_ready_for_training:
            self.lbl_gate_status.setText("GATE PASSED: READY FOR ML TRAINING")
            self.lbl_gate_status.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            self.btn_train.setEnabled(True)
            self.btn_train.setToolTip("Dataset meets all gating requirements.")
        else:
            self.lbl_gate_status.setText("GATE LOCKED: DATASET NOT READY")
            self.lbl_gate_status.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
            self.btn_train.setEnabled(False)
            self.btn_train.setToolTip("Training is blocked until genuine human dataset satisfies all criteria.")

        lines = [report.verdict_message, ""]
        if report.failed_gates:
            lines.append("Failed Gates:")
            for g in report.failed_gates:
                lines.append(f"  [X] {g}")
            lines.append("")

        if report.passed_gates:
            lines.append("Passed Gates:")
            for g in report.passed_gates:
                lines.append(f"  [V] {g}")

        self.txt_gate_reasons.setText("\n".join(lines))

        # Check for existing evaluated models
        models_dir = resolve_path("models")
        meta_files = list(models_dir.glob("active_*_metadata.json"))
        if meta_files:
            latest_meta = sorted(meta_files, key=lambda p: p.stat().st_mtime)[-1]
            with open(latest_meta, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.txt_report.setText(f"=== ACTIVE MODEL METADATA: {latest_meta.name} ===\n\n" + json.dumps(data, indent=2))
        else:
            self.txt_report.setText(
                "No trained machine-learning model currently active.\n"
                "System is operating in primary EXPLAINABLE RULE-BASED MODE.\n\n"
                "To unlock ML training:\n"
                "1. Collect real-human sessions with participant consent.\n"
                "2. Provide independent ground-truth human annotations.\n"
                "3. Ensure at least 10 participants and acceptable class representation.\n"
                "4. Run Quality Audits to verify data regularity."
            )

    @Slot()
    def _train_model(self):
        choice = self.cmb_model_type.currentText()
        type_mapping = {
            "Random Forest": "random_forest",
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Majority-Class Baseline": "majority",
        }
        mod_type = type_mapping.get(choice, "random_forest")

        try:
            pipeline, eval_report = self.trainer.train_and_evaluate(model_type=mod_type)
            self.refresh_readiness()
            QMessageBox.information(
                self,
                "Model Trained Successfully",
                f"Trained {choice} on genuine human dataset.\n"
                f"Macro F1: {eval_report.test_metrics.macro_f1:.3f}\n"
                f"Balanced Accuracy: {eval_report.test_metrics.balanced_accuracy:.3f}"
            )
        except ReadinessGateViolationError as e:
            QMessageBox.critical(self, "ML Training Blocked", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Training Exception", str(e))

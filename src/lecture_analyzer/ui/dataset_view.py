"""Dataset governance, provenance breakdown, and export tab."""

import json
import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lecture_analyzer.dataset.export_dataset import DatasetExporter
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import SessionRepository

logger = logging.getLogger("lecture_analyzer.ui.dataset_view")


class DatasetViewTab(QWidget):
    """Tab presenting data provenance governance, dataset audits, and manifest-backed exports."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.session_repo = SessionRepository(db)
        self.exporter = DatasetExporter(db)

        self._init_ui()
        self.refresh_summary()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Controls & Summary Cards
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Summary Card Box
        summary_box = QGroupBox("Database Inventory")
        form = QFormLayout(summary_box)

        self.lbl_real_human = QLabel("0 sessions (0 participants)")
        self.lbl_real_human.setStyleSheet("color: #2ecc71; font-weight: bold;")
        self.lbl_simulated = QLabel("0 sessions")
        self.lbl_pipeline_val = QLabel("0 sessions")
        self.lbl_unknown = QLabel("0 sessions")
        self.lbl_total_obs = QLabel("0 observations")

        form.addRow("REAL_HUMAN (Research):", self.lbl_real_human)
        form.addRow("SIMULATED (Demo/Test):", self.lbl_simulated)
        form.addRow("PIPELINE_VALIDATION:", self.lbl_pipeline_val)
        form.addRow("UNKNOWN Provenance:", self.lbl_unknown)
        form.addRow("Total Frame Records:", self.lbl_total_obs)

        left_layout.addWidget(summary_box)

        # Export Controls Box
        export_box = QGroupBox("Provenance-Filtered Export")
        exp_layout = QVBoxLayout(export_box)

        self.cmb_prov = QComboBox()
        self.cmb_prov.addItems([
            DataProvenance.REAL_HUMAN.value,
            DataProvenance.PIPELINE_VALIDATION.value,
            DataProvenance.SIMULATED.value,
            DataProvenance.UNKNOWN.value,
        ])
        exp_layout.addWidget(QLabel("Select Target Provenance:"))
        exp_layout.addWidget(self.cmb_prov)

        self.btn_export = QPushButton("Export Dataset & Generate Manifest")
        self.btn_export.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self._run_export)
        exp_layout.addWidget(self.btn_export)

        left_layout.addWidget(export_box)
        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right: Manifest & Export Log Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Export Manifest & Audit Trail:"))
        self.txt_manifest = QTextEdit()
        self.txt_manifest.setReadOnly(True)
        self.txt_manifest.setFontFamily("Consolas")
        self.txt_manifest.setStyleSheet("background-color: #1a1a1a; color: #ecf0f1; padding: 8px;")
        right_layout.addWidget(self.txt_manifest)

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 620])
        main_layout.addWidget(splitter)

    def refresh_summary(self):
        """Queries SQLite and updates the inventory statistics."""
        def count_prov(prov: str) -> int:
            return self.db.execute_query("SELECT COUNT(*) FROM sessions WHERE provenance = ?;", (prov,))[0][0]

        real_cnt = count_prov("REAL_HUMAN")
        real_parts = self.db.execute_query("SELECT COUNT(DISTINCT participant_code) FROM sessions WHERE provenance = 'REAL_HUMAN';")[0][0]
        sim_cnt = count_prov("SIMULATED")
        val_cnt = count_prov("PIPELINE_VALIDATION")
        unk_cnt = count_prov("UNKNOWN")
        obs_total = self.db.execute_query("SELECT COUNT(*) FROM observations;")[0][0]

        self.lbl_real_human.setText(f"{real_cnt} session(s) ({real_parts} participant(s))")
        self.lbl_simulated.setText(f"{sim_cnt} session(s)")
        self.lbl_pipeline_val.setText(f"{val_cnt} session(s)")
        self.lbl_unknown.setText(f"{unk_cnt} session(s)")
        self.lbl_total_obs.setText(f"{obs_total:,} rows")

    @Slot()
    def _run_export(self):
        prov_str = self.cmb_prov.currentText()
        prov_enum = DataProvenance(prov_str)

        try:
            csv_path, manifest = self.exporter.export_dataset(
                provenance=prov_enum,
                require_annotations=(prov_enum == DataProvenance.REAL_HUMAN),
                require_passed_audit=(prov_enum == DataProvenance.REAL_HUMAN),
            )

            if manifest is None:
                self.txt_manifest.setText("Export aborted: No rows matched the requested criteria.")
                return

            # Display formatted manifest JSON
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            self.txt_manifest.setText(manifest_json)
            self.refresh_summary()

            if csv_path:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Dataset exported successfully:\n{csv_path.name}\n"
                    f"Included Sessions: {manifest.unique_session_count}\n"
                    f"Total Rows: {manifest.total_observation_rows:,}\n"
                    f"SHA256: {manifest.sha256_hash[:16]}..."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Export Empty",
                    f"No sessions were eligible for export under {prov_str}.\n"
                    f"Excluded sessions ({len(manifest.excluded_sessions)}):\n  - " +
                    "\n  - ".join([f"{k}: {v}" for k, v in manifest.excluded_sessions.items()] or ["None"])
                )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

"""Provenance-governed dataset export with manifest generation and integrity checks."""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from lecture_analyzer.annotation.annotation_service import AnnotationService
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.dataset.quality_audit import AuditVerdict, QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import SessionRepository
from lecture_analyzer.utils.paths import resolve_path, safe_relative_path
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.dataset.export_dataset")

# Standard predictive feature columns (Excludes target, participant ID, session ID, and timestamps)
PREDICTIVE_FEATURE_COLUMNS = [
    "ear_avg",
    "ear_left",
    "ear_right",
    "mar",
    "face_quality",
    "face_detected",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_pose_valid",
    "pose_detected",
    "shoulder_tilt_deg",
    "torso_upright_ratio",
    "head_shoulder_offset",
    "wrists_near_head",
    "wrists_near_table",
    "phone_detected",
    "phone_confidence",
    "observability_level",
]


@dataclass
class ExportManifest:
    export_id: str
    export_timestamp: str
    export_type: str
    allowed_provenance: str
    included_session_ids: List[str]
    excluded_sessions: Dict[str, str]  # session_id -> reason
    unique_participant_count: int
    unique_session_count: int
    total_observation_rows: int
    feature_columns: List[str]
    target_column: str
    label_distribution: Dict[str, int]
    missing_value_stats: Dict[str, float]
    sha256_hash: str
    csv_filename: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetExporter:
    """Exports structured datasets strictly honoring provenance and audit boundaries."""

    def __init__(self, db: Database, exports_dir: str = "data/exports"):
        self.db = db
        self.exports_dir = resolve_path(exports_dir)
        self.session_repo = SessionRepository(db)
        self.annot_service = AnnotationService(db)
        self.auditor = QualityAuditor(db)

    def export_dataset(
        self,
        provenance: DataProvenance = DataProvenance.REAL_HUMAN,
        require_annotations: bool = True,
        require_passed_audit: bool = True,
        export_prefix: str = "dataset",
    ) -> Tuple[Optional[Path], Optional[ExportManifest]]:
        """Exports a filtered, validated dataset and generates a forensic manifest."""
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        export_id = f"{export_prefix}_{provenance.value.lower()}_{int(pd.Timestamp.now().timestamp())}"

        # 1. Fetch sessions matching provenance
        all_sessions = self.session_repo.list_all(provenance=provenance.value)
        included_sessions = []
        excluded_sessions: Dict[str, str] = {}

        for sess in all_sessions:
            sid = sess.session_id

            # Audit check
            if require_passed_audit:
                audit = self.auditor.audit_session(sid, persist_result=False)
                if audit.verdict == AuditVerdict.FAIL:
                    excluded_sessions[sid] = f"Failed quality audit: {'; '.join(audit.reasons)}"
                    continue

            # Annotation check
            annots = self.annot_service.list_annotations(sid)
            if require_annotations and not annots:
                excluded_sessions[sid] = "No independent human annotations recorded."
                continue

            included_sessions.append((sess, annots))

        if not included_sessions:
            logger.warning(f"No sessions eligible for export under provenance {provenance.value}.")
            manifest = ExportManifest(
                export_id=export_id,
                export_timestamp=now_utc_iso(),
                export_type=f"{provenance.value}_EXPORT",
                allowed_provenance=provenance.value,
                included_session_ids=[],
                excluded_sessions=excluded_sessions,
                unique_participant_count=0,
                unique_session_count=0,
                total_observation_rows=0,
                feature_columns=PREDICTIVE_FEATURE_COLUMNS,
                target_column="human_behavior_label",
                label_distribution={},
                missing_value_stats={},
                sha256_hash="",
                csv_filename="",
            )
            return None, manifest

        # 2. Extract observations and align with human annotations
        rows = []
        participants = set()

        for sess, annots in included_sessions:
            sid = sess.session_id
            if sess.participant_code:
                participants.add(sess.participant_code)

            obs_rows = self.db.execute_query(
                "SELECT * FROM observations WHERE session_id = ? ORDER BY elapsed_seconds ASC;",
                (sid,)
            )

            for obs in obs_rows:
                el_s = float(obs["elapsed_seconds"])
                tid = obs["track_id"]

                # Match human annotation
                matched_label = None
                matched_annotator = None
                for a in annots:
                    if a.start_seconds <= el_s <= a.end_seconds:
                        if a.track_id is None or a.track_id == tid:
                            matched_label = a.behavior_label
                            matched_annotator = a.annotator_id
                            break

                if require_annotations and matched_label is None:
                    # Skip unannotated frames when ground truth is required
                    continue

                row_dict = {
                    # Grouping / metadata identifiers (NOT predictive features)
                    "session_id": sid,
                    "participant_id": sess.participant_code or "ANONYMOUS",
                    "track_id": tid,
                    "elapsed_seconds": el_s,
                    "wall_timestamp": obs["wall_timestamp"],
                    "provenance": sess.provenance,
                    # Target ground-truth label
                    "human_behavior_label": matched_label,
                    "annotator_id": matched_annotator,
                    # Automated rule output (for comparison / auditing)
                    "system_rule_behavior": obs["smoothed_behavior"],
                    "system_rule_score": obs["smoothed_score"],
                }

                # Predictive features
                for col in PREDICTIVE_FEATURE_COLUMNS:
                    row_dict[col] = obs[col]

                rows.append(row_dict)

        if not rows:
            logger.warning("No rows extracted matching export criteria.")
            manifest = ExportManifest(
                export_id=export_id,
                export_timestamp=now_utc_iso(),
                export_type=f"{provenance.value}_EXPORT",
                allowed_provenance=provenance.value,
                included_session_ids=[s[0].session_id for s in included_sessions],
                excluded_sessions=excluded_sessions,
                unique_participant_count=len(participants),
                unique_session_count=len(included_sessions),
                total_observation_rows=0,
                feature_columns=PREDICTIVE_FEATURE_COLUMNS,
                target_column="human_behavior_label",
                label_distribution={},
                missing_value_stats={},
                sha256_hash="",
                csv_filename="",
            )
            return None, manifest

        df = pd.DataFrame(rows)
        csv_filename = f"{export_id}.csv"
        csv_path = self.exports_dir / csv_filename
        df.to_csv(csv_path, index=False)

        # 3. Compute Manifest Statistics
        label_dist = df["human_behavior_label"].value_counts().to_dict() if "human_behavior_label" in df else {}
        missing_stats = {
            col: round(float(df[col].isna().mean()), 3)
            for col in PREDICTIVE_FEATURE_COLUMNS if col in df
        }

        # Compute SHA256 of the CSV
        sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()

        manifest = ExportManifest(
            export_id=export_id,
            export_timestamp=now_utc_iso(),
            export_type=f"{provenance.value}_EXPORT",
            allowed_provenance=provenance.value,
            included_session_ids=[s[0].session_id for s in included_sessions],
            excluded_sessions=excluded_sessions,
            unique_participant_count=len(participants),
            unique_session_count=len(included_sessions),
            total_observation_rows=len(df),
            feature_columns=PREDICTIVE_FEATURE_COLUMNS,
            target_column="human_behavior_label",
            label_distribution=label_dist,
            missing_value_stats=missing_stats,
            sha256_hash=sha256,
            csv_filename=csv_filename,
        )

        manifest_path = self.exports_dir / f"{export_id}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Record in dataset_exports table
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO dataset_exports (
                    export_id, created_at, export_type, manifest_path, csv_path,
                    session_count, participant_count, observation_count, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                export_id,
                manifest.export_timestamp,
                manifest.export_type,
                safe_relative_path(manifest_path),
                safe_relative_path(csv_path),
                manifest.unique_session_count,
                manifest.unique_participant_count,
                manifest.total_observation_rows,
                f"SHA256: {sha256[:16]}...",
            ))

        logger.info(f"Dataset exported: {csv_filename} ({len(df)} rows, {len(included_sessions)} sessions, {len(participants)} participants)")
        return csv_path, manifest

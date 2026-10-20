"""Comprehensive session data quality audit and validation engine."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from lecture_analyzer.analytics.session_metrics import compute_session_metrics
from lecture_analyzer.annotation.annotation_validation import compute_annotation_coverage
from lecture_analyzer.collection.recording_service import load_timestamp_sidecar
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import AnnotationRepository, SessionRepository
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.dataset.quality_audit")


class AuditVerdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


@dataclass
class QualityAuditReport:
    session_id: str
    verdict: AuditVerdict
    audit_timestamp: str
    observation_count: int
    duration_seconds: float
    effective_fps: float
    mean_interval_ms: float
    std_interval_ms: float
    face_quality_mean: float
    observability_full_fraction: float
    annotation_coverage_fraction: float
    has_overlapping_annotations: bool
    descriptive_agreement_percentage: Optional[float]
    warnings: List[str]
    reasons: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


class QualityAuditor:
    """Executes systematic multi-point audits of recorded sessions."""

    def __init__(self, db: Database):
        self.db = db
        self.session_repo = SessionRepository(db)
        self.annot_repo = AnnotationRepository(db)

    def audit_session(self, session_id: str, persist_result: bool = True) -> QualityAuditReport:
        """Audits a session for temporal regularity, sensor integrity, and annotation coverage."""
        session = self.session_repo.get(session_id)
        if not session:
            return QualityAuditReport(
                session_id=session_id,
                verdict=AuditVerdict.FAIL,
                audit_timestamp=now_utc_iso(),
                observation_count=0,
                duration_seconds=0.0,
                effective_fps=0.0,
                mean_interval_ms=0.0,
                std_interval_ms=0.0,
                face_quality_mean=0.0,
                observability_full_fraction=0.0,
                annotation_coverage_fraction=0.0,
                has_overlapping_annotations=False,
                descriptive_agreement_percentage=None,
                warnings=["Session ID not found in database."],
                reasons=["Session does not exist."],
                details={},
            )

        warnings: List[str] = []
        fail_reasons: List[str] = []

        metrics = compute_session_metrics(self.db, session_id)
        annots = self.annot_repo.list_by_session(session_id)
        coverage = compute_annotation_coverage(annots, session.duration_seconds or 0.0)

        # 1. Observation Count & Duration Check
        obs_count = metrics.total_observations if metrics else 0
        dur_s = session.duration_seconds or 0.0

        if obs_count == 0:
            fail_reasons.append("Zero observation records stored for this session.")
        elif obs_count < 30:
            warnings.append(f"Low observation count: only {obs_count} frames recorded.")

        if dur_s <= 0.0:
            warnings.append("Session duration is zero or negative.")

        # 2. Timing Jitter Analysis
        timing = metrics.timing if metrics else None
        eff_fps = timing.effective_fps if timing else 0.0
        mean_int = timing.mean_interval_ms if timing else 0.0
        std_int = timing.std_interval_ms if timing else 0.0

        if timing and timing.total_observations > 10:
            if timing.std_interval_ms > 150.0:
                warnings.append(f"High timestamp jitter detected (std={timing.std_interval_ms:.1f}ms).")
            if timing.max_interval_ms > 2000.0:
                warnings.append(f"Significant capture interruption detected (max gap={timing.max_interval_ms:.1f}ms).")

        # 3. Observability and Face Quality
        full_obs_pct = metrics.observability_distribution.get("FULL", 0.0) if metrics else 0.0
        unobs_pct = metrics.observability_distribution.get("UNOBSERVABLE", 0.0) if metrics else 0.0
        fq_mean = metrics.mean_face_quality if metrics else 0.0

        if unobs_pct > 60.0:
            warnings.append(f"Majority of session was unobservable ({unobs_pct:.1f}% frames unobservable).")

        # 4. Annotation Analysis
        if coverage.has_overlaps:
            warnings.append("Human annotations contain overlapping intervals.")

        # 5. Descriptive Comparison between Automated Output and Human Annotations
        descriptive_agreement: Optional[float] = None
        if annots and obs_count > 0:
            descriptive_agreement = self._compute_descriptive_comparison(session_id, annots)

        # 6. Video Recording & Queue Integrity Check
        if session.quality_status == "INTERRUPTED":
            fail_reasons.append(session.error_message or "Session recording was interrupted unexpectedly.")
        elif session.quality_status == "WARNING_DROPPED_FRAMES":
            warnings.append(session.error_message or "Video frames were dropped due to recording queue saturation.")

        if session.recording_path:
            sidecar = load_timestamp_sidecar(session.recording_path)
            if sidecar:
                dropped = sidecar.get("frames_dropped", 0)
                if dropped > 0:
                    drop_msg = f"Recording sidecar reports {dropped} dropped video frame(s)."
                    if drop_msg not in warnings:
                        warnings.append(drop_msg)
                w_err = sidecar.get("writer_error")
                if w_err:
                    fail_reasons.append(f"Video writer error in sidecar: {w_err}")

        # Determine Verdict
        if fail_reasons:
            verdict = AuditVerdict.FAIL
        elif warnings:
            if len(warnings) >= 3 or coverage.has_overlaps:
                verdict = AuditVerdict.REVIEW_REQUIRED
            else:
                verdict = AuditVerdict.PASS_WITH_WARNINGS
        else:
            verdict = AuditVerdict.PASS

        report = QualityAuditReport(
            session_id=session_id,
            verdict=verdict,
            audit_timestamp=now_utc_iso(),
            observation_count=obs_count,
            duration_seconds=dur_s,
            effective_fps=eff_fps,
            mean_interval_ms=mean_int,
            std_interval_ms=std_int,
            face_quality_mean=fq_mean,
            observability_full_fraction=round(full_obs_pct / 100.0, 3),
            annotation_coverage_fraction=coverage.coverage_fraction,
            has_overlapping_annotations=coverage.has_overlaps,
            descriptive_agreement_percentage=descriptive_agreement,
            warnings=warnings,
            reasons=fail_reasons if fail_reasons else warnings,
            details={
                "session_provenance": session.provenance,
                "collection_mode": session.collection_mode,
                "consent_confirmed": session.consent_confirmed,
                "timing": asdict(timing) if timing else {},
                "smoothed_distributions": metrics.smoothed_behavior_distribution if metrics else {},
                "raw_distributions": metrics.raw_behavior_distribution if metrics else {},
                "coverage": asdict(coverage),
            },
        )

        if persist_result:
            self._save_report_to_db(report)

        return report

    def _compute_descriptive_comparison(self, session_id: str, annots: list) -> Optional[float]:
        """Calculates descriptive system-to-annotation match percentage during annotated spans."""
        obs_rows = self.db.execute_query(
            "SELECT elapsed_seconds, smoothed_behavior, track_id FROM observations WHERE session_id = ?;",
            (session_id,)
        )
        if not obs_rows:
            return None

        total_annotated_observations = 0
        matching_observations = 0

        for r in obs_rows:
            el_s = float(r["elapsed_seconds"])
            sb = r["smoothed_behavior"]
            tid = r["track_id"]

            # Find matching annotation
            matched_annot = None
            for a in annots:
                if a.start_seconds <= el_s <= a.end_seconds:
                    if a.track_id is None or a.track_id == tid:
                        matched_annot = a
                        break

            if matched_annot is not None:
                total_annotated_observations += 1
                if matched_annot.behavior_label == sb:
                    matching_observations += 1

        if total_annotated_observations == 0:
            return None

        pct = (matching_observations / total_annotated_observations) * 100.0
        return round(pct, 1)

    def _save_report_to_db(self, report: QualityAuditReport) -> None:
        sql = """
        INSERT INTO quality_reports (
            session_id, verdict, audit_timestamp, observation_count,
            mean_interval_ms, fps_effective, face_quality_mean,
            observability_fraction, annotation_coverage_fraction,
            warning_count, warnings_json, report_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            verdict = excluded.verdict,
            audit_timestamp = excluded.audit_timestamp,
            observation_count = excluded.observation_count,
            mean_interval_ms = excluded.mean_interval_ms,
            fps_effective = excluded.fps_effective,
            face_quality_mean = excluded.face_quality_mean,
            observability_fraction = excluded.observability_fraction,
            annotation_coverage_fraction = excluded.annotation_coverage_fraction,
            warning_count = excluded.warning_count,
            warnings_json = excluded.warnings_json,
            report_json = excluded.report_json;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (
                report.session_id,
                report.verdict.value,
                report.audit_timestamp,
                report.observation_count,
                report.mean_interval_ms,
                report.effective_fps,
                report.face_quality_mean,
                report.observability_full_fraction,
                report.annotation_coverage_fraction,
                len(report.warnings),
                json.dumps(report.warnings),
                json.dumps(report.to_dict()),
            ))
            # Also update session quality status
            cur.execute(
                "UPDATE sessions SET quality_status = ? WHERE session_id = ?;",
                (report.verdict.value, report.session_id)
            )

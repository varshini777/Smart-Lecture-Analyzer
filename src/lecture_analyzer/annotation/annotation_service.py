"""Human annotation service managing ground-truth interval records."""

import logging
from typing import List, Optional

from lecture_analyzer.annotation.annotation_validation import (
    AnnotationValidationResult,
    CoverageReport,
    compute_annotation_coverage,
    validate_annotation,
)
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import AnnotationRecord, AnnotationRepository, SessionRepository
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.annotation.annotation_service")


class AnnotationService:
    """Provides high-level management of independent human ground-truth annotations."""

    def __init__(self, db: Database):
        self.db = db
        self.annot_repo = AnnotationRepository(db)
        self.session_repo = SessionRepository(db)

    def add_annotation(
        self,
        session_id: str,
        start_seconds: float,
        end_seconds: float,
        behavior_label: str,
        annotator_id: str,
        track_id: Optional[str] = None,
        confidence_level: str = "HIGH",
        notes: Optional[str] = None,
    ) -> AnnotationRecord:
        """Validates and persists a new independent human ground-truth annotation."""
        session = self.session_repo.get(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found in database.")

        val_result = validate_annotation(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            behavior_label=behavior_label,
            annotator_id=annotator_id,
            session_duration=session.duration_seconds,
        )

        if not val_result.is_valid:
            raise ValueError(val_result.error_message)

        if val_result.warnings:
            for w in val_result.warnings:
                logger.warning(f"Annotation warning for session {session_id}: {w}")

        record = AnnotationRecord(
            id=None,
            session_id=session_id,
            track_id=track_id,
            start_seconds=round(start_seconds, 2),
            end_seconds=round(end_seconds, 2),
            behavior_label=behavior_label,
            annotator_id=annotator_id.strip(),
            confidence_level=confidence_level,
            notes=notes.strip() if notes else None,
            created_at=now_utc_iso(),
        )

        inserted_id = self.annot_repo.save(record)
        record.id = inserted_id
        logger.info(f"Saved human annotation #{inserted_id} for session {session_id}: {behavior_label} [{start_seconds:.1f}s - {end_seconds:.1f}s]")
        return record

    def list_annotations(self, session_id: str) -> List[AnnotationRecord]:
        """Returns all annotations recorded for a session."""
        return self.annot_repo.list_by_session(session_id)

    def delete_annotation(self, annotation_id: int) -> bool:
        """Deletes an annotation by ID."""
        return self.annot_repo.delete(annotation_id)

    def get_coverage(self, session_id: str) -> Optional[CoverageReport]:
        """Calculates temporal coverage metrics for a session."""
        session = self.session_repo.get(session_id)
        if session is None or session.duration_seconds is None:
            return None

        annots = self.list_annotations(session_id)
        return compute_annotation_coverage(annots, session.duration_seconds)

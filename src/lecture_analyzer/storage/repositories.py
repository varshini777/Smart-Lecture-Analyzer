"""Data access repositories for sessions, observations, annotations, and audits."""

from dataclasses import dataclass
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.storage.database import Database
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.storage.repositories")


@dataclass
class SessionRecord:
    session_id: str
    participant_code: Optional[str]
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    collection_source: str
    provenance: str
    collection_mode: str
    consent_confirmed: bool
    recording_path: Optional[str]
    feature_schema_version: str
    pipeline_version: str
    model_versions: Optional[str]
    quality_status: str
    error_message: Optional[str]
    notes: Optional[str]


@dataclass
class AnnotationRecord:
    id: Optional[int]
    session_id: str
    track_id: Optional[str]
    start_seconds: float
    end_seconds: float
    behavior_label: str
    annotator_id: str
    confidence_level: str
    notes: Optional[str]
    created_at: str


class SessionRepository:
    """Repository managing Session lifecycles in the SQLite database."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, session: SessionRecord) -> None:
        sql = """
        INSERT INTO sessions (
            session_id, participant_code, start_time, end_time, duration_seconds,
            collection_source, provenance, collection_mode, consent_confirmed,
            recording_path, feature_schema_version, pipeline_version, model_versions,
            quality_status, error_message, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (
                session.session_id,
                session.participant_code,
                session.start_time,
                session.end_time,
                session.duration_seconds,
                session.collection_source,
                session.provenance,
                session.collection_mode,
                1 if session.consent_confirmed else 0,
                session.recording_path,
                session.feature_schema_version,
                session.pipeline_version,
                session.model_versions,
                session.quality_status,
                session.error_message,
                session.notes,
            ))

    def get(self, session_id: str) -> Optional[SessionRecord]:
        rows = self.db.execute_query("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
        if not rows:
            return None
        r = rows[0]
        return SessionRecord(
            session_id=r["session_id"],
            participant_code=r["participant_code"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            duration_seconds=r["duration_seconds"],
            collection_source=r["collection_source"],
            provenance=r["provenance"],
            collection_mode=r["collection_mode"],
            consent_confirmed=bool(r["consent_confirmed"]),
            recording_path=r["recording_path"],
            feature_schema_version=r["feature_schema_version"],
            pipeline_version=r["pipeline_version"],
            model_versions=r["model_versions"],
            quality_status=r["quality_status"],
            error_message=r["error_message"],
            notes=r["notes"],
        )

    def list_all(self, provenance: Optional[str] = None) -> List[SessionRecord]:
        if provenance:
            rows = self.db.execute_query(
                "SELECT * FROM sessions WHERE provenance = ? ORDER BY start_time DESC;", (provenance,)
            )
        else:
            rows = self.db.execute_query("SELECT * FROM sessions ORDER BY start_time DESC;")

        return [
            SessionRecord(
                session_id=r["session_id"],
                participant_code=r["participant_code"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                duration_seconds=r["duration_seconds"],
                collection_source=r["collection_source"],
                provenance=r["provenance"],
                collection_mode=r["collection_mode"],
                consent_confirmed=bool(r["consent_confirmed"]),
                recording_path=r["recording_path"],
                feature_schema_version=r["feature_schema_version"],
                pipeline_version=r["pipeline_version"],
                model_versions=r["model_versions"],
                quality_status=r["quality_status"],
                error_message=r["error_message"],
                notes=r["notes"],
            )
            for r in rows
        ]

    def finish(
        self,
        session_id: str,
        end_time: str,
        duration_seconds: float,
        quality_status: str = "PENDING_AUDIT",
        error_message: Optional[str] = None
    ) -> None:
        sql = """
        UPDATE sessions
        SET end_time = ?, duration_seconds = ?, quality_status = ?, error_message = ?
        WHERE session_id = ?;
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (end_time, duration_seconds, quality_status, error_message, session_id))

    def delete(self, session_id: str) -> bool:
        """Deletes session and all cascading dependent records."""
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
            return cur.rowcount > 0


class ObservationRepository:
    """Manages high-throughput incremental observation persistence."""

    def __init__(self, db: Database):
        self.db = db

    def insert_batch(
        self,
        session_id: str,
        records: List[Tuple[ObservationFeatures, SmoothedBehaviorResult, int, float]]
    ) -> int:
        """Inserts a batch of observation records in a single database transaction.

        Records format: list of (features, smoothed_result, frame_index, elapsed_seconds)
        """
        if not records:
            return 0

        sql = """
        INSERT INTO observations (
            session_id, track_id, frame_index, wall_timestamp, monotonic_ts, elapsed_seconds,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            ear_avg, ear_left, ear_right, mar, face_quality, face_detected,
            head_yaw, head_pitch, head_roll, head_pose_valid,
            pose_detected, shoulder_tilt_deg, torso_upright_ratio, head_shoulder_offset,
            wrists_near_head, wrists_near_table,
            phone_detected, phone_confidence,
            observability_level, observability_reasons,
            raw_behavior, raw_score, smoothed_behavior, smoothed_score, score_type
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?, ?, ?
        );
        """
        rows_to_insert = []
        track_updates: Dict[str, Tuple[str, str, int]] = {}

        for feats, smoothed, frame_idx, elapsed_s in records:
            rows_to_insert.append((
                session_id,
                feats.track_id,
                frame_idx,
                feats.wall_timestamp,
                feats.monotonic_ts,
                elapsed_s,
                feats.bbox_x1, feats.bbox_y1, feats.bbox_x2, feats.bbox_y2,
                feats.ear_avg, feats.ear_left, feats.ear_right, feats.mar, feats.face_quality,
                1 if feats.face_detected else 0,
                feats.head_yaw, feats.head_pitch, feats.head_roll,
                1 if feats.head_pose_valid else 0,
                1 if feats.pose_detected else 0,
                feats.shoulder_tilt_deg, feats.torso_upright_ratio, feats.head_shoulder_offset,
                1 if feats.wrists_near_head else 0,
                1 if feats.wrists_near_table else 0,
                1 if feats.phone_detected else 0,
                feats.phone_confidence,
                feats.observability_level,
                feats.observability_reasons,
                smoothed.raw_behavior.value,
                smoothed.raw_score,
                smoothed.smoothed_behavior.value,
                smoothed.smoothed_score,
                "HEURISTIC_RULE_SCORE",
            ))

            # Track aggregates
            tid = feats.track_id
            if tid not in track_updates:
                track_updates[tid] = (feats.wall_timestamp, feats.wall_timestamp, 1)
            else:
                first, _, cnt = track_updates[tid]
                track_updates[tid] = (first, feats.wall_timestamp, cnt + 1)

        with self.db.transaction() as cur:
            cur.executemany(sql, rows_to_insert)

            # Update session_tracks table
            for tid, (first_ts, last_ts, count) in track_updates.items():
                cur.execute("""
                    INSERT INTO session_tracks (session_id, track_id, first_seen_ts, last_seen_ts, total_observations)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, track_id) DO UPDATE SET
                        last_seen_ts = excluded.last_seen_ts,
                        total_observations = total_observations + excluded.total_observations;
                """, (session_id, tid, first_ts, last_ts, count))

        return len(rows_to_insert)

    def get_session_observations(self, session_id: str, track_id: Optional[str] = None) -> List[sqlite3.Row]:
        if track_id:
            return self.db.execute_query(
                "SELECT * FROM observations WHERE session_id = ? AND track_id = ? ORDER BY elapsed_seconds ASC;",
                (session_id, track_id)
            )
        return self.db.execute_query(
            "SELECT * FROM observations WHERE session_id = ? ORDER BY elapsed_seconds ASC;",
            (session_id,)
        )

    def count(self, session_id: str) -> int:
        rows = self.db.execute_query("SELECT COUNT(*) FROM observations WHERE session_id = ?;", (session_id,))
        return rows[0][0] if rows else 0


class AnnotationRepository:
    """Manages ground-truth human annotations independently of automated predictions."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, record: AnnotationRecord) -> int:
        sql = """
        INSERT INTO annotations (
            session_id, track_id, start_seconds, end_seconds,
            behavior_label, annotator_id, confidence_level, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (
                record.session_id,
                record.track_id,
                record.start_seconds,
                record.end_seconds,
                record.behavior_label,
                record.annotator_id,
                record.confidence_level,
                record.notes,
                record.created_at,
            ))
            return cur.lastrowid

    def list_by_session(self, session_id: str) -> List[AnnotationRecord]:
        rows = self.db.execute_query(
            "SELECT * FROM annotations WHERE session_id = ? ORDER BY start_seconds ASC;", (session_id,)
        )
        return [
            AnnotationRecord(
                id=r["id"],
                session_id=r["session_id"],
                track_id=r["track_id"],
                start_seconds=r["start_seconds"],
                end_seconds=r["end_seconds"],
                behavior_label=r["behavior_label"],
                annotator_id=r["annotator_id"],
                confidence_level=r["confidence_level"],
                notes=r["notes"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete(self, annotation_id: int) -> bool:
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM annotations WHERE id = ?;", (annotation_id,))
            return cur.rowcount > 0

"""Session lifecycle service coordinating collection, storage, and video archival."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.temporal_smoothing import SmoothedBehaviorResult
from lecture_analyzer.collection.consent import verify_research_collection_eligibility
from lecture_analyzer.collection.recording_service import VideoRecordingService
from lecture_analyzer.config import AppConfig
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import ObservationRepository, SessionRecord, SessionRepository
from lecture_analyzer.utils.paths import resolve_path, safe_relative_path
from lecture_analyzer.utils.timestamps import monotonic_time, now_utc_iso

logger = logging.getLogger("lecture_analyzer.collection.session_service")


class SessionService:
    """Manages active session lifecycle, incremental persistence, and recording coordination."""

    def __init__(self, config: Optional[AppConfig] = None, db: Optional[Database] = None):
        self.config = config or AppConfig()
        self.db = db or Database(self.config.storage.database_path)
        self.session_repo = SessionRepository(self.db)
        self.obs_repo = ObservationRepository(self.db)
        self.recording_service = VideoRecordingService(self.config.storage.recordings_dir)

        # Active session state
        self.active_session_id: Optional[str] = None
        self.active_record: Optional[SessionRecord] = None
        self.session_start_monotonic: float = 0.0
        self.last_batch_write_monotonic: float = 0.0
        self.total_observations_recorded = 0

        # In-memory batch buffer for incremental persistence
        self._batch_buffer: List[Tuple[ObservationFeatures, SmoothedBehaviorResult, int, float]] = []

    def start_session(
        self,
        collection_source: str,
        provenance: DataProvenance,
        collection_mode: str,
        participant_code: Optional[str] = None,
        consent_confirmed: bool = False,
        record_video: bool = False,
        frame_width: int = 1280,
        frame_height: int = 720,
        notes: Optional[str] = None,
        model_versions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Starts a new observation session and registers it in SQLite."""
        if self.is_session_active:
            self.stop_session()

        # If REAL_HUMAN provenance is requested, enforce consent gate
        if provenance == DataProvenance.REAL_HUMAN:
            gate = verify_research_collection_eligibility(
                participant_code=participant_code,
                consent_confirmed=consent_confirmed,
                is_live_camera=(collection_mode in ["LIVE_CAMERA", "CONTROLLED_STUDY"]),
            )
            if not gate.is_valid:
                raise ValueError(f"Research collection consent gate failed: {gate.error_message}")

        # Generate unique session ID: SES_YYYYMMDD_HHMMSS_xxxx
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        session_id = f"SES_{ts_str}_{rand_suffix}"

        # Record session start monotonic time origin
        self.session_start_monotonic = monotonic_time()
        self.last_batch_write_monotonic = self.session_start_monotonic
        self.total_observations_recorded = 0
        self._batch_buffer.clear()

        # Initialize optional video recording if requested and consented
        rec_path_str: Optional[str] = None
        if record_video:
            saved_path = self.recording_service.start_recording(
                session_id=session_id,
                frame_width=frame_width,
                frame_height=frame_height,
                nominal_fps=30.0,
                codec=self.config.privacy.video_codec,
                session_start_monotonic=self.session_start_monotonic,
            )
            if saved_path:
                rec_path_str = safe_relative_path(saved_path)

        # Create session record
        record = SessionRecord(
            session_id=session_id,
            participant_code=participant_code.strip() if participant_code else None,
            start_time=now_utc_iso(),
            end_time=None,
            duration_seconds=None,
            collection_source=collection_source,
            provenance=provenance.value,
            collection_mode=collection_mode,
            consent_confirmed=consent_confirmed,
            recording_path=rec_path_str,
            feature_schema_version="1.0.0",
            pipeline_version=self.config.system.version,
            model_versions=json.dumps(model_versions or {}),
            quality_status="PENDING_AUDIT",
            error_message=None,
            notes=notes,
        )

        self.session_repo.create(record)
        self.active_session_id = session_id
        self.active_record = record

        logger.info(
            f"Session started: {session_id} [Provenance: {provenance.value}, Mode: {collection_mode}, "
            f"Participant: {participant_code or 'ANONYMOUS'}]"
        )
        return session_id

    @property
    def is_session_active(self) -> bool:
        return self.active_session_id is not None

    def record_video_frame(self, packet: Any) -> bool:
        """Submits an acquired camera FramePacket directly to the video recording service."""
        if not self.is_session_active or not self.recording_service.is_recording or packet is None:
            return False
        return self.recording_service.write_frame_packet(packet, self.session_start_monotonic)

    def record_frame_observations(
        self,
        observations: List[Tuple[ObservationFeatures, SmoothedBehaviorResult, int]],
        frame_bgr: Optional[any] = None,
    ) -> None:
        """Appends observations to the in-memory batch and flushes incrementally."""
        if not self.is_session_active or not self.active_session_id:
            return

        # Write frame to video file if recording (fallback for standalone test callers)
        if frame_bgr is not None and self.recording_service.is_recording:
            self.recording_service.write_frame(frame_bgr, session_start_monotonic=self.session_start_monotonic)

        now_mono = monotonic_time()
        for feats, smoothed, frame_idx in observations:
            # Consistent time mapping: session-relative time uses frame acquisition monotonic_ts
            obs_elapsed_s = max(0.0, feats.monotonic_ts - self.session_start_monotonic)
            self._batch_buffer.append((feats, smoothed, frame_idx, round(obs_elapsed_s, 3)))
            self.total_observations_recorded += 1

        # Check batch flush triggers
        time_since_last_flush = now_mono - self.last_batch_write_monotonic
        if (time_since_last_flush >= self.config.storage.batch_write_interval_seconds or
                len(self._batch_buffer) >= self.config.storage.batch_write_max_queue):
            self.flush_buffer()

    def flush_buffer(self) -> int:
        """Flushes buffered observations to SQLite inside a single transaction."""
        if not self.is_session_active or not self.active_session_id or not self._batch_buffer:
            return 0

        to_write = list(self._batch_buffer)
        self._batch_buffer.clear()
        self.last_batch_write_monotonic = monotonic_time()

        try:
            inserted = self.obs_repo.insert_batch(self.active_session_id, to_write)
            logger.debug(f"Flushed {inserted} observations to database for {self.active_session_id}.")
            return inserted
        except Exception as e:
            logger.error(f"Failed to flush observation batch: {e}", exc_info=True)
            # Re-queue on failure so records aren't dropped
            self._batch_buffer = to_write + self._batch_buffer
            return 0

    def stop_session(self, error_message: Optional[str] = None) -> Optional[SessionRecord]:
        """Stops the active session, flushes remaining data, and closes video writers."""
        if not self.is_session_active or not self.active_session_id:
            return None

        # 1. Flush in-memory observations
        self.flush_buffer()

        # 2. Stop video recording
        if self.recording_service.is_recording:
            self.recording_service.stop_recording()

        # 3. Check for dropped frames or recording errors
        dropped_frames = self.recording_service.last_frames_dropped
        writer_err = self.recording_service.last_writer_error

        # 4. Finalize session metrics and evaluate quality status
        end_time_iso = now_utc_iso()
        duration_s = max(0.0, monotonic_time() - self.session_start_monotonic)

        quality_status = "PENDING_AUDIT"
        if error_message:
            quality_status = "INTERRUPTED"
        elif writer_err:
            quality_status = "INTERRUPTED"
            error_message = f"Video writer error: {writer_err}"
        elif dropped_frames > 0:
            quality_status = "WARNING_DROPPED_FRAMES"
            error_message = f"Warning: {dropped_frames} video frame(s) dropped due to recording queue saturation."

        self.session_repo.finish(
            session_id=self.active_session_id,
            end_time=end_time_iso,
            duration_seconds=round(duration_s, 2),
            quality_status=quality_status,
            error_message=error_message,
        )

        completed_record = self.session_repo.get(self.active_session_id)
        logger.info(
            f"Session {self.active_session_id} finished. Duration: {duration_s:.1f}s, "
            f"Total observations: {self.total_observations_recorded}"
        )

        self.active_session_id = None
        self.active_record = None
        self._batch_buffer.clear()
        return completed_record

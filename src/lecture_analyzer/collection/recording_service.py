"""Local raw video recording service supporting precision timestamp sidecars and non-blocking capture."""

import bisect
import json
import logging
from pathlib import Path
import queue
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.timestamps import monotonic_time, now_utc_iso

logger = logging.getLogger("lecture_analyzer.collection.recording_service")


class VideoRecordingService:
    """Safely manages local video file recording with precision timestamp tracking and async queue writes."""

    def __init__(self, recordings_dir: str = "data/recordings", min_free_mb: int = 500, queue_maxsize: int = 150):
        self.recordings_dir = resolve_path(recordings_dir)
        self.min_free_mb = min_free_mb
        self.queue_maxsize = queue_maxsize

        self.writer: Optional[cv2.VideoWriter] = None
        self.current_file_path: Optional[Path] = None
        self.current_sidecar_path: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.nominal_fps: float = 30.0
        self.frames_written = 0
        self.is_recording = False

        # Async writer queue & thread
        self._queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_metadata: List[Dict[str, Any]] = []
        self._session_start_monotonic: float = 0.0
        self._next_video_frame_idx: int = 0
        self._frames_dropped: int = 0
        self._writer_error: Optional[str] = None
        self.last_frames_dropped: int = 0
        self.last_writer_error: Optional[str] = None

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def writer_error(self) -> Optional[str]:
        return self._writer_error

    def start_recording(
        self,
        session_id: str,
        frame_width: int,
        frame_height: int,
        nominal_fps: float = 30.0,
        codec: str = "mp4v",
        session_start_monotonic: Optional[float] = None,
    ) -> Optional[Path]:
        """Initializes a non-blocking VideoWriter for the session with precision timestamp sidecar."""
        self.stop_recording()
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        # Disk space check
        total, used, free = shutil.disk_usage(str(self.recordings_dir))
        free_mb = free / (1024 * 1024)
        if free_mb < self.min_free_mb:
            logger.error(
                f"Insufficient disk space ({free_mb:.1f} MB free, required {self.min_free_mb} MB). "
                f"Video recording aborted."
            )
            return None

        video_filename = f"{session_id}.mp4"
        sidecar_filename = f"{session_id}_timestamps.json"
        out_path = self.recordings_dir / video_filename
        sidecar_path = self.recordings_dir / sidecar_filename

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            valid_nominal_fps = max(1.0, float(nominal_fps))
            self.writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                valid_nominal_fps,
                (frame_width, frame_height),
                True
            )
            if not self.writer.isOpened():
                logger.error(f"Failed to open VideoWriter for {out_path}")
                self.writer = None
                return None

            self.session_id = session_id
            self.current_file_path = out_path
            self.current_sidecar_path = sidecar_path
            self.nominal_fps = valid_nominal_fps
            self.frames_written = 0
            self._next_video_frame_idx = 0
            self._frames_dropped = 0
            self._writer_error = None
            self.last_frames_dropped = 0
            self.last_writer_error = None
            self._frame_metadata.clear()
            self._session_start_monotonic = session_start_monotonic if session_start_monotonic is not None else monotonic_time()

            # Start background writer thread
            self._stop_event.clear()
            self._queue = queue.Queue(maxsize=self.queue_maxsize)
            self._worker_thread = threading.Thread(
                target=self._writer_worker,
                name=f"VideoWriter-{session_id}",
                daemon=True,
            )
            self._worker_thread.start()

            self.is_recording = True
            logger.info(
                f"Video recording started: {out_path.name} "
                f"({frame_width}x{frame_height} @ {valid_nominal_fps:.1f} nominal FPS)"
            )
            return out_path
        except Exception as e:
            logger.error(f"Exception creating VideoWriter: {e}", exc_info=True)
            self.stop_recording()
            return None

    def _writer_worker(self) -> None:
        """Background worker consuming frames from the queue and writing to disk."""
        logger.debug(f"Writer thread started for {self.current_file_path}")
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                # Sentinel signaling end of stream
                self._queue.task_done()
                break

            frame_bgr, meta_dict = item
            try:
                if self.writer is not None:
                    self.writer.write(frame_bgr)
                    self._frame_metadata.append(meta_dict)
                    self.frames_written += 1
            except Exception as e:
                self._writer_error = str(e)
                logger.error(f"Error in video writer worker: {e}")
            finally:
                self._queue.task_done()

        logger.debug(f"Writer thread finished. Total frames written: {self.frames_written}")

    def write_frame_packet(self, packet: Any, session_start_monotonic: float) -> bool:
        """Enqueues a FramePacket for asynchronous recording."""
        if not self.is_recording or self._queue is None or packet is None:
            return False

        return self.write_frame(
            frame_bgr=packet.frame,
            camera_frame_idx=packet.frame_index,
            monotonic_ts=packet.monotonic_ts,
            wall_timestamp=packet.wall_timestamp,
            session_start_monotonic=session_start_monotonic,
        )

    def write_frame(
        self,
        frame_bgr: np.ndarray,
        camera_frame_idx: int = 0,
        monotonic_ts: Optional[float] = None,
        wall_timestamp: Optional[str] = None,
        session_start_monotonic: Optional[float] = None,
    ) -> bool:
        """Enqueues a single frame and metadata to the async video writer queue."""
        if not self.is_recording or self._queue is None or frame_bgr is None:
            return False

        mono_ts = monotonic_ts if monotonic_ts is not None else monotonic_time()
        wall_ts = wall_timestamp if wall_timestamp is not None else now_utc_iso()
        sess_start = session_start_monotonic if session_start_monotonic is not None else self._session_start_monotonic

        elapsed_s = max(0.0, mono_ts - sess_start)

        meta = {
            "video_frame_idx": self._next_video_frame_idx,
            "camera_frame_idx": camera_frame_idx,
            "elapsed_s": round(elapsed_s, 4),
            "monotonic_ts": round(mono_ts, 4),
            "wall_timestamp": wall_ts,
        }
        self._next_video_frame_idx += 1

        try:
            self._queue.put_nowait((frame_bgr, meta))
            return True
        except queue.Full:
            self._frames_dropped += 1
            logger.warning(
                f"Video writer queue full ({self.queue_maxsize}). "
                f"Dropped video frame {camera_frame_idx} (total dropped: {self._frames_dropped})."
            )
            return False

    def stop_recording(self) -> Optional[Tuple[Path, int, Optional[Path]]]:
        """Finalizes the video file, writes the timestamp sidecar, and releases resources."""
        if not self.is_recording and self.writer is None:
            return None

        self.is_recording = False
        self._stop_event.set()

        # Stop queue worker and flush remaining frames
        if self._queue is not None:
            try:
                self._queue.put(None, timeout=0.5)
            except Exception:
                pass

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                unflushed = self._queue.qsize() if self._queue else 0
                self._frames_dropped += unflushed
                self._writer_error = f"Writer timed out during shutdown; {unflushed} frames could not be flushed."
                logger.error(self._writer_error)
            self._worker_thread = None

        # Capture finalized status before instance state is cleared
        self.last_frames_dropped = self._frames_dropped
        self.last_writer_error = self._writer_error

        # Release OpenCV VideoWriter
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                logger.warning(f"Error releasing VideoWriter: {e}")
            self.writer = None

        result = None
        saved_video_path = self.current_file_path
        saved_sidecar_path = self.current_sidecar_path
        total_written = self.frames_written

        if saved_video_path and total_written > 0:
            # Generate precision timestamp sidecar
            first_ts = self._frame_metadata[0]["monotonic_ts"]
            last_ts = self._frame_metadata[-1]["monotonic_ts"]
            elapsed_span = max(0.001, last_ts - first_ts)
            actual_fps = (total_written / elapsed_span) if elapsed_span > 0.05 else self.nominal_fps
            session_duration_s = self._frame_metadata[-1]["elapsed_s"]
            video_duration_s = total_written / self.nominal_fps

            sidecar_data = {
                "session_id": self.session_id,
                "video_file": saved_video_path.name,
                "nominal_fps": round(self.nominal_fps, 2),
                "actual_fps": round(actual_fps, 2),
                "total_frames": total_written,
                "frames_dropped": self._frames_dropped,
                "first_frame_monotonic": first_ts,
                "last_frame_monotonic": last_ts,
                "session_duration_s": round(session_duration_s, 3),
                "video_duration_s": round(video_duration_s, 3),
                "writer_error": self._writer_error,
                "frames": self._frame_metadata,
            }

            if saved_sidecar_path:
                try:
                    with open(saved_sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar_data, f, indent=2)
                    logger.info(f"Precision timestamp sidecar saved: {saved_sidecar_path.name}")
                except Exception as e:
                    logger.error(f"Failed to write timestamp sidecar {saved_sidecar_path}: {e}")

            result = (saved_video_path, total_written, saved_sidecar_path)
            logger.info(
                f"Video recording finalized: {saved_video_path.name} "
                f"({total_written} frames, {actual_fps:.1f} actual FPS, {video_duration_s:.1f}s video duration)"
            )

        self.current_file_path = None
        self.current_sidecar_path = None
        self.session_id = None
        self.frames_written = 0
        self._next_video_frame_idx = 0
        self._frames_dropped = 0
        self._frame_metadata.clear()
        self._queue = None

        return result


def load_timestamp_sidecar(video_or_sidecar_path: Path | str) -> Optional[Dict[str, Any]]:
    """Loads a precision timestamp sidecar JSON for a given video or sidecar file path."""
    p = resolve_path(str(video_or_sidecar_path))
    if p.suffix.lower() == ".json":
        sidecar_path = p
    else:
        sidecar_path = p.with_name(f"{p.stem}_timestamps.json")

    if not sidecar_path.exists():
        return None

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load timestamp sidecar from {sidecar_path}: {e}")
        return None


def frame_idx_for_time(sidecar: Dict[str, Any], elapsed_s: float) -> int:
    """Finds the video frame index corresponding to a session elapsed time using binary search."""
    frames = sidecar.get("frames", [])
    if not frames:
        fps = sidecar.get("nominal_fps", 30.0)
        return max(0, int(elapsed_s * fps))

    elapsed_list = [f["elapsed_s"] for f in frames]
    idx = bisect.bisect_left(elapsed_list, elapsed_s)
    if idx >= len(elapsed_list):
        return len(elapsed_list) - 1
    if idx > 0 and (elapsed_s - elapsed_list[idx - 1] < elapsed_list[idx] - elapsed_s):
        return idx - 1
    return idx


def time_for_frame_idx(sidecar: Dict[str, Any], frame_idx: int) -> float:
    """Returns the session elapsed seconds corresponding to a given video frame index."""
    frames = sidecar.get("frames", [])
    if not frames:
        fps = sidecar.get("nominal_fps", 30.0)
        return frame_idx / fps if fps > 0 else 0.0

    valid_idx = max(0, min(frame_idx, len(frames) - 1))
    return float(frames[valid_idx]["elapsed_s"])

"""Camera service providing robust physical webcam and video file acquisition."""

from dataclasses import dataclass
from enum import Enum
import logging
import platform
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from lecture_analyzer.config import CameraConfig
from lecture_analyzer.utils.timestamps import monotonic_time, now_utc_iso

logger = logging.getLogger("lecture_analyzer.vision.camera")


class CameraLifecycleState(str, Enum):
    """Explicit camera lifecycle state machine states."""
    CLOSED = "CLOSED"
    OPENING = "OPENING"
    READY = "READY"
    STREAMING = "STREAMING"
    STOPPING = "STOPPING"


@dataclass
class FramePacket:
    """Represents a single acquired video frame with precision timing metadata."""
    frame: np.ndarray
    frame_index: int
    wall_timestamp: str       # ISO8601 UTC string
    monotonic_ts: float       # High-resolution monotonic timestamp (seconds)
    acquisition_latency_ms: float
    width: int
    height: int


class CameraService:
    """Manages physical camera acquisition, lifecycle, error recovery, and frame timing."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.state: CameraLifecycleState = CameraLifecycleState.CLOSED
        self.actual_backend: str = "NONE"
        self.actual_source: Optional[int | str] = None
        self.frame_index = 0
        self.last_frame_ts: Optional[float] = None
        self.actual_width = 0
        self.actual_height = 0
        self.effective_fps = 0.0
        self.first_frame_acquired = False
        self.last_error: Optional[str] = None
        self._fps_alpha = 0.1  # Exponential moving average smoothing for FPS
        self.is_file: bool = False
        self.file_fps: float = 30.0
        self.total_file_frames: int = 0

    def open(self, source: Optional[int | str] = None) -> bool:
        """Opens camera index or video file path with verified first frame and backend fallback.

        Returns True if successfully opened, verified, and ready; False otherwise.
        """
        self.state = CameraLifecycleState.OPENING
        self.release()
        self.last_error = None
        self.first_frame_acquired = False

        cam_source = self.config.index if source is None else source
        self.actual_source = cam_source

        logger.info(f"Opening camera source: {cam_source} (configured backend={self.config.backend})")

        # Determine backend candidates to try in priority order
        candidates = []
        is_windows = platform.system() == "Windows"

        if isinstance(cam_source, int):
            if is_windows:
                cfg_backend = (self.config.backend or "DEFAULT").upper()
                if cfg_backend == "DSHOW":
                    candidates = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "CAP_ANY")]
                elif cfg_backend == "MSMF":
                    candidates = [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "CAP_ANY")]
                else:  # DEFAULT or other: try DirectShow first as standard Windows webcam API
                    candidates = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "CAP_ANY")]
            else:
                candidates = [(cv2.CAP_ANY, "DEFAULT")]
        else:
            candidates = [(cv2.CAP_ANY, "FILE")]

        opened_successfully = False

        for backend_flag, backend_name in candidates:
            temp_cap = None
            try:
                logger.debug(f"Attempting to open camera source {cam_source} via backend {backend_name}...")
                temp_cap = cv2.VideoCapture(cam_source, backend_flag)

                if not temp_cap.isOpened():
                    logger.debug(f"Backend {backend_name} failed to open camera {cam_source}.")
                    temp_cap.release()
                    continue

                # Set requested dimensions and minimize internal driver buffer for physical camera
                if isinstance(cam_source, int):
                    temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                    try:
                        temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                # Verify actual hardware delivery by test-reading the first frame
                ret, test_frame = temp_cap.read()
                if not ret or test_frame is None or test_frame.size == 0 or len(test_frame.shape) < 2:
                    logger.warning(
                        f"Camera {cam_source} opened with backend {backend_name} but initial test frame read failed."
                    )
                    temp_cap.release()
                    continue

                # Successfully verified live frame
                h, w = test_frame.shape[:2]
                self.cap = temp_cap
                self.actual_width = w
                self.actual_height = h
                self.actual_backend = backend_name
                self.is_opened = True
                self.first_frame_acquired = True
                self.frame_index = 0
                self.last_frame_ts = monotonic_time()
                self.effective_fps = 0.0
                self.is_file = isinstance(cam_source, str)
                if self.is_file:
                    try:
                        raw_fps = float(temp_cap.get(cv2.CAP_PROP_FPS))
                        self.file_fps = raw_fps if raw_fps > 0 and not np.isnan(raw_fps) else 30.0
                    except (TypeError, ValueError):
                        self.file_fps = 30.0
                    try:
                        self.total_file_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    except (TypeError, ValueError):
                        self.total_file_frames = 0
                else:
                    self.file_fps = float(getattr(self.config, "fps", 30.0))
                    self.total_file_frames = 0

                self.state = CameraLifecycleState.READY
                opened_successfully = True

                logger.info(
                    f"Camera source {cam_source} successfully opened via {backend_name}. "
                    f"First frame verified: {w}x{h} ({test_frame.dtype}), is_file={self.is_file}, fps={self.file_fps}"
                )
                break


            except Exception as e:
                logger.warning(f"Exception trying backend {backend_name} on camera {cam_source}: {e}")
                if temp_cap is not None:
                    try:
                        temp_cap.release()
                    except Exception:
                        pass

        if not opened_successfully:
            self.state = CameraLifecycleState.CLOSED
            self.is_opened = False
            self.cap = None
            self.last_error = (
                f"Failed to open video source {cam_source}. "
                f"Attempted backends: {[b[1] for b in candidates]}. "
                "Ensure device is connected, not in use by another application, and access permissions are granted."
            )
            logger.error(self.last_error)
            return False

        return True

    def read_frame(self, loop: bool = True) -> Optional[FramePacket]:
        """Reads a single frame from the camera with accurate acquisition timestamps."""
        if not self.is_opened or self.cap is None:
            return None

        if self.state == CameraLifecycleState.READY:
            self.state = CameraLifecycleState.STREAMING

        # If reading from a video file, pace frame acquisition to native video FPS
        if self.is_file and self.file_fps > 0 and self.last_frame_ts is not None:
            target_interval = 1.0 / self.file_fps
            elapsed = monotonic_time() - self.last_frame_ts
            sleep_needed = target_interval - elapsed
            if sleep_needed > 0.002:
                time.sleep(min(sleep_needed, 0.05))

        t_start = monotonic_time()
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                if self.is_file and loop:
                    logger.debug(f"Looping video file {self.actual_source} from frame 0.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret or frame is None or frame.size == 0:
                    logger.warning(f"Failed to read frame at index {self.frame_index}")
                    return None


            t_acquired = monotonic_time()
            wall_ts = now_utc_iso()
            latency_ms = (t_acquired - t_start) * 1000.0

            # Calculate real effective FPS from elapsed monotonic delta
            if self.last_frame_ts is not None:
                delta = t_acquired - self.last_frame_ts
                if delta > 0.001:
                    instant_fps = 1.0 / delta
                    if self.effective_fps == 0.0:
                        self.effective_fps = instant_fps
                    else:
                        self.effective_fps = (1.0 - self._fps_alpha) * self.effective_fps + self._fps_alpha * instant_fps

            self.last_frame_ts = t_acquired
            self.frame_index += 1

            h, w = frame.shape[:2]

            logger.debug(
                f"Camera frame acquired: #{self.frame_index} ({w}x{h}), "
                f"latency={latency_ms:.1f}ms, acq_fps={self.effective_fps:.1f}"
            )

            return FramePacket(
                frame=frame,
                frame_index=self.frame_index,
                wall_timestamp=wall_ts,
                monotonic_ts=t_acquired,
                acquisition_latency_ms=latency_ms,
                width=w,
                height=h,
            )
        except Exception as e:
            logger.error(f"Error during frame acquisition: {e}", exc_info=True)
            return None

    def release(self) -> None:
        """Safely releases the camera and resets state."""
        self.state = CameraLifecycleState.STOPPING
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera capture: {e}")
            self.cap = None
        self.is_opened = False
        self.state = CameraLifecycleState.CLOSED
        logger.info("Camera capture released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


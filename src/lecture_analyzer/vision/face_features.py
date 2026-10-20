"""MediaPipe 3D FaceLandmarker wrapper and facial feature extraction (EAR, MAR, quality)."""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from lecture_analyzer.config import FacialConfig, ModelsConfig
from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.validation import clamp, is_finite_number

logger = logging.getLogger("lecture_analyzer.vision.face_features")

# Standard MediaPipe FaceMesh indices for EAR
# Left eye
LEFT_EYE_HORIZ = (33, 133)
LEFT_EYE_VERT1 = (160, 144)
LEFT_EYE_VERT2 = (158, 153)

# Right eye
RIGHT_EYE_HORIZ = (362, 263)
RIGHT_EYE_VERT1 = (385, 380)
RIGHT_EYE_VERT2 = (387, 373)

# Mouth indices for MAR
MOUTH_HORIZ = (61, 291)
MOUTH_VERT_MID = (13, 14)
MOUTH_VERT_LEFT = (81, 178)
MOUTH_VERT_RIGHT = (311, 402)

# Head pose 6 points: Nose, Chin, LeftEyeOuter, RightEyeOuter, LeftMouth, RightMouth
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class FaceObservation:
    """Extracted geometric features for a single detected face."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    ear_left: Optional[float]
    ear_right: Optional[float]
    ear_avg: Optional[float]
    mar: Optional[float]
    face_quality: float              # [0.0, 1.0]
    head_pose_points_2d: Optional[np.ndarray] = None  # (6, 2)
    landmarks_pixel: Optional[np.ndarray] = None      # (N, 2)
    is_valid: bool = True
    error_message: Optional[str] = None


class FaceFeatureExtractor:
    """Runs MediaPipe FaceLandmarker and computes geometry-based observable metrics."""

    def __init__(self, models_cfg: Optional[ModelsConfig] = None, facial_cfg: Optional[FacialConfig] = None):
        self.models_cfg = models_cfg or ModelsConfig()
        self.facial_cfg = facial_cfg or FacialConfig()
        self.detector: Optional[vision.FaceLandmarker] = None
        self.is_initialized = False

        self._initialize_detector()

    def _initialize_detector(self) -> None:
        model_path = resolve_path(self.models_cfg.face_landmarker_path)
        if not model_path.exists():
            logger.warning(
                f"FaceLandmarker model not found at {model_path}. "
                "Facial feature extraction will run in fallback (disabled) mode."
            )
            self.is_initialized = False
            return

        try:
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.is_initialized = True
            logger.info("FaceLandmarker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceLandmarker: {e}", exc_info=True)
            self.is_initialized = False

    def process_frame(self, frame_bgr: np.ndarray) -> List[FaceObservation]:
        """Detects faces in frame and extracts geometric features."""
        if not self.is_initialized or self.detector is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            logger.debug(f"FaceLandmarker inference error: {e}")
            return []

        if not result.face_landmarks:
            return []

        observations: List[FaceObservation] = []

        for face_lms in result.face_landmarks:
            obs = self._extract_face_metrics(face_lms, frame_bgr, w, h)
            if obs is not None:
                observations.append(obs)

        return observations

    def _extract_face_metrics(
        self,
        face_lms,
        frame_bgr: np.ndarray,
        w: int,
        h: int
    ) -> Optional[FaceObservation]:
        """Calculates EAR, MAR, face quality, and head pose points for one landmark set."""
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)

        if len(pts) < 468:
            return None

        # Bounding box
        min_x = max(0, int(np.min(pts[:, 0])))
        min_y = max(0, int(np.min(pts[:, 1])))
        max_x = min(w - 1, int(np.max(pts[:, 0])))
        max_y = min(h - 1, int(np.max(pts[:, 1])))

        bw = max_x - min_x
        bh = max_y - min_y
        if bw <= 10 or bh <= 10:
            return None

        bbox = (min_x, min_y, max_x, max_y)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

        # 1. EAR Left Eye
        ear_left = self._calc_ear(pts, LEFT_EYE_HORIZ, LEFT_EYE_VERT1, LEFT_EYE_VERT2)
        # 2. EAR Right Eye
        ear_right = self._calc_ear(pts, RIGHT_EYE_HORIZ, RIGHT_EYE_VERT1, RIGHT_EYE_VERT2)

        ear_avg: Optional[float] = None
        if ear_left is not None and ear_right is not None:
            # If one eye is compressed by profile angle or asymmetric perspective,
            # do not allow an open visible eye to be falsely dragged below the closed threshold.
            max_ear = max(ear_left, ear_right)
            min_ear = min(ear_left, ear_right)
            if max_ear >= 0.20 and (max_ear - min_ear) > 0.08:
                # Strong asymmetry with one eye clearly open: weight toward the open visible eye
                ear_avg = round(0.75 * max_ear + 0.25 * min_ear, 3)
            else:
                ear_avg = round((ear_left + ear_right) / 2.0, 3)
        elif ear_left is not None:
            ear_avg = round(ear_left, 3)
        elif ear_right is not None:
            ear_avg = round(ear_right, 3)

        # 3. MAR Mouth
        mar = self._calc_mar(pts)

        # 4. Face Quality
        quality = self._calc_face_quality(frame_bgr, bbox, w, h)

        # 5. Extract 6 points for head pose
        head_pose_pts = np.array([pts[idx] for idx in HEAD_POSE_INDICES], dtype=np.float64)

        return FaceObservation(
            bbox=bbox,
            center=center,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            mar=mar,
            face_quality=quality,
            head_pose_points_2d=head_pose_pts,
            landmarks_pixel=pts,
            is_valid=True,
        )

    def _calc_ear(
        self,
        pts: np.ndarray,
        horiz: Tuple[int, int],
        vert1: Tuple[int, int],
        vert2: Tuple[int, int]
    ) -> Optional[float]:
        """Calculates Eye Aspect Ratio (EAR) avoiding division by zero."""
        p_h1, p_h2 = pts[horiz[0]], pts[horiz[1]]
        p_v1_top, p_v1_bot = pts[vert1[0]], pts[vert1[1]]
        p_v2_top, p_v2_bot = pts[vert2[0]], pts[vert2[1]]

        dist_h = float(np.linalg.norm(p_h1 - p_h2))
        if dist_h < 1.0:
            return None

        dist_v1 = float(np.linalg.norm(p_v1_top - p_v1_bot))
        dist_v2 = float(np.linalg.norm(p_v2_top - p_v2_bot))

        ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
        if is_finite_number(ear):
            return round(clamp(ear, 0.0, 1.0), 3)
        return None

    def _calc_mar(self, pts: np.ndarray) -> Optional[float]:
        """Calculates Mouth Aspect Ratio (MAR) avoiding division by zero."""
        p_h1, p_h2 = pts[MOUTH_HORIZ[0]], pts[MOUTH_HORIZ[1]]
        dist_h = float(np.linalg.norm(p_h1 - p_h2))
        if dist_h < 1.0:
            return None

        p_v_mid_t, p_v_mid_b = pts[MOUTH_VERT_MID[0]], pts[MOUTH_VERT_MID[1]]
        p_v_l_t, p_v_l_b = pts[MOUTH_VERT_LEFT[0]], pts[MOUTH_VERT_LEFT[1]]
        p_v_r_t, p_v_r_b = pts[MOUTH_VERT_RIGHT[0]], pts[MOUTH_VERT_RIGHT[1]]

        dist_mid = float(np.linalg.norm(p_v_mid_t - p_v_mid_b))
        dist_left = float(np.linalg.norm(p_v_l_t - p_v_l_b))
        dist_right = float(np.linalg.norm(p_v_r_t - p_v_r_b))

        mar = (dist_mid + dist_left + dist_right) / (3.0 * dist_h)
        if is_finite_number(mar):
            return round(clamp(mar, 0.0, 2.0), 3)
        return None

    def _calc_face_quality(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        w: int,
        h: int
    ) -> float:
        """Estimates face quality from ROI sharpness and relative size."""
        x1, y1, x2, y2 = bbox
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Sharpness via Laplacian variance
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
            sharpness_score = clamp(laplacian_var / 250.0, 0.0, 1.0)
        except Exception:
            sharpness_score = 0.5

        # Size ratio
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        size_ratio = face_area / max(1.0, frame_area)
        # Optimal face size is roughly between 2% and 25% of frame
        size_score = clamp(size_ratio / 0.15, 0.1, 1.0)

        composite = 0.6 * sharpness_score + 0.4 * size_score
        return round(clamp(composite, 0.0, 1.0), 2)

"""SolvePnP-based 3D head pose estimation from facial landmark correspondences.

Coordinate Convention:
- Pitch: Rotation around horizontal axis (X-axis).
         Positive (+) = Looking Upwards
         Negative (-) = Looking Downwards (e.g. at desk/notes/phone)
- Yaw:   Rotation around vertical axis (Y-axis).
         Positive (+) = Turned to person's right
         Negative (-) = Turned to person's left
         |Yaw| > threshold = Looking Away
- Roll:  Rotation around optical axis (Z-axis).
         Tilt of the head towards left/right shoulder.
"""

from dataclasses import dataclass
import logging
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from lecture_analyzer.utils.validation import is_finite_number

logger = logging.getLogger("lecture_analyzer.vision.head_pose")

# 3D Anthropometric model landmark coordinates (aligned with camera coordinate system:
# X right, Y down, Z away from camera into the scene)
# Order: Nose tip, Chin, Left eye outer corner, Right eye outer corner, Left mouth corner, Right mouth corner
CANONICAL_3D_FACE_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (index 1) - reference center
    [0.0, 120.0, 20.0],       # Chin (index 152) - below nose (+Y), slightly posterior (+Z)
    [-70.0, -50.0, 40.0],     # Left eye outer corner (index 33) - above nose (-Y), posterior (+Z)
    [70.0, -50.0, 40.0],      # Right eye outer corner (index 263) - above nose (-Y), posterior (+Z)
    [-50.0, 60.0, 30.0],      # Left mouth corner (index 61) - below nose (+Y), posterior (+Z)
    [50.0, 60.0, 30.0],       # Right mouth corner (index 291) - below nose (+Y), posterior (+Z)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class HeadPoseResult:
    """Estimated 3D head pose angles and solver diagnostics."""
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    is_valid: bool
    solver_success: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    error_message: Optional[str] = None


class HeadPoseEstimator:
    """Estimates 3D head orientation angles using OpenCV solvePnP."""

    def __init__(self):
        self.model_points_3d = CANONICAL_3D_FACE_POINTS

    def estimate_pose(
        self,
        image_points_2d: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseResult:
        """Estimates head pose angles given 6 2D facial landmark points.

        Args:
            image_points_2d: (6, 2) array of pixel coordinates corresponding to
                             [Nose, Chin, LeftEye, RightEye, LeftMouth, RightMouth].
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            HeadPoseResult with yaw, pitch, roll in degrees, or invalid result if failed.
        """
        if image_points_2d is None or len(image_points_2d) != 6:
            return HeadPoseResult(
                yaw=None, pitch=None, roll=None,
                is_valid=False, solver_success=False,
                error_message="Requires exactly 6 2D landmark correspondences."
            )

        # Check for non-finite coordinates
        if not np.all(np.isfinite(image_points_2d)):
            return HeadPoseResult(
                yaw=None, pitch=None, roll=None,
                is_valid=False, solver_success=False,
                error_message="Image landmark coordinates contain NaN or Inf."
            )

        # Check for degenerate configuration (e.g. eye distance <= 0)
        left_eye = image_points_2d[2]
        right_eye = image_points_2d[3]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 5.0:
            return HeadPoseResult(
                yaw=None, pitch=None, roll=None,
                is_valid=False, solver_success=False,
                error_message=f"Degenerate face landmark scale (eye distance {eye_dist:.1f}px too small)."
            )

        # Camera intrinsic matrix approximation
        focal_length = float(frame_width)
        center_x = float(frame_width) / 2.0
        center_y = float(frame_height) / 2.0
        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            # Solve PnP using SQPNP (globally convergent, robust against local minima flips)
            # with fallback to ITERATIVE
            success, rvec, tvec = False, None, None
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points_3d,
                    image_points_2d.astype(np.float64),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP
                )
            except Exception:
                success = False

            if not success or rvec is None or tvec is None:
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        self.model_points_3d,
                        image_points_2d.astype(np.float64),
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                except Exception:
                    success = False

            if not success or rvec is None or tvec is None:
                return HeadPoseResult(
                    yaw=None, pitch=None, roll=None,
                    is_valid=False, solver_success=False,
                    error_message="cv2.solvePnP failed to converge."
                )

            # Compute rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles
            # RQDecomp3x3 returns angles in degrees around X, Y, Z axes
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Extract angles in degrees
            raw_pitch = float(angles[0])
            raw_yaw = float(angles[1])
            raw_roll = float(angles[2])

            # Normalize angles to [-180, 180]
            def normalize_angle(deg: float) -> float:
                while deg > 180.0:
                    deg -= 360.0
                while deg < -180.0:
                    deg += 360.0
                return deg

            # Coordinate convention: positive = upward head tilt, negative = downward head tilt
            norm_pitch = normalize_angle(raw_pitch)
            norm_yaw = normalize_angle(raw_yaw)
            norm_roll = normalize_angle(raw_roll)

            # Safeguard against 2-fold planar ambiguity conjugate solutions:
            # A human subject looking at a webcam cannot rotate their head upside-down (|roll| > 90°).
            # When this occurs due to planar depth ambiguity, unwrap to the forward-facing domain.
            if norm_roll > 90.0:
                norm_roll -= 180.0
                norm_pitch = -norm_pitch
                norm_yaw = -norm_yaw
            elif norm_roll < -90.0:
                norm_roll += 180.0
                norm_pitch = -norm_pitch
                norm_yaw = -norm_yaw

            if not (is_finite_number(norm_pitch) and is_finite_number(norm_yaw) and is_finite_number(norm_roll)):
                return HeadPoseResult(
                    yaw=None, pitch=None, roll=None,
                    is_valid=False, solver_success=False,
                    error_message="Decomposed Euler angles non-finite."
                )

            return HeadPoseResult(
                yaw=round(norm_yaw, 2),
                pitch=round(norm_pitch, 2),
                roll=round(norm_roll, 2),
                is_valid=True,
                solver_success=True,
                rvec=rvec,
                tvec=tvec,
            )

        except Exception as e:
            logger.debug(f"Head pose estimation error: {e}")
            return HeadPoseResult(
                yaw=None, pitch=None, roll=None,
                is_valid=False, solver_success=False,
                error_message=str(e)
            )

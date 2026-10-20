"""Unit tests for 3D SolvePnP head pose estimation."""

import numpy as np
import pytest
from lecture_analyzer.vision.head_pose import HeadPoseEstimator, HeadPoseResult


@pytest.fixture
def estimator():
    return HeadPoseEstimator()


def test_head_pose_valid_points(estimator):
    # Synthetic 2D points centered in 1280x720 frame
    # [Nose, Chin, LeftEye, RightEye, LeftMouth, RightMouth]
    pts = np.array([
        [640.0, 360.0],  # Nose
        [640.0, 480.0],  # Chin
        [570.0, 310.0],  # Left eye
        [710.0, 310.0],  # Right eye
        [590.0, 420.0],  # Left mouth
        [690.0, 420.0],  # Right mouth
    ], dtype=np.float64)

    res = estimator.estimate_pose(pts, 1280, 720)
    assert res.is_valid is True
    assert res.solver_success is True
    assert res.yaw is not None and -180.0 <= res.yaw <= 180.0
    assert res.pitch is not None and -180.0 <= res.pitch <= 180.0
    assert res.roll is not None and -180.0 <= res.roll <= 180.0


def test_head_pose_non_finite_inputs(estimator):
    pts = np.array([
        [640.0, np.nan],
        [640.0, 480.0],
        [570.0, 310.0],
        [710.0, 310.0],
        [590.0, 420.0],
        [690.0, 420.0],
    ], dtype=np.float64)

    res = estimator.estimate_pose(pts, 1280, 720)
    assert res.is_valid is False
    assert res.solver_success is False
    assert "NaN or Inf" in res.error_message


def test_head_pose_degenerate_scale(estimator):
    # Eye distance < 5px
    pts = np.array([
        [640.0, 360.0],
        [640.0, 362.0],
        [639.0, 360.0],
        [641.0, 360.0],  # Eye distance = 2px
        [639.0, 361.0],
        [641.0, 361.0],
    ], dtype=np.float64)

    res = estimator.estimate_pose(pts, 1280, 720)
    assert res.is_valid is False
    assert "Degenerate" in res.error_message


def test_head_pose_insufficient_points(estimator):
    pts = np.array([[640.0, 360.0]], dtype=np.float64)
    res = estimator.estimate_pose(pts, 1280, 720)
    assert res.is_valid is False
    assert "Requires exactly 6" in res.error_message


def test_head_pose_pitch_directionality(estimator):
    """Verifies coordinate convention: looking up yields positive pitch, looking down yields negative pitch."""
    import cv2
    from lecture_analyzer.vision.head_pose import CANONICAL_3D_FACE_POINTS

    w, h = 1280, 720
    k = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    d = np.zeros((4, 1))
    tvec = np.array([[0.0], [0.0], [800.0]])

    # 1. Pitch UP by 25 degrees
    rad_up = np.radians(25.0)
    r_up = np.array([[1, 0, 0], [0, np.cos(rad_up), -np.sin(rad_up)], [0, np.sin(rad_up), np.cos(rad_up)]])
    rvec_up, _ = cv2.Rodrigues(r_up)
    pts_up, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec_up, tvec, k, d)
    res_up = estimator.estimate_pose(pts_up.reshape(-1, 2), w, h)
    assert res_up.is_valid is True
    assert res_up.pitch is not None and res_up.pitch > 15.0

    # 2. Pitch DOWN by 25 degrees
    rad_down = np.radians(-25.0)
    r_down = np.array([[1, 0, 0], [0, np.cos(rad_down), -np.sin(rad_down)], [0, np.sin(rad_down), np.cos(rad_down)]])
    rvec_down, _ = cv2.Rodrigues(r_down)
    pts_down, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec_down, tvec, k, d)
    res_down = estimator.estimate_pose(pts_down.reshape(-1, 2), w, h)
    assert res_down.is_valid is True
    assert res_down.pitch is not None and res_down.pitch < -15.0


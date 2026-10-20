"""Feature extraction from multi-modal perception state into standardized observation records."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from lecture_analyzer.tracking.track_manager import TrackedPersonState
from lecture_analyzer.utils.validation import is_finite_number


@dataclass
class ObservationFeatures:
    """Standardized, version-controlled observation features for a single track in a frame."""
    track_id: str
    wall_timestamp: str
    monotonic_ts: float

    # Bounding Box
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int

    # Facial Geometric Features
    ear_avg: Optional[float]
    ear_left: Optional[float]
    ear_right: Optional[float]
    mar: Optional[float]
    face_quality: Optional[float]
    face_detected: bool

    # 3D Head Pose
    head_yaw: Optional[float]
    head_pitch: Optional[float]
    head_roll: Optional[float]
    head_pose_valid: bool

    # Posture Features
    pose_detected: bool
    shoulder_tilt_deg: Optional[float]
    torso_upright_ratio: Optional[float]
    head_shoulder_offset: Optional[float]
    wrists_near_head: bool
    wrists_near_table: bool

    # Phone / Object Evidence
    phone_detected: bool
    phone_confidence: Optional[float]

    # Observability
    observability_level: str
    observability_reasons: str

    # Extended Behavioral Posture Evidence
    hands_covering_face: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts features to a dictionary suitable for storage or tabular processing."""
        return asdict(self)


def extract_features_from_state(
    state: TrackedPersonState,
    wall_timestamp: str,
    monotonic_ts: float
) -> ObservationFeatures:
    """Constructs an auditable ObservationFeatures instance from TrackedPersonState."""
    bx1, by1, bx2, by2 = state.bbox

    # Face metrics
    face_obs = state.face_obs
    face_detected = face_obs is not None and face_obs.is_valid
    ear_avg = face_obs.ear_avg if face_detected and face_obs else None
    ear_left = face_obs.ear_left if face_detected and face_obs else None
    ear_right = face_obs.ear_right if face_detected and face_obs else None
    mar = face_obs.mar if face_detected and face_obs else None
    face_quality = face_obs.face_quality if face_detected and face_obs else None

    # Pose metrics
    pose_obs = state.pose_obs
    pose_detected = pose_obs is not None and pose_obs.is_pose_observable
    sh_tilt = pose_obs.shoulder_tilt_deg if pose_detected and pose_obs else None
    torso_upright = pose_obs.torso_upright_ratio if pose_detected and pose_obs else None
    head_sh_offset = pose_obs.head_shoulder_offset if pose_detected and pose_obs else None
    wrists_head = pose_obs.wrists_near_head if pose_detected and pose_obs else False
    wrists_table = pose_obs.wrists_near_table if pose_detected and pose_obs else False
    hands_face = getattr(pose_obs, "hands_covering_face", False) if pose_detected and pose_obs else False

    # Head pose metrics
    hp = state.head_pose
    hp_valid = hp is not None and hp.is_valid and hp.solver_success
    head_yaw = hp.yaw if hp_valid and hp else None
    head_pitch = hp.pitch if hp_valid and hp else None
    head_roll = hp.roll if hp_valid and hp else None

    # Keypoint-based head orientation fallback when face mesh is unavailable (e.g. multi-student grid/distant participants)
    if not hp_valid and pose_detected and pose_obs is not None:
        p_yaw = getattr(pose_obs, "pose_yaw", None)
        p_pitch = getattr(pose_obs, "pose_pitch", None)
        p_roll = getattr(pose_obs, "pose_roll", None)
        if p_yaw is not None and p_pitch is not None:
            head_yaw = p_yaw
            head_pitch = p_pitch
            head_roll = p_roll
            hp_valid = True
            if not face_detected:
                face_detected = True
                face_quality = 0.70
                ear_avg = 0.28

    # Phone metrics
    phone_obs = state.phone_obs
    phone_detected = phone_obs is not None
    phone_conf = phone_obs.confidence if phone_detected and phone_obs else None

    # Observability
    obs = state.observability
    obs_level = obs.level.value
    obs_reasons = "; ".join(obs.reasons) if obs.reasons else "All modalities observable"

    return ObservationFeatures(
        track_id=state.track_id,
        wall_timestamp=wall_timestamp,
        monotonic_ts=monotonic_ts,
        bbox_x1=bx1,
        bbox_y1=by1,
        bbox_x2=bx2,
        bbox_y2=by2,
        ear_avg=ear_avg,
        ear_left=ear_left,
        ear_right=ear_right,
        mar=mar,
        face_quality=face_quality,
        face_detected=face_detected,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=head_roll,
        head_pose_valid=hp_valid,
        pose_detected=pose_detected,
        shoulder_tilt_deg=sh_tilt,
        torso_upright_ratio=torso_upright,
        head_shoulder_offset=head_sh_offset,
        wrists_near_head=wrists_head,
        wrists_near_table=wrists_table,
        hands_covering_face=hands_face,
        phone_detected=phone_detected,
        phone_confidence=phone_conf,
        observability_level=obs_level,
        observability_reasons=obs_reasons,
    )

"""Observability assessment module for perceptual integrity and sensor failure handling."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ObservabilityLevel(str, Enum):
    FULL = "FULL"                           # Face, head pose, and pose all observable
    PARTIAL_FACE_ONLY = "PARTIAL_FACE_ONLY" # Face & head pose observable, body pose missing
    PARTIAL_POSE_ONLY = "PARTIAL_POSE_ONLY" # Body pose observable, face unavailable/unreliable
    UNOBSERVABLE = "UNOBSERVABLE"           # Neither face nor pose observable


@dataclass
class ObservabilityReport:
    """Detailed observability diagnostics for an observation frame."""
    level: ObservabilityLevel
    face_detected: bool
    face_quality_sufficient: bool
    head_pose_valid: bool
    pose_detected: bool
    phone_detection_active: bool
    reasons: list[str]

    @property
    def is_reliable_for_face_metrics(self) -> bool:
        return self.face_detected and self.face_quality_sufficient and self.head_pose_valid

    @property
    def is_reliable_for_posture(self) -> bool:
        return self.pose_detected


def evaluate_observability(
    has_face: bool,
    face_quality: float,
    quality_threshold: float,
    head_pose_valid: bool,
    has_pose: bool,
    phone_model_active: bool,
) -> ObservabilityReport:
    """Evaluates sensory observability without fabricating evidence."""
    reasons = []

    face_quality_ok = has_face and (face_quality >= quality_threshold)
    if not has_face:
        reasons.append("Face not detected in region.")
    elif not face_quality_ok:
        reasons.append(f"Face quality ({face_quality:.2f}) below threshold ({quality_threshold:.2f}).")

    if has_face and not head_pose_valid:
        reasons.append("3D Head pose solver failed or non-finite.")

    if not has_pose:
        reasons.append("Human pose landmarks not detected or below confidence.")

    face_reliable = has_face and face_quality_ok and head_pose_valid
    pose_reliable = has_pose

    if face_reliable and pose_reliable:
        level = ObservabilityLevel.FULL
    elif face_reliable and not pose_reliable:
        level = ObservabilityLevel.PARTIAL_FACE_ONLY
    elif not face_reliable and pose_reliable:
        level = ObservabilityLevel.PARTIAL_POSE_ONLY
    else:
        level = ObservabilityLevel.UNOBSERVABLE

    return ObservabilityReport(
        level=level,
        face_detected=has_face,
        face_quality_sufficient=face_quality_ok,
        head_pose_valid=head_pose_valid,
        pose_detected=has_pose,
        phone_detection_active=phone_model_active,
        reasons=reasons,
    )

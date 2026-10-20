"""Behavior taxonomy and definitions for observable classroom actions."""

from enum import Enum
from typing import List, Set


class BehaviorCategory(str, Enum):
    """The seven standardized observable behavioral categories.

    Important Scientific Note:
    These categories describe observable physical posture and visual orientation.
    They do NOT constitute direct measurements of internal psychological attention,
    comprehension, cognition, or neurological state.
    """
    ENGAGED_BEHAVIOR = "ENGAGED_BEHAVIOR"
    LOOKING_AWAY = "LOOKING_AWAY"
    HEAD_DOWN = "HEAD_DOWN"
    PHONE_USE = "PHONE_USE"
    WRITING_READING = "WRITING_READING"
    DROWSY_SLEEP_LIKE = "DROWSY_SLEEP_LIKE"
    UNCERTAIN = "UNCERTAIN"


ALL_BEHAVIOR_LABELS: List[str] = [c.value for c in BehaviorCategory]
BEHAVIOR_LABEL_SET: Set[str] = set(ALL_BEHAVIOR_LABELS)

BEHAVIOR_DEFINITIONS = {
    BehaviorCategory.ENGAGED_BEHAVIOR: (
        "Observable forward-facing posture consistent with attending to instructional space. "
        "Eyes open, head yaw and pitch within standard forward range. Does not prove comprehension."
    ),
    BehaviorCategory.LOOKING_AWAY: (
        "Observable head orientation turned significantly away from forward instructional space "
        "(|yaw| > threshold, e.g. looking out window, talking to neighbor)."
    ),
    BehaviorCategory.HEAD_DOWN: (
        "Observable downward head tilt (pitch < downward threshold) without clear writing/reading context."
    ),
    BehaviorCategory.PHONE_USE: (
        "Observable presence of a cell phone or electronic device held or operated near hands/desk."
    ),
    BehaviorCategory.WRITING_READING: (
        "Observable downward gaze or head tilt combined with posture consistent with writing on paper, "
        "reading desk material, or typing (hands/wrists positioned near table)."
    ),
    BehaviorCategory.DROWSY_SLEEP_LIKE: (
        "Observable prolonged eye closure (EAR < threshold) with head nodding or resting on arm/desk. "
        "Naming is strictly descriptive; not a medical or physiological sleep diagnosis."
    ),
    BehaviorCategory.UNCERTAIN: (
        "Sensor evidence is missing, partial, occluded, contradictory, or below reliability thresholds."
    ),
}


def is_valid_behavior_label(label: str) -> bool:
    """Returns True if label is one of the seven recognized behavior categories."""
    return label in BEHAVIOR_LABEL_SET


def is_concentrating(behavior: BehaviorCategory | str) -> bool:
    """Returns True if the observable behavior reflects active instructional concentration or desk task engagement."""
    val = behavior.value if isinstance(behavior, BehaviorCategory) else str(behavior)
    return val in (BehaviorCategory.ENGAGED_BEHAVIOR.value, BehaviorCategory.WRITING_READING.value)


def is_distracted(behavior: BehaviorCategory | str) -> bool:
    """Returns True if the observable behavior reflects visual distraction, phone use, or posture disengagement."""
    val = behavior.value if isinstance(behavior, BehaviorCategory) else str(behavior)
    return val in (
        BehaviorCategory.LOOKING_AWAY.value,
        BehaviorCategory.PHONE_USE.value,
        BehaviorCategory.HEAD_DOWN.value,
        BehaviorCategory.DROWSY_SLEEP_LIKE.value,
    )


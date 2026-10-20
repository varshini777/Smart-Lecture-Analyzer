"""Explainable behavioral rule engine evaluating observable physical evidence."""

from dataclasses import dataclass
import logging
from typing import List, Optional

from lecture_analyzer.behavior.feature_extraction import ObservationFeatures
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.config import AppConfig, FacialConfig, HeadPoseConfig, VisionConfig

logger = logging.getLogger("lecture_analyzer.behavior.rules")


@dataclass
class RuleEvaluationResult:
    """Detailed output from the rule-based behavioral engine."""
    track_id: str
    behavior: BehaviorCategory
    heuristic_score: float         # Explicitly a heuristic score in [0.0, 1.0], NOT a calibrated probability!
    explanation: str
    evidence_used: List[str]
    is_observable: bool
    wall_timestamp: str
    monotonic_ts: float


class BehaviorRuleEngine:
    """Evaluates standardized observations against explainable geometric rules.

    Missing-Feature Safeguards:
    - Missing face or EAR never triggers DROWSY_SLEEP_LIKE.
    - Zero yaw/pitch with head_pose_valid=False is never treated as forward-facing.
    - Missing pose is never assumed to be upright.
    - Low-quality face landmarks (below threshold) cannot trigger confident classifications.
    - Unobservable or ambiguous states strictly yield UNCERTAIN.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.head_cfg: HeadPoseConfig = self.config.head_pose
        self.facial_cfg: FacialConfig = self.config.facial
        self.vision_cfg: VisionConfig = self.config.vision

    def evaluate(self, feats: ObservationFeatures) -> RuleEvaluationResult:
        """Evaluates features using deterministic priority rules."""
        evidence: List[str] = []

        # -------------------------------------------------------------
        # Gate 0: Observability Check
        # -------------------------------------------------------------
        if feats.observability_level == "UNOBSERVABLE":
            return RuleEvaluationResult(
                track_id=feats.track_id,
                behavior=BehaviorCategory.UNCERTAIN,
                heuristic_score=0.20,
                explanation="Perceptual modalities unobservable (neither face nor posture reliably detected).",
                evidence_used=["Observability: UNOBSERVABLE", feats.observability_reasons],
                is_observable=False,
                wall_timestamp=feats.wall_timestamp,
                monotonic_ts=feats.monotonic_ts,
            )

        # -------------------------------------------------------------
        # Rule 1: Phone / Device Use (High Priority Disruption)
        # -------------------------------------------------------------
        if feats.phone_detected and (feats.phone_confidence is not None and feats.phone_confidence >= 0.40):
            evidence.append(f"Phone detected (conf={feats.phone_confidence:.2f})")
            # Only use pitch if head pose is explicitly valid
            if (
                feats.head_pose_valid
                and feats.head_pitch is not None
                and feats.head_pitch < self.head_cfg.pitch_head_down_threshold
            ):
                evidence.append(f"Downward pitch ({feats.head_pitch:.1f}°) towards device")
                score = 0.90
                desc = "Phone detected in immediate vicinity with downward head orientation."
            else:
                score = 0.75
                desc = "Phone detected in participant interaction zone."

            return RuleEvaluationResult(
                track_id=feats.track_id,
                behavior=BehaviorCategory.PHONE_USE,
                heuristic_score=score,
                explanation=desc,
                evidence_used=evidence,
                is_observable=True,
                wall_timestamp=feats.wall_timestamp,
                monotonic_ts=feats.monotonic_ts,
            )

        # -------------------------------------------------------------
        # Rule 2: Drowsy / Sleep-Like Observation (Eye Closure)
        # Safeguards:
        # - Requires detected face with sufficient landmark quality
        # - Requires valid head pose
        # - EAR must be finite and > 0.02 (0.0 implies degenerate/missing landmarks, not closed eyes)
        # - When both eyes are detected, both must be closed to avoid false drowsiness from profile head turns
        # -------------------------------------------------------------
        if (
            feats.face_detected
            and feats.face_quality is not None
            and feats.face_quality >= self.vision_cfg.face_quality_threshold
            and feats.head_pose_valid
            and feats.ear_avg is not None
            and feats.ear_avg > 0.02
        ):
            if feats.ear_avg < self.facial_cfg.ear_closed_threshold:
                evidence.append(f"Low EAR ({feats.ear_avg:.2f} < {self.facial_cfg.ear_closed_threshold:.2f})")
                if getattr(feats, "hands_covering_face", False) or feats.wrists_near_head:
                    evidence.append("Hands covering face or supporting head during eye closure")
                    score = 0.90
                    desc = "Observable eye closure combined with hands covering or supporting face (fatigue/drowsiness)."
                elif feats.head_pitch is not None and feats.head_pitch < self.head_cfg.pitch_head_down_threshold:
                    evidence.append(f"Downward head tilt ({feats.head_pitch:.1f}°)")
                    score = 0.85
                    desc = "Observable eye closure combined with downward head nodding."
                else:
                    score = 0.80
                    desc = "Observable eye closure with upright head position."

                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=score,
                    explanation=desc,
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

        # -------------------------------------------------------------
        # Rule 2b: Hand-to-Face Fatigue / Rest / Face-Covering
        # Educational psychology & classroom observation literature:
        # - Touching or rubbing the face/eyes indicates fatigue, eye strain, boredom, or restlessness.
        # - Propping the head or chin up with a hand/palm indicates boredom, low vigilance, or fatigue.
        # - Covering the face or eyes with hands indicates withdrawal, exhaustion, or complete visual disengagement.
        # Safeguards:
        # - Hand-to-face postures must NEVER qualify as ENGAGED_BEHAVIOR.
        # - If eyes are closed while hand is at face: high-confidence drowsiness (0.90).
        # - If face is occluded or degraded by hands: disengagement / face covered (0.88).
        # - If head is tilted or slumping while supported by hand: fatigue posture (0.85).
        # - General face touching / rubbing / chin resting: hand-to-face fatigue posture (0.82).
        # -------------------------------------------------------------
        has_hands_on_face = feats.wrists_near_head or getattr(feats, "hands_covering_face", False)
        if has_hands_on_face:
            evidence.append("Hand/wrist detected in close proximity to face/head")
            if getattr(feats, "hands_covering_face", False):
                evidence.append("Hand geometry indicates covering, rubbing, or shielding face")

            # Check if eyes are also closed
            if (
                feats.face_detected
                and feats.ear_avg is not None
                and feats.ear_avg < self.facial_cfg.ear_closed_threshold
            ):
                evidence.append(f"Eye closure detected (EAR={feats.ear_avg:.2f}) while resting head on hand")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.90,
                    explanation="Observable eye closure combined with hand supporting or covering the face (fatigue/drowsiness).",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

            # Check if face landmarks are degraded/occluded by the hands
            face_degraded = (
                not feats.face_detected
                or not feats.head_pose_valid
                or (feats.face_quality is not None and feats.face_quality < self.vision_cfg.face_quality_threshold)
            )
            if face_degraded:
                evidence.append("Face features partially or fully occluded by hands covering face")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.88,
                    explanation="Observable disengagement: face covered, closed, or shielded by hands.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

            # Head tilted or slumping while resting on hand
            if (
                feats.head_pose_valid
                and (
                    (feats.head_pitch is not None and feats.head_pitch < self.head_cfg.pitch_head_down_threshold)
                    or (feats.head_roll is not None and abs(feats.head_roll) > 20.0)
                )
            ):
                evidence.append("Head tilted/slumping while supported by hand")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.85,
                    explanation="Observable fatigue posture: chin or head propped up and tilted on hand.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

            # If wrists are on table AND head is pitched downward (reading/writing notes),
            # allow falling through to Rule 4 (Writing/Reading) unless hands are actively covering the face.
            if not getattr(feats, "hands_covering_face", False) and feats.wrists_near_table and feats.head_pose_valid:
                if feats.head_pitch is not None and feats.head_pitch < self.head_cfg.pitch_head_down_threshold:
                    pass  # Let Rule 4 evaluate writing posture
                else:
                    return RuleEvaluationResult(
                        track_id=feats.track_id,
                        behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                        heuristic_score=0.82,
                        explanation="Observable hand-to-face fatigue posture: chin or cheek propped on hand at desk.",
                        evidence_used=evidence,
                        is_observable=True,
                        wall_timestamp=feats.wall_timestamp,
                        monotonic_ts=feats.monotonic_ts,
                    )
            else:
                # General hand-to-face resting or face rubbing
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.82,
                    explanation="Observable hand-to-face fatigue posture: rubbing face or resting cheek/chin on hand.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

        # -------------------------------------------------------------
        # Rule 3: Looking Away (Horizontal Yaw or Upward Pitch Deviation)
        # Safeguards:
        # - Requires head_pose_valid=True
        # - Requires face_detected=True and sufficient face quality
        # -------------------------------------------------------------
        if (
            feats.head_pose_valid
            and feats.face_detected
            and feats.face_quality is not None
            and feats.face_quality >= self.vision_cfg.face_quality_threshold
        ):
            # 3a. Horizontal Yaw Deviation (turned left/right away from front)
            if feats.head_yaw is not None:
                abs_yaw = abs(feats.head_yaw)
                if abs_yaw > self.head_cfg.yaw_looking_away_threshold:
                    turn_dir = "right" if feats.head_yaw > 0 else "left"
                    evidence.append(f"Head yaw {feats.head_yaw:.1f}° exceeds threshold ±{self.head_cfg.yaw_looking_away_threshold:.1f}°")
                    score = min(0.95, 0.70 + (abs_yaw - self.head_cfg.yaw_looking_away_threshold) / 50.0)
                    return RuleEvaluationResult(
                        track_id=feats.track_id,
                        behavior=BehaviorCategory.LOOKING_AWAY,
                        heuristic_score=round(score, 2),
                        explanation=f"Observable head turn to the {turn_dir} away from forward instructional field.",
                        evidence_used=evidence,
                        is_observable=True,
                        wall_timestamp=feats.wall_timestamp,
                        monotonic_ts=feats.monotonic_ts,
                    )

            # 3b. Upward Pitch Deviation (looking up at ceiling / away from instructional focus)
            if feats.head_pitch is not None and feats.head_pitch > self.head_cfg.pitch_head_up_threshold:
                evidence.append(f"Head pitch {feats.head_pitch:.1f}° exceeds upward threshold +{self.head_cfg.pitch_head_up_threshold:.1f}°")
                score = min(0.95, 0.70 + (feats.head_pitch - self.head_cfg.pitch_head_up_threshold) / 40.0)
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.LOOKING_AWAY,
                    heuristic_score=round(score, 2),
                    explanation="Observable upward head orientation directed away from instructional space.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

        # -------------------------------------------------------------
        # Rule 4: Head Down vs. Writing / Reading (Vertical Pitch Deviation)
        # Safeguards:
        # - Requires head_pose_valid=True
        # - Writing/Reading requires pose_detected=True and wrists_near_table=True
        # -------------------------------------------------------------
        if (
            feats.head_pose_valid
            and feats.head_pitch is not None
            and (feats.face_quality is None or feats.face_quality >= self.vision_cfg.face_quality_threshold)
        ):
            if feats.head_pitch < self.head_cfg.pitch_head_down_threshold:
                evidence.append(f"Head pitch {feats.head_pitch:.1f}° below threshold {self.head_cfg.pitch_head_down_threshold:.1f}°")

                # Context check: writing/reading indicators (wrists on table require detected pose)
                if feats.pose_detected and feats.wrists_near_table:
                    evidence.append("Wrists positioned at table/desk level")
                    score = 0.80
                    return RuleEvaluationResult(
                        track_id=feats.track_id,
                        behavior=BehaviorCategory.WRITING_READING,
                        heuristic_score=score,
                        explanation="Observable downward head tilt with hands/wrists positioned near desk (reading/writing posture).",
                        evidence_used=evidence,
                        is_observable=True,
                        wall_timestamp=feats.wall_timestamp,
                        monotonic_ts=feats.monotonic_ts,
                    )
                else:
                    score = 0.82
                    return RuleEvaluationResult(
                        track_id=feats.track_id,
                        behavior=BehaviorCategory.HEAD_DOWN,
                        heuristic_score=score,
                        explanation="Observable downward head pitch without clear writing posture.",
                        evidence_used=evidence,
                        is_observable=True,
                        wall_timestamp=feats.wall_timestamp,
                        monotonic_ts=feats.monotonic_ts,
                    )

        # -------------------------------------------------------------
        # Rule 4b: Extreme Lateral Head Roll (resting on shoulder/hand)
        # -------------------------------------------------------------
        if (
            feats.head_pose_valid
            and feats.head_roll is not None
            and abs(feats.head_roll) > 45.0
            and feats.face_detected
        ):
            tilt_dir = "right" if feats.head_roll > 0 else "left"
            evidence.append(f"Extreme lateral head roll ({feats.head_roll:.1f}°)")
            if feats.ear_avg is not None and feats.ear_avg < self.facial_cfg.ear_closed_threshold:
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.85,
                    explanation=f"Observable extreme head tilt to the {tilt_dir} with eyes closed (resting/slumping posture).",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )
            else:
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.LOOKING_AWAY,
                    heuristic_score=0.75,
                    explanation=f"Observable extreme lateral head tilt to the {tilt_dir} (head resting or turned sideways).",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

        # -------------------------------------------------------------
        # Rule 5: Engaged Forward-Facing Posture
        # Safeguards:
        # - Requires face_detected=True AND sufficient face quality
        # - Requires head_pose_valid=True
        # - Requires eyes open (ear_avg is NOT None, and ear_avg >= threshold)
        # - Natural lateral head tilt (|roll| <= 45°) is permitted
        # - Hands must NOT be at face or covering face (wrists_near_head=False, hands_covering_face=False)
        # - Missing face, missing EAR, or degraded landmarks NEVER qualify as engaged!
        # -------------------------------------------------------------
        if (
            feats.face_detected
            and feats.face_quality is not None
            and feats.face_quality >= self.vision_cfg.face_quality_threshold
            and feats.head_pose_valid
            and feats.head_yaw is not None
            and feats.head_pitch is not None
            and feats.ear_avg is not None
            and feats.ear_avg > 0.02
        ):
            abs_yaw = abs(feats.head_yaw)
            pitch = feats.head_pitch
            yaw_ok = abs_yaw <= self.head_cfg.yaw_looking_away_threshold
            pitch_ok = (self.head_cfg.pitch_head_down_threshold <= pitch <= self.head_cfg.pitch_head_up_threshold)
            ear_ok = feats.ear_avg >= self.facial_cfg.ear_closed_threshold
            roll_ok = (feats.head_roll is None or abs(feats.head_roll) <= 45.0)
            hands_free = not feats.wrists_near_head and not getattr(feats, "hands_covering_face", False)

            if yaw_ok and pitch_ok and ear_ok and roll_ok and hands_free:
                evidence.append(f"Head yaw {feats.head_yaw:.1f}° within forward range")
                evidence.append(f"Head pitch {feats.head_pitch:.1f}° within forward range")
                evidence.append(f"Eyes open (EAR={feats.ear_avg:.2f})")
                if feats.head_roll is not None and abs(feats.head_roll) > 15.0:
                    evidence.append(f"Natural lateral head tilt ({feats.head_roll:.1f}°)")
                if feats.pose_detected and feats.torso_upright_ratio is not None:
                    evidence.append(f"Torso uprightness={feats.torso_upright_ratio:.2f}")

                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.ENGAGED_BEHAVIOR,
                    heuristic_score=0.85,
                    explanation="Observable forward-facing head orientation and open eyes consistent with attending to instruction.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

        # -------------------------------------------------------------
        # Rule 6: Partial Posture Only (Face Missing or Unreliable)
        # -------------------------------------------------------------
        face_unreliable = (
            not feats.face_detected
            or not feats.head_pose_valid
            or (feats.face_quality is not None and feats.face_quality < self.vision_cfg.face_quality_threshold)
        )
        if feats.pose_detected and face_unreliable:
            evidence.append("Body pose visible but face landmarks missing/degraded")

            # 6a: Face covered, closed, or supported with hands
            if feats.wrists_near_head or getattr(feats, "hands_covering_face", False):
                evidence.append("Hand/wrist detected at head/face while face landmarks are occluded")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.DROWSY_SLEEP_LIKE,
                    heuristic_score=0.88,
                    explanation="Observable disengagement: face covered, closed, or supported by hands.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

            # 6b: Wrists positioned at table/desk level (note-taking, coursework, or desk reading)
            if feats.wrists_near_table:
                evidence.append("Hands/wrists positioned at desk/table level (active coursework posture)")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.WRITING_READING,
                    heuristic_score=0.78,
                    explanation="Observable desk-level hand/wrist posture consistent with writing, reading, or active coursework.",
                    evidence_used=evidence,
                    is_observable=True,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )

            # 6c: Heavy slouch or leaning body posture
            if feats.torso_upright_ratio is not None and feats.torso_upright_ratio < 0.65:
                evidence.append(f"Torso leaning heavily (ratio={feats.torso_upright_ratio:.2f})")
                return RuleEvaluationResult(
                    track_id=feats.track_id,
                    behavior=BehaviorCategory.UNCERTAIN,
                    heuristic_score=0.45,
                    explanation="Slouched or leaning body posture detected, but face orientation is unobservable or degraded.",
                    evidence_used=evidence,
                    is_observable=False,
                    wall_timestamp=feats.wall_timestamp,
                    monotonic_ts=feats.monotonic_ts,
                )


        # -------------------------------------------------------------
        # Fallback: Uncertain (Never fabricate engaged behavior)
        # -------------------------------------------------------------
        evidence.append("Perceptual evidence is partial, ambiguous, or below decision thresholds")
        return RuleEvaluationResult(
            track_id=feats.track_id,
            behavior=BehaviorCategory.UNCERTAIN,
            heuristic_score=0.30,
            explanation="Visual evidence insufficient or ambiguous to support an observable behavior category.",
            evidence_used=evidence,
            is_observable=False,
            wall_timestamp=feats.wall_timestamp,
            monotonic_ts=feats.monotonic_ts,
        )

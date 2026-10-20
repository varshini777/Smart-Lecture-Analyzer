"""Human pose estimation and postural metric calculation using YOLOv8-pose."""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from lecture_analyzer.config import ModelsConfig, VisionConfig
from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.validation import clamp, is_finite_number, validate_bbox

logger = logging.getLogger("lecture_analyzer.vision.person_pose")

# COCO Keypoint Index Mapping
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


@dataclass
class PersonPoseObservation:
    """Postural features for a single detected person."""
    bbox: Tuple[int, int, int, int]              # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[np.ndarray]              # (17, 2) in pixel coordinates
    keypoint_conf: Optional[np.ndarray]          # (17,) confidence scores [0, 1]
    shoulder_tilt_deg: Optional[float]           # Degrees deviation from horizontal
    torso_upright_ratio: Optional[float]         # Ratio of vertical spine span to total spine length
    head_shoulder_offset: Optional[float]        # Normalized nose to shoulder distance
    is_pose_observable: bool
    wrists_near_head: bool = False
    wrists_near_table: bool = False
    hands_covering_face: bool = False
    pose_yaw: Optional[float] = None
    pose_pitch: Optional[float] = None
    pose_roll: Optional[float] = None


class PersonPoseEstimator:
    """Manages YOLOv8-pose inference and extracts posture measurements."""

    def __init__(self, models_cfg: Optional[ModelsConfig] = None, vision_cfg: Optional[VisionConfig] = None):
        self.models_cfg = models_cfg or ModelsConfig()
        self.vision_cfg = vision_cfg or VisionConfig()
        self.model = None
        self.is_initialized = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        model_path = resolve_path(self.models_cfg.yolo_pose_path)
        if not model_path.exists():
            logger.warning(
                f"YOLO pose model not found at {model_path}. "
                "Pose estimation will run in fallback (disabled) mode."
            )
            self.is_initialized = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.is_initialized = True
            logger.info(f"YOLOv8-pose initialized successfully from {model_path.name}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO pose model: {e}", exc_info=True)
            self.is_initialized = False

    def process_frame(self, frame_bgr: np.ndarray) -> List[PersonPoseObservation]:
        """Detects people and calculates postural features in frame."""
        if not self.is_initialized or self.model is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        device = self.models_cfg.device
        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.vision_cfg.person_confidence_threshold,
                verbose=False,
                device=device,
            )
        except Exception as e:
            logger.debug(f"YOLO pose prediction error: {e}")
            return []

        if not results or len(results) == 0:
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        observations: List[PersonPoseObservation] = []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        boxes_conf = res.boxes.conf.cpu().numpy()
        kps_xy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        kps_conf = res.keypoints.conf.cpu().numpy() if res.keypoints is not None else None

        for i in range(len(boxes_xyxy)):
            raw_bbox = tuple(boxes_xyxy[i])
            conf = float(boxes_conf[i])

            valid_bbox = validate_bbox(raw_bbox, w, h)
            if valid_bbox is None:
                continue

            person_kps = kps_xy[i] if (kps_xy is not None and i < len(kps_xy)) else None
            person_conf = kps_conf[i] if (kps_conf is not None and i < len(kps_conf)) else None

            obs = self._extract_posture_metrics(valid_bbox, conf, person_kps, person_conf, w, h)
            observations.append(obs)

        return observations

    def _extract_posture_metrics(
        self,
        bbox: Tuple[int, int, int, int],
        conf: float,
        kps: Optional[np.ndarray],
        kps_conf: Optional[np.ndarray],
        w: int,
        h: int,
    ) -> PersonPoseObservation:
        """Extracts posture metrics with defensive missing-keypoint handling."""
        if kps is None or kps_conf is None or len(kps) < 13:
            return PersonPoseObservation(
                bbox=bbox,
                confidence=conf,
                keypoints=None,
                keypoint_conf=None,
                shoulder_tilt_deg=None,
                torso_upright_ratio=None,
                head_shoulder_offset=None,
                is_pose_observable=False,
            )

        # Confidence threshold for individual keypoints
        kp_thresh = 0.35
        l_sh_ok = kps_conf[KP_LEFT_SHOULDER] >= kp_thresh
        r_sh_ok = kps_conf[KP_RIGHT_SHOULDER] >= kp_thresh
        l_hip_ok = kps_conf[KP_LEFT_HIP] >= kp_thresh
        r_hip_ok = kps_conf[KP_RIGHT_HIP] >= kp_thresh
        nose_ok = kps_conf[KP_NOSE] >= kp_thresh

        # 1. Shoulder Tilt Angle
        shoulder_tilt: Optional[float] = None
        shoulder_width: Optional[float] = None
        if l_sh_ok and r_sh_ok:
            l_sh = kps[KP_LEFT_SHOULDER]
            r_sh = kps[KP_RIGHT_SHOULDER]
            dx = float(r_sh[0] - l_sh[0])
            dy = float(r_sh[1] - l_sh[1])
            shoulder_width = float(math.hypot(dx, dy))
            if shoulder_width > 10.0:
                angle_rad = math.atan2(dy, dx)
                deg = math.degrees(angle_rad)
                # Compute absolute deviation from horizontal (0° when level)
                tilt_from_horiz = min(abs(deg), abs(abs(deg) - 180.0))
                sign = 1.0 if dy >= 0 else -1.0
                shoulder_tilt = round(sign * tilt_from_horiz, 2)


        # 2. Torso Uprightness Ratio
        torso_upright: Optional[float] = None
        if l_sh_ok and r_sh_ok and l_hip_ok and r_hip_ok:
            sh_mid = (kps[KP_LEFT_SHOULDER] + kps[KP_RIGHT_SHOULDER]) / 2.0
            hip_mid = (kps[KP_LEFT_HIP] + kps[KP_RIGHT_HIP]) / 2.0
            spine_dx = float(sh_mid[0] - hip_mid[0])
            spine_dy = float(sh_mid[1] - hip_mid[1])  # Should be negative if shoulders are above hips
            spine_len = float(math.hypot(spine_dx, spine_dy))
            if spine_len > 15.0:
                # Vertical component ratio: |dy| / spine_len
                # Upright posture is near 1.0; heavy slouch or lean is < 0.75
                vert_ratio = abs(spine_dy) / spine_len
                torso_upright = round(clamp(vert_ratio, 0.0, 1.0), 3)

        # 3. Head-to-Shoulder Offset
        head_sh_offset: Optional[float] = None
        if nose_ok and l_sh_ok and r_sh_ok and shoulder_width is not None and shoulder_width > 10.0:
            sh_mid = (kps[KP_LEFT_SHOULDER] + kps[KP_RIGHT_SHOULDER]) / 2.0
            nose = kps[KP_NOSE]
            head_dist = float(np.linalg.norm(nose - sh_mid))
            head_sh_offset = round(head_dist / shoulder_width, 3)

        # 4. Wrist and Hand-to-Face positions (Scale-invariant)
        if shoulder_width is not None and shoulder_width > 10.0:
            ref_scale = shoulder_width
        else:
            ref_scale = max(20.0, float(bbox[3] - bbox[1]) * 0.45)

        kp_wrist_thresh = 0.32
        l_wr_ok = kps_conf[KP_LEFT_WRIST] >= kp_wrist_thresh
        r_wr_ok = kps_conf[KP_RIGHT_WRIST] >= kp_wrist_thresh
        wrists_near_head = False
        wrists_near_table = False
        hands_covering_face = False

        # Reference shoulder height
        if l_sh_ok and r_sh_ok:
            sh_y = min(kps[KP_LEFT_SHOULDER][1], kps[KP_RIGHT_SHOULDER][1])
        elif l_sh_ok:
            sh_y = kps[KP_LEFT_SHOULDER][1]
        elif r_sh_ok:
            sh_y = kps[KP_RIGHT_SHOULDER][1]
        elif nose_ok:
            sh_y = kps[KP_NOSE][1] + ref_scale * 0.40
        else:
            sh_y = float(bbox[1] + (bbox[3] - bbox[1]) * 0.40)

        # Reference head center
        if nose_ok:
            head_ref = kps[KP_NOSE]
        elif kps_conf[KP_LEFT_EYE] >= 0.25 and kps_conf[KP_RIGHT_EYE] >= 0.25:
            head_ref = (kps[KP_LEFT_EYE] + kps[KP_RIGHT_EYE]) / 2.0
        else:
            head_ref = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[1] + (bbox[3] - bbox[1]) * 0.20])

        head_dist_thresh = ref_scale * 0.65

        l_near_head = False
        r_near_head = False

        if l_wr_ok:
            l_wr_pt = kps[KP_LEFT_WRIST]
            dist_l = float(np.linalg.norm(l_wr_pt - head_ref))
            if l_wr_pt[1] < sh_y + ref_scale * 0.15 or dist_l < head_dist_thresh:
                l_near_head = True
            elif l_wr_pt[1] > sh_y + ref_scale * 0.25:
                wrists_near_table = True

        if r_wr_ok:
            r_wr_pt = kps[KP_RIGHT_WRIST]
            dist_r = float(np.linalg.norm(r_wr_pt - head_ref))
            if r_wr_pt[1] < sh_y + ref_scale * 0.15 or dist_r < head_dist_thresh:
                r_near_head = True
            elif r_wr_pt[1] > sh_y + ref_scale * 0.25:
                wrists_near_table = True

        # Check if elbows are elevated towards head/shoulders (propping head up with hands)
        l_elb_ok = kps_conf[KP_LEFT_ELBOW] >= 0.35
        r_elb_ok = kps_conf[KP_RIGHT_ELBOW] >= 0.35
        if (l_elb_ok and kps[KP_LEFT_ELBOW][1] < sh_y + ref_scale * 0.10) or (r_elb_ok and kps[KP_RIGHT_ELBOW][1] < sh_y + ref_scale * 0.10):
            if not wrists_near_table:
                wrists_near_head = True

        wrists_near_head = wrists_near_head or l_near_head or r_near_head

        # Hands covering face: both hands near head, or hand in immediate proximity to nose/eyes (< 0.42 * ref_scale)
        face_cov_thresh = ref_scale * 0.42
        if l_near_head and r_near_head and l_wr_ok and r_wr_ok:
            hands_covering_face = True
        elif wrists_near_head and l_wr_ok and float(np.linalg.norm(kps[KP_LEFT_WRIST] - head_ref)) < face_cov_thresh:
            hands_covering_face = True
        elif wrists_near_head and r_wr_ok and float(np.linalg.norm(kps[KP_RIGHT_WRIST] - head_ref)) < face_cov_thresh:
            hands_covering_face = True

        # 5. Pose-derived head orientation (from nose, eyes, ears)
        pose_yaw: Optional[float] = None
        pose_pitch: Optional[float] = None
        pose_roll: Optional[float] = None

        if nose_ok and (kps_conf[KP_LEFT_EYE] >= 0.35) and (kps_conf[KP_RIGHT_EYE] >= 0.35):
            eye_mid = (kps[KP_LEFT_EYE] + kps[KP_RIGHT_EYE]) / 2.0
            eye_dx = float(kps[KP_LEFT_EYE][0] - kps[KP_RIGHT_EYE][0])
            eye_dy = float(kps[KP_LEFT_EYE][1] - kps[KP_RIGHT_EYE][1])
            eye_w = max(5.0, float(math.hypot(eye_dx, eye_dy)))

            # Roll (lateral tilt): angle of eye line relative to horizontal
            pose_roll = round(float(math.degrees(math.atan2(eye_dy, eye_dx))), 1)

            # Yaw: horizontal displacement of nose relative to eye midpoint
            yaw_ratio = float(kps[KP_NOSE][0] - eye_mid[0]) / eye_w
            pose_yaw = round(clamp(yaw_ratio * 90.0, -90.0, 90.0), 1)

            # Pitch: vertical displacement of nose relative to expected neutral position
            pitch_ratio = (float(kps[KP_NOSE][1] - eye_mid[1]) / eye_w) - 0.40
            pose_pitch = round(clamp(pitch_ratio * 70.0, -90.0, 90.0), 1)

        is_observable = (
            (l_sh_ok and r_sh_ok)
            or ((l_sh_ok or r_sh_ok) and conf >= 0.40)
            or (nose_ok and (kps_conf[KP_LEFT_EYE] >= 0.35) and conf >= 0.35)
            or (wrists_near_head and conf >= 0.40)
        )

        return PersonPoseObservation(
            bbox=bbox,
            confidence=conf,
            keypoints=kps,
            keypoint_conf=kps_conf,
            shoulder_tilt_deg=shoulder_tilt,
            torso_upright_ratio=torso_upright,
            head_shoulder_offset=head_sh_offset,
            is_pose_observable=is_observable,
            wrists_near_head=wrists_near_head,
            wrists_near_table=wrists_near_table,
            hands_covering_face=hands_covering_face,
            pose_yaw=pose_yaw,
            pose_pitch=pose_pitch,
            pose_roll=pose_roll,
        )

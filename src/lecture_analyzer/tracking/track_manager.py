"""Multi-person tracking manager associating face, pose, and objects to anonymous IDs."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

from lecture_analyzer.tracking.tracker import AnonymousPersonTracker, Track
from lecture_analyzer.utils.validation import compute_iou
from lecture_analyzer.vision.face_features import FaceObservation
from lecture_analyzer.vision.head_pose import HeadPoseResult
from lecture_analyzer.vision.object_detection import DetectedObject
from lecture_analyzer.vision.observability import ObservabilityReport, evaluate_observability
from lecture_analyzer.vision.person_pose import PersonPoseObservation

logger = logging.getLogger("lecture_analyzer.tracking.track_manager")


@dataclass
class TrackedPersonState:
    """Consolidated state and multi-modal observations associated with an anonymous track."""
    track_id: str                          # e.g. "S001"
    bbox: Tuple[int, int, int, int]
    pose_obs: Optional[PersonPoseObservation]
    face_obs: Optional[FaceObservation]
    head_pose: Optional[HeadPoseResult]
    phone_obs: Optional[DetectedObject]
    observability: ObservabilityReport


class TrackManager:
    """Orchestrates person tracking and conservative multi-modal sensor association."""

    def __init__(self, tracker: Optional[AnonymousPersonTracker] = None, face_quality_thresh: float = 0.35):
        self.tracker = tracker or AnonymousPersonTracker()
        self.face_quality_threshold = face_quality_thresh

    def reset(self) -> None:
        """Resets tracker state at session boundaries."""
        self.tracker.reset()

    def process(
        self,
        person_poses: List[PersonPoseObservation],
        faces: List[FaceObservation],
        head_poses: List[Optional[HeadPoseResult]],
        objects: List[DetectedObject],
        monotonic_ts: float,
        phone_model_active: bool = True,
    ) -> Dict[str, TrackedPersonState]:
        """Performs multi-person tracking and spatial association.

        Returns a dictionary mapping anonymous track_id to TrackedPersonState.
        """
        # 1. Update anonymous person tracks
        person_bboxes = [p.bbox for p in person_poses]
        active_tracks = self.tracker.update(person_bboxes, current_monotonic_ts=monotonic_ts)

        if not active_tracks:
            return {}

        # Map person_poses to tracks by best IoU
        pose_by_track: Dict[str, PersonPoseObservation] = {}
        for track in active_tracks:
            best_iou = 0.0
            best_pose = None
            for p in person_poses:
                iou = compute_iou(track.bbox, p.bbox)
                if iou > best_iou and iou >= 0.2:
                    best_iou = iou
                    best_pose = p
            if best_pose is not None:
                pose_by_track[track.track_id] = best_pose

        # 2. Match faces to person tracks geometrically
        # Face must be located inside the upper half of the person's bounding box
        face_by_track: Dict[str, Tuple[FaceObservation, Optional[HeadPoseResult]]] = {}
        assigned_faces = set()

        for track in active_tracks:
            px1, py1, px2, py2 = track.bbox
            pw = px2 - px1
            ph = py2 - py1
            upper_body_bbox = (px1 - int(pw * 0.1), py1 - int(ph * 0.1), px2 + int(pw * 0.1), py1 + int(ph * 0.65))

            candidate_faces = []
            for f_idx, face in enumerate(faces):
                if f_idx in assigned_faces:
                    continue
                fx1, fy1, fx2, fy2 = face.bbox
                fcx, fcy = face.center

                # Check if face center falls inside upper body region
                if upper_body_bbox[0] <= fcx <= upper_body_bbox[2] and upper_body_bbox[1] <= fcy <= upper_body_bbox[3]:
                    # Also check IoU or overlap with upper body
                    iou = compute_iou(face.bbox, upper_body_bbox)
                    candidate_faces.append((f_idx, iou))

            if len(candidate_faces) == 1:
                f_idx, _ = candidate_faces[0]
                assigned_faces.add(f_idx)
                hp = head_poses[f_idx] if f_idx < len(head_poses) else None
                face_by_track[track.track_id] = (faces[f_idx], hp)
            elif len(candidate_faces) > 1:
                # Ambiguous: multiple faces in one person box, sort by vertical proximity to top of person box
                candidate_faces.sort(key=lambda item: item[1], reverse=True)
                best_idx, _ = candidate_faces[0]
                assigned_faces.add(best_idx)
                hp = head_poses[best_idx] if best_idx < len(head_poses) else None
                face_by_track[track.track_id] = (faces[best_idx], hp)

        # 3. Match detected phones to person tracks
        phone_by_track: Dict[str, DetectedObject] = {}
        phones = [obj for obj in objects if obj.class_name == "cell phone" or obj.class_id == 67]
        assigned_phones = set()

        for track in active_tracks:
            for p_idx, phone in enumerate(phones):
                if p_idx in assigned_phones:
                    continue
                # Check proximity to track
                tx1, ty1, tx2, ty2 = track.bbox
                expanded_track = (tx1 - 30, ty1, tx2 + 30, ty2 + 60)
                iou = compute_iou(phone.bbox, expanded_track)
                phone_cx = (phone.bbox[0] + phone.bbox[2]) / 2.0
                phone_cy = (phone.bbox[1] + phone.bbox[3]) / 2.0

                if iou > 0.05 or (expanded_track[0] <= phone_cx <= expanded_track[2] and expanded_track[1] <= phone_cy <= expanded_track[3]):
                    phone_by_track[track.track_id] = phone
                    assigned_phones.add(p_idx)
                    break

        # 4. Construct final TrackedPersonState with Observability Report
        results: Dict[str, TrackedPersonState] = {}
        for track in active_tracks:
            tid = track.track_id
            pose_obs = pose_by_track.get(tid)
            face_tuple = face_by_track.get(tid)

            face_obs = face_tuple[0] if face_tuple else None
            head_pose = face_tuple[1] if face_tuple else None
            phone_obs = phone_by_track.get(tid)

            # Evaluate observability
            has_face = face_obs is not None and face_obs.is_valid
            face_quality = face_obs.face_quality if face_obs else 0.0
            hp_valid = head_pose is not None and head_pose.is_valid and head_pose.solver_success
            has_pose = pose_obs is not None and pose_obs.is_pose_observable

            # If face mesh is missing/distant but pose provides facial orientation keypoints:
            if not has_face and pose_obs is not None and getattr(pose_obs, "pose_yaw", None) is not None:
                has_face = True
                face_quality = 0.70
                hp_valid = True

            obs_report = evaluate_observability(
                has_face=has_face,
                face_quality=face_quality,
                quality_threshold=self.face_quality_threshold,
                head_pose_valid=hp_valid,
                has_pose=has_pose,
                phone_model_active=phone_model_active,
            )

            results[tid] = TrackedPersonState(
                track_id=tid,
                bbox=track.bbox,
                pose_obs=pose_obs,
                face_obs=face_obs,
                head_pose=head_pose,
                phone_obs=phone_obs,
                observability=obs_report,
            )

        return results

"""End-to-end computer-vision perception pipeline."""

from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Optional
import numpy as np

from lecture_analyzer.config import AppConfig
from lecture_analyzer.tracking.track_manager import TrackManager, TrackedPersonState
from lecture_analyzer.utils.timestamps import monotonic_time
from lecture_analyzer.vision.camera import FramePacket
from lecture_analyzer.vision.face_features import FaceFeatureExtractor, FaceObservation
from lecture_analyzer.vision.head_pose import HeadPoseEstimator, HeadPoseResult
from lecture_analyzer.vision.object_detection import DetectedObject, ObjectDetector
from lecture_analyzer.vision.person_pose import PersonPoseEstimator, PersonPoseObservation

logger = logging.getLogger("lecture_analyzer.vision.pipeline")


@dataclass
class FrameAnalysisResult:
    """Consolidated vision analysis result for a single processed frame."""
    frame_packet: FramePacket
    tracked_persons: Dict[str, TrackedPersonState]
    all_faces: List[FaceObservation]
    all_poses: List[PersonPoseObservation]
    all_objects: List[DetectedObject]
    inference_latency_ms: float
    model_status: Dict[str, bool]


class VisionPipeline:
    """Integrates camera acquisition, pose, face mesh, 3D head pose, object detection, and tracking."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        logger.info("Initializing Vision Pipeline perception components...")
        self.pose_estimator = PersonPoseEstimator(
            models_cfg=self.config.models,
            vision_cfg=self.config.vision,
        )
        self.face_extractor = FaceFeatureExtractor(
            models_cfg=self.config.models,
            facial_cfg=self.config.facial,
        )
        self.head_pose_estimator = HeadPoseEstimator()
        self.object_detector = ObjectDetector(
            models_cfg=self.config.models,
            vision_cfg=self.config.vision,
        )
        self.track_manager = TrackManager(
            face_quality_thresh=self.config.vision.face_quality_threshold
        )

        logger.info("Vision Pipeline components loaded.")

    def reset_tracking(self) -> None:
        """Resets tracking state at session boundaries."""
        self.track_manager.reset()

    def warmup(self, width: int = 640, height: int = 480) -> None:
        """Executes a single dummy inference pass to pre-compile neural graphs and allocate buffers."""
        try:
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            packet = FramePacket(
                frame=dummy,
                frame_index=0,
                wall_timestamp="2026-10-18T00:00:00Z",
                monotonic_ts=monotonic_time(),
                acquisition_latency_ms=0.0,
                width=width,
                height=height,
            )
            self.process_frame(packet)
            self.reset_tracking()
            logger.info("Vision Pipeline warmup completed.")
        except Exception as e:
            logger.warning(f"Vision Pipeline warmup warning: {e}")

    def get_model_status(self) -> Dict[str, bool]:
        """Returns the readiness status of offline vision models."""
        return {
            "yolo_pose": self.pose_estimator.is_initialized,
            "face_landmarker": self.face_extractor.is_initialized,
            "yolo_detect": self.object_detector.is_initialized,
        }

    def process_frame(self, packet: FramePacket) -> FrameAnalysisResult:
        """Processes a single frame packet through all vision modules."""
        t_start = monotonic_time()
        frame_bgr = packet.frame
        h, w = packet.height, packet.width

        # 1. Run human pose estimation
        poses = self.pose_estimator.process_frame(frame_bgr)

        # 2. Run face landmarks
        faces = self.face_extractor.process_frame(frame_bgr)

        # 3. Run 3D head pose for each detected face
        head_poses: List[Optional[HeadPoseResult]] = []
        for face in faces:
            if face.head_pose_points_2d is not None and face.is_valid:
                hp = self.head_pose_estimator.estimate_pose(
                    image_points_2d=face.head_pose_points_2d,
                    frame_width=w,
                    frame_height=h,
                )
                head_poses.append(hp)
            else:
                head_poses.append(None)

        # 4. Run object/phone detection
        objects = self.object_detector.process_frame(frame_bgr)

        # 5. Multi-person tracking and association
        phone_active = self.object_detector.is_initialized
        tracked_persons = self.track_manager.process(
            person_poses=poses,
            faces=faces,
            head_poses=head_poses,
            objects=objects,
            monotonic_ts=packet.monotonic_ts,
            phone_model_active=phone_active,
        )

        t_end = monotonic_time()
        latency_ms = (t_end - t_start) * 1000.0

        return FrameAnalysisResult(
            frame_packet=packet,
            tracked_persons=tracked_persons,
            all_faces=faces,
            all_poses=poses,
            all_objects=objects,
            inference_latency_ms=round(latency_ms, 2),
            model_status=self.get_model_status(),
        )

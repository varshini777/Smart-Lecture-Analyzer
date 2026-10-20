"""Configuration loader and schema validator for Lecture Engagement Pattern Analyzer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigurationError(Exception):
    """Raised when application configuration is invalid or missing."""
    pass


@dataclass
class SystemConfig:
    app_name: str = "Lecture Engagement Pattern Analyzer"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_dir: str = "data/reports"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    backend: str = "DEFAULT"
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0


@dataclass
class StorageConfig:
    database_path: str = "data/database/lecture_analyzer.db"
    recordings_dir: str = "data/recordings"
    exports_dir: str = "data/exports"
    reports_dir: str = "data/reports"
    batch_write_interval_seconds: float = 1.0
    batch_write_max_queue: int = 500


@dataclass
class ModelsConfig:
    face_landmarker_path: str = "models/face_landmarker.task"
    yolo_pose_path: str = "models/yolov8n-pose.pt"
    yolo_detect_path: str = "models/yolov8n.pt"
    device: str = "auto"


@dataclass
class VisionConfig:
    person_confidence_threshold: float = 0.50
    object_confidence_threshold: float = 0.35
    face_min_confidence: float = 0.50
    face_quality_threshold: float = 0.35
    phone_class_id: int = 67
    phone_person_iou_threshold: float = 0.15
    mirror_display: bool = True


@dataclass
class HeadPoseConfig:
    yaw_looking_away_threshold: float = 28.0
    pitch_head_down_threshold: float = -18.0
    pitch_head_up_threshold: float = 25.0
    roll_tilt_threshold: float = 25.0


@dataclass
class FacialConfig:
    ear_closed_threshold: float = 0.20
    mar_open_threshold: float = 0.50
    drowsy_closed_duration_seconds: float = 2.0
    head_down_duration_seconds: float = 1.5


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.30
    track_retention_seconds: float = 2.5
    min_hits_to_confirm: int = 2


@dataclass
class BehaviorConfig:
    smoothing_window_seconds: float = 1.5
    min_observable_frames_fraction: float = 0.60
    enable_smoothing: bool = True


@dataclass
class MLConfig:
    enabled: bool = False
    active_model_path: str = "models/active_behavior_classifier.joblib"
    active_model_metadata: str = "models/active_model_metadata.json"


@dataclass
class PrivacyConfig:
    allow_raw_video_recording: bool = False
    video_codec: str = "mp4v"
    retain_unconsented_recordings: bool = False


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    facial: FacialConfig = field(default_factory=FacialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def validate_config(cfg: AppConfig) -> None:
    """Performs rigorous range and sanity checking on application configuration."""
    if cfg.camera.width <= 0 or cfg.camera.height <= 0:
        raise ConfigurationError(
            f"Invalid camera resolution: {cfg.camera.width}x{cfg.camera.height}. Dimensions must be positive."
        )

    if not (0.0 <= cfg.vision.person_confidence_threshold <= 1.0):
        raise ConfigurationError("person_confidence_threshold must be between 0.0 and 1.0")

    if not (0.0 <= cfg.vision.object_confidence_threshold <= 1.0):
        raise ConfigurationError("object_confidence_threshold must be between 0.0 and 1.0")

    if not (0.0 < cfg.facial.ear_closed_threshold < 1.0):
        raise ConfigurationError("ear_closed_threshold must be between 0.0 and 1.0")

    if not (0.0 < cfg.facial.mar_open_threshold < 2.0):
        raise ConfigurationError("mar_open_threshold must be between 0.0 and 2.0")

    if cfg.head_pose.yaw_looking_away_threshold <= 0:
        raise ConfigurationError("yaw_looking_away_threshold must be a positive angle in degrees")

    if cfg.head_pose.pitch_head_down_threshold >= 0:
        raise ConfigurationError(
            "pitch_head_down_threshold must be negative for downward pitch (degrees)"
        )

    if cfg.behavior.smoothing_window_seconds <= 0:
        raise ConfigurationError("smoothing_window_seconds must be positive")

    if cfg.tracking.track_retention_seconds <= 0:
        raise ConfigurationError("track_retention_seconds must be positive")


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads and validates configuration from a YAML file or returns validated defaults.

    Args:
        config_path: Path to config.yaml. If None, tries 'configs/config.yaml'.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigurationError: If the file is malformed or parameters violate constraints.
    """
    if config_path is None:
        default_file = Path("configs/config.yaml")
        if default_file.exists():
            config_path = str(default_file)

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file {config_path}: {e}")

        system_data = data.get("system", {})
        camera_data = data.get("camera", {})
        storage_data = data.get("storage", {})
        models_data = data.get("models", {})
        vision_data = data.get("vision", {})
        head_pose_data = data.get("head_pose", {})
        facial_data = data.get("facial", {})
        tracking_data = data.get("tracking", {})
        behavior_data = data.get("behavior", {})
        ml_data = data.get("ml", {})
        privacy_data = data.get("privacy", {})

        cfg = AppConfig(
            system=SystemConfig(**system_data),
            camera=CameraConfig(**camera_data),
            storage=StorageConfig(**storage_data),
            models=ModelsConfig(**models_data),
            vision=VisionConfig(**vision_data),
            head_pose=HeadPoseConfig(**head_pose_data),
            facial=FacialConfig(**facial_data),
            tracking=TrackingConfig(**tracking_data),
            behavior=BehaviorConfig(**behavior_data),
            ml=MLConfig(**ml_data),
            privacy=PrivacyConfig(**privacy_data),
        )
    else:
        cfg = AppConfig()

    validate_config(cfg)
    return cfg

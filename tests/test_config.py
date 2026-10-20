"""Unit tests for configuration loading and validation."""

import pytest
from lecture_analyzer.config import AppConfig, ConfigurationError, load_config, validate_config


def test_default_config_is_valid():
    cfg = AppConfig()
    validate_config(cfg)
    assert cfg.system.app_name == "Lecture Engagement Pattern Analyzer"
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.facial.ear_closed_threshold == 0.20


def test_load_config_from_file():
    cfg = load_config("configs/config.yaml")
    assert cfg is not None
    assert cfg.head_pose.yaw_looking_away_threshold > 0
    assert cfg.head_pose.pitch_head_down_threshold < 0


def test_invalid_resolution_raises():
    cfg = AppConfig()
    cfg.camera.width = -1
    with pytest.raises(ConfigurationError, match="Invalid camera resolution"):
        validate_config(cfg)


def test_invalid_confidence_threshold_raises():
    cfg = AppConfig()
    cfg.vision.person_confidence_threshold = 1.5
    with pytest.raises(ConfigurationError, match="person_confidence_threshold"):
        validate_config(cfg)


def test_invalid_ear_threshold_raises():
    cfg = AppConfig()
    cfg.facial.ear_closed_threshold = -0.1
    with pytest.raises(ConfigurationError, match="ear_closed_threshold"):
        validate_config(cfg)


def test_invalid_pitch_down_must_be_negative():
    cfg = AppConfig()
    cfg.head_pose.pitch_head_down_threshold = 15.0  # Must be negative for downward pitch
    with pytest.raises(ConfigurationError, match="pitch_head_down_threshold"):
        validate_config(cfg)

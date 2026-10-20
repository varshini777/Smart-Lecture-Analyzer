"""Unit and regression tests for VideoAttentionAuditService and Audit Report generation."""

import json
import os
import tempfile
import pytest
import numpy as np

from lecture_analyzer.analytics.video_audit_service import (
    StudentAttentionSummary,
    TimeIntervalSummary,
    VideoAttentionAuditService,
    VideoAuditResult,
)
from lecture_analyzer.behavior.labels import BehaviorCategory
from lecture_analyzer.config import AppConfig


@pytest.fixture
def audit_service():
    config = AppConfig()
    return VideoAttentionAuditService(config)


def test_video_audit_service_initialization(audit_service):
    assert audit_service is not None
    assert audit_service.config is not None


def test_video_audit_missing_file_raises(audit_service):
    with pytest.raises(FileNotFoundError):
        audit_service.audit_video("nonexistent_video_path_xyz.mp4")


def test_video_audit_aggregation_and_report_generation(audit_service):
    """Verifies that mathematical aggregation and markdown generation correctly reflect concentration and distraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock observations for 2 students over 30 frames (1.0s at 30fps)
        # S001: 25 concentrating frames (ENGAGED_BEHAVIOR), 5 looking away frames
        # S002: 10 concentrating frames (WRITING_READING), 20 phone use frames
        track_obs = {
            "S001": [
                {
                    "frame_index": i + 1,
                    "elapsed_s": i / 30.0,
                    "behavior": BehaviorCategory.ENGAGED_BEHAVIOR if i < 25 else BehaviorCategory.LOOKING_AWAY,
                    "raw_behavior": BehaviorCategory.ENGAGED_BEHAVIOR,
                    "is_concentrating": (i < 25),
                    "is_distracted": (i >= 25),
                    "is_uncertain": False,
                    "face_quality": 0.85,
                    "head_pitch": 2.0,
                    "head_yaw": 0.0,
                    "ear_avg": 0.28,
                    "bbox": (100, 100, 200, 300),
                }
                for i in range(30)
            ],
            "S002": [
                {
                    "frame_index": i + 1,
                    "elapsed_s": i / 30.0,
                    "behavior": BehaviorCategory.WRITING_READING if i < 10 else BehaviorCategory.PHONE_USE,
                    "raw_behavior": BehaviorCategory.WRITING_READING,
                    "is_concentrating": (i < 10),
                    "is_distracted": (i >= 10),
                    "is_uncertain": False,
                    "face_quality": 0.70,
                    "head_pitch": -22.0,
                    "head_yaw": 0.0,
                    "ear_avg": 0.25,
                    "bbox": (300, 100, 400, 300),
                }
                for i in range(30)
            ],
        }

        frame_metrics = {
            i + 1: {
                "concentrating": (1 if i < 25 else 0) + (1 if i < 10 else 0),
                "distracted": (1 if i >= 25 else 0) + (1 if i >= 10 else 0),
                "uncertain": 0,
                "total": 2,
                "elapsed_s": i / 30.0,
            }
            for i in range(30)
        }

        res = audit_service._aggregate_results(
            video_path="test_video.mp4",
            video_basename="test_video.mp4",
            duration_s=1.0,
            total_frames=30,
            processed_frames=30,
            frame_stride=1,
            fps=30.0,
            resolution="1280x720",
            total_audit_time=0.5,
            overall_analysis_fps=60.0,
            track_observations=track_obs,
            frame_metrics=frame_metrics,
            interval_seconds=1.0,
            annotated_video_path=None,
            reports_dir=tmpdir,
        )

        assert res.total_unique_students == 2
        assert res.average_students_per_frame == 2.0
        # Total student observations = 30 + 30 = 60
        # Concentrated = 25 (S001) + 10 (S002) = 35 -> 35/60 = 58.3%
        # Distracted = 5 (S001) + 20 (S002) = 25 -> 25/60 = 41.7%
        assert 58.0 <= res.overall_concentration_percentage <= 59.0
        assert 41.0 <= res.overall_distraction_percentage <= 42.0

        # Check per-student summaries
        s1 = next(s for s in res.students if s.track_id == "S001")
        assert s1.dominant_behavior == "ENGAGED_BEHAVIOR"
        assert s1.concentrating_percentage == round((25 / 30) * 100.0, 1)  # 83.3%
        assert s1.attention_rating == "HIGH_ATTENTION"

        s2 = next(s for s in res.students if s.track_id == "S002")
        assert s2.dominant_behavior == "PHONE_USE"
        assert s2.concentrating_percentage == round((10 / 30) * 100.0, 1)  # 33.3%
        assert s2.attention_rating == "INATTENTIVE"

        # Check output report files
        assert res.report_markdown_path is not None and os.path.exists(res.report_markdown_path)
        assert res.report_json_path is not None and os.path.exists(res.report_json_path)

        with open(res.report_markdown_path, "r", encoding="utf-8") as f:
            md = f.read()
            assert "# Classroom Attention & Behavioral Audit Report" in md
            assert "S001" in md
            assert "S002" in md
            assert "HIGH_ATTENTION" in md
            assert "INATTENTIVE" in md

        with open(res.report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_unique_students"] == 2
            assert len(data["students"]) == 2

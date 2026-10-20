"""Unit tests for independent human annotation validation and coverage calculation."""

import pytest
from lecture_analyzer.annotation.annotation_validation import compute_annotation_coverage, validate_annotation
from lecture_analyzer.storage.repositories import AnnotationRecord


def test_validate_annotation_success():
    res = validate_annotation(
        start_seconds=1.0,
        end_seconds=5.0,
        behavior_label="ENGAGED_BEHAVIOR",
        annotator_id="HUMAN_ANN_1",
        session_duration=10.0,
    )
    assert res.is_valid is True
    assert res.error_message is None


def test_validate_annotation_invalid_times():
    res = validate_annotation(
        start_seconds=5.0,
        end_seconds=2.0,  # start > end
        behavior_label="ENGAGED_BEHAVIOR",
        annotator_id="HUMAN_ANN_1",
    )
    assert res.is_valid is False
    assert "Invalid time bounds" in res.error_message


def test_validate_annotation_invalid_label():
    res = validate_annotation(
        start_seconds=1.0,
        end_seconds=5.0,
        behavior_label="STUDYING_INTENTLY",  # Not in 7 standard categories
        annotator_id="HUMAN_ANN_1",
    )
    assert res.is_valid is False
    assert "Unrecognized behavior label" in res.error_message


def test_validate_annotation_empty_annotator():
    res = validate_annotation(
        start_seconds=1.0,
        end_seconds=5.0,
        behavior_label="ENGAGED_BEHAVIOR",
        annotator_id="   ",
    )
    assert res.is_valid is False
    assert "Annotator ID cannot be empty" in res.error_message


def test_compute_annotation_coverage_and_overlaps():
    annots = [
        AnnotationRecord(
            id=1, session_id="S1", track_id="S001",
            start_seconds=0.0, end_seconds=10.0,
            behavior_label="ENGAGED_BEHAVIOR", annotator_id="A1",
            confidence_level="HIGH", notes=None, created_at="2026-10-18"
        ),
        AnnotationRecord(
            id=2, session_id="S1", track_id="S001",
            start_seconds=8.0, end_seconds=15.0,  # Overlaps with #1 between 8.0 and 10.0
            behavior_label="LOOKING_AWAY", annotator_id="A1",
            confidence_level="HIGH", notes=None, created_at="2026-10-18"
        ),
    ]

    report = compute_annotation_coverage(annots, session_duration_seconds=20.0)
    assert report.total_session_seconds == 20.0
    assert report.total_annotated_seconds == 15.0  # Union of [0, 10] and [8, 15] is [0, 15]
    assert report.coverage_fraction == 0.75
    assert report.has_overlaps is True
    assert len(report.overlapping_intervals) == 1
    assert report.overlapping_intervals[0] == (0.0, 10.0, 8.0, 15.0)

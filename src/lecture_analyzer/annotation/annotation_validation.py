"""Validation and integrity checking for independent human annotations."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from lecture_analyzer.behavior.labels import is_valid_behavior_label
from lecture_analyzer.storage.repositories import AnnotationRecord


@dataclass
class AnnotationValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None


@dataclass
class CoverageReport:
    total_session_seconds: float
    total_annotated_seconds: float
    coverage_fraction: float
    has_overlaps: bool
    overlapping_intervals: List[Tuple[float, float, float, float]]  # (start1, end1, start2, end2)
    label_durations: dict[str, float]


def validate_annotation(
    start_seconds: float,
    end_seconds: float,
    behavior_label: str,
    annotator_id: str,
    session_duration: Optional[float] = None
) -> AnnotationValidationResult:
    """Validates an individual annotation entry before database insertion."""
    warnings = []

    if start_seconds < 0.0 or end_seconds <= start_seconds:
        return AnnotationValidationResult(
            is_valid=False,
            error_message=f"Invalid time bounds: start_seconds ({start_seconds:.2f}) must be >= 0 and < end_seconds ({end_seconds:.2f}).",
            warnings=warnings,
        )

    if not is_valid_behavior_label(behavior_label):
        return AnnotationValidationResult(
            is_valid=False,
            error_message=f"Unrecognized behavior label: '{behavior_label}'. Must be one of the seven official taxonomy categories.",
            warnings=warnings,
        )

    if not annotator_id or not annotator_id.strip():
        return AnnotationValidationResult(
            is_valid=False,
            error_message="Annotator ID cannot be empty. Independent human attribution is required.",
            warnings=warnings,
        )

    if session_duration is not None and end_seconds > session_duration + 1.0:
        warnings.append(
            f"Annotation end ({end_seconds:.1f}s) exceeds recorded session duration ({session_duration:.1f}s)."
        )

    return AnnotationValidationResult(is_valid=True, warnings=warnings)


def compute_annotation_coverage(
    annotations: List[AnnotationRecord],
    session_duration_seconds: float
) -> CoverageReport:
    """Calculates temporal coverage, label breakdown, and checks for overlapping intervals."""
    if not annotations or session_duration_seconds <= 0.0:
        return CoverageReport(
            total_session_seconds=max(0.0, session_duration_seconds),
            total_annotated_seconds=0.0,
            coverage_fraction=0.0,
            has_overlaps=False,
            overlapping_intervals=[],
            label_durations={},
        )

    # Sort annotations by start time
    sorted_annots = sorted(annotations, key=lambda a: a.start_seconds)
    overlaps: List[Tuple[float, float, float, float]] = []
    label_durations: dict[str, float] = {}

    for i in range(len(sorted_annots)):
        cur = sorted_annots[i]
        dur = max(0.0, cur.end_seconds - cur.start_seconds)
        label_durations[cur.behavior_label] = label_durations.get(cur.behavior_label, 0.0) + dur

        # Check overlap with next
        for j in range(i + 1, len(sorted_annots)):
            nxt = sorted_annots[j]
            # If for same track (or both unassigned)
            if cur.track_id == nxt.track_id:
                if cur.end_seconds > nxt.start_seconds:
                    overlaps.append((cur.start_seconds, cur.end_seconds, nxt.start_seconds, nxt.end_seconds))

    # Calculate union of annotated intervals
    intervals = [(a.start_seconds, a.end_seconds) for a in sorted_annots]
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            prev_start, prev_end = merged[-1]
            if interval[0] <= prev_end:
                merged[-1] = (prev_start, max(prev_end, interval[1]))
            else:
                merged.append(interval)

    total_annotated_s = sum(end - start for start, end in merged)
    coverage_fraction = min(1.0, total_annotated_s / session_duration_seconds)

    return CoverageReport(
        total_session_seconds=round(session_duration_seconds, 2),
        total_annotated_seconds=round(total_annotated_s, 2),
        coverage_fraction=round(coverage_fraction, 3),
        has_overlaps=len(overlaps) > 0,
        overlapping_intervals=overlaps,
        label_durations={k: round(v, 2) for k, v in label_durations.items()},
    )

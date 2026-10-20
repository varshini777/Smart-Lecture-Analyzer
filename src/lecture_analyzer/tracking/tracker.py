"""Anonymous session-local multi-person tracker.

Strict Privacy & Integrity Guarantee:
- Anonymous IDs are generated purely per-session as 'S001', 'S002', etc.
- No facial recognition, biometric descriptors, or face embeddings are ever computed or stored.
- Tracking state is completely wiped at session boundaries.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from lecture_analyzer.config import TrackingConfig
from lecture_analyzer.utils.timestamps import monotonic_time
from lecture_analyzer.utils.validation import compute_iou

logger = logging.getLogger("lecture_analyzer.tracking.tracker")


@dataclass
class Track:
    """Represents a session-local anonymous track."""
    track_id: str                          # e.g. "S001"
    numeric_id: int
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2)
    hits: int = 1
    misses: int = 0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    is_confirmed: bool = False


class AnonymousPersonTracker:
    """IoU-based anonymous person tracker maintaining spatial continuity."""

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    def reset(self) -> None:
        """Resets all tracks and resets anonymous ID numbering to S001."""
        self.tracks.clear()
        self._next_id = 1
        logger.info("Tracker state reset: all session tracks cleared.")

    def update(
        self,
        detected_bboxes: List[Tuple[int, int, int, int]],
        current_monotonic_ts: Optional[float] = None
    ) -> List[Track]:
        """Updates tracks with newly observed person bounding boxes.

        Args:
            detected_bboxes: List of (x1, y1, x2, y2) tuples for detected persons.
            current_monotonic_ts: Monotonic time in seconds.

        Returns:
            List of active confirmed Tracks.
        """
        now = current_monotonic_ts if current_monotonic_ts is not None else monotonic_time()

        if not self.tracks and not detected_bboxes:
            return []

        # If no existing tracks, initialize new tracks for all detections
        if not self.tracks:
            for bbox in detected_bboxes:
                self._create_track(bbox, now)
            return [t for t in self.tracks.values() if t.is_confirmed]

        # If no detections this frame, increment misses for all existing tracks
        if not detected_bboxes:
            self._handle_missed_tracks(now)
            return [t for t in self.tracks.values() if t.is_confirmed]

        # Compute IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detected_bboxes)

        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            track_box = self.tracks[tid].bbox
            for j, det_box in enumerate(detected_bboxes):
                iou_matrix[i, j] = compute_iou(track_box, det_box)

        # Greedy bipartite matching
        matched_tracks = set()
        matched_dets = set()

        # Find best matches above threshold
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = float(np.max(iou_matrix))
            if max_iou < self.config.iou_threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tid = track_ids[i]

            # Update matched track
            track = self.tracks[tid]
            track.bbox = detected_bboxes[j]
            track.hits += 1
            track.misses = 0
            track.last_seen_ts = now
            if track.hits >= self.config.min_hits_to_confirm:
                track.is_confirmed = True

            matched_tracks.add(tid)
            matched_dets.add(j)

            # Invalidate matched row and column
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Create new tracks for unmatched detections
        for j, det_box in enumerate(detected_bboxes):
            if j not in matched_dets:
                self._create_track(det_box, now)

        # Handle unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.misses += 1

        self._cleanup_stale_tracks(now)
        return [t for t in self.tracks.values() if t.is_confirmed]

    def _create_track(self, bbox: Tuple[int, int, int, int], now: float) -> Track:
        nid = self._next_id
        self._next_id += 1
        tid_str = f"S{nid:03d}"

        # If min_hits is 1, instantly confirm
        is_conf = (self.config.min_hits_to_confirm <= 1)
        track = Track(
            track_id=tid_str,
            numeric_id=nid,
            bbox=bbox,
            hits=1,
            misses=0,
            first_seen_ts=now,
            last_seen_ts=now,
            is_confirmed=is_conf,
        )
        self.tracks[nid] = track
        return track

    def _handle_missed_tracks(self, now: float) -> None:
        for track in self.tracks.values():
            track.misses += 1
        self._cleanup_stale_tracks(now)

    def _cleanup_stale_tracks(self, now: float) -> None:
        to_delete = []
        for nid, track in self.tracks.items():
            age_since_seen = now - track.last_seen_ts
            if age_since_seen > self.config.track_retention_seconds:
                to_delete.append(nid)

        for nid in to_delete:
            del self.tracks[nid]

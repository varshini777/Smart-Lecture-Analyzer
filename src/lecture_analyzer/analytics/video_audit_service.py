"""Video batch attention analyzer and comprehensive classroom audit report generator.

Processes 2-5 minute recorded lecture videos to evaluate:
- Total student count and individual student tracking.
- Overall classroom concentration vs. distraction metrics.
- Per-student attention time, distraction time, dominant behaviors, and ratings.
- Time-sliced engagement timeline (e.g. 10-15s intervals).
- Exportable formal Markdown Audit Report, JSON summary, and optional annotated video.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lecture_analyzer.behavior.feature_extraction import extract_features_from_state
from lecture_analyzer.behavior.labels import (
    ALL_BEHAVIOR_LABELS,
    BehaviorCategory,
    is_concentrating,
    is_distracted,
)
from lecture_analyzer.behavior.rules import BehaviorRuleEngine
from lecture_analyzer.behavior.temporal_smoothing import TemporalBehaviorSmoother
from lecture_analyzer.config import AppConfig
from lecture_analyzer.utils.timestamps import now_utc_iso
from lecture_analyzer.vision.camera import FramePacket
from lecture_analyzer.vision.pipeline import VisionPipeline

logger = logging.getLogger("lecture_analyzer.analytics.video_audit_service")


@dataclass
class StudentAttentionSummary:
    """Consolidated attention and behavioral metrics for a single tracked student."""
    track_id: str
    total_visible_frames: int
    total_visible_seconds: float
    visibility_percentage: float
    concentrating_frames: int
    concentrating_seconds: float
    concentrating_percentage: float
    distracted_frames: int
    distracted_seconds: float
    distracted_percentage: float
    uncertain_frames: int
    uncertain_seconds: float
    uncertain_percentage: float
    dominant_behavior: str
    attention_rating: str  # "HIGH_ATTENTION", "MODERATE_ATTENTION", "INATTENTIVE"
    behavior_breakdown: Dict[str, float]  # percentage for each of the 7 observable behaviors
    distraction_breakdown: Dict[str, int]  # counts of looking_away, phone, head_down, drowsy
    distraction_event_count: int
    mean_face_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeIntervalSummary:
    """Aggregated attention statistics across a discrete time window (e.g. 15s)."""
    interval_index: int
    start_seconds: float
    end_seconds: float
    time_label: str
    students_present: int
    students_concentrating: int
    students_distracted: int
    students_uncertain: int
    attention_percentage: float
    attention_status: str  # "High Concentration", "Moderate", "Attention Lapse"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAuditResult:
    """Comprehensive output of an end-to-end video attention audit."""
    video_path: str
    video_filename: str
    analysis_timestamp: str
    video_duration_seconds: float
    total_video_frames: int
    processed_frames: int
    frame_stride: int
    analysis_duration_seconds: float
    analysis_fps: float
    fps: float
    resolution: str
    total_unique_students: int
    average_students_per_frame: float
    overall_concentration_percentage: float
    overall_distraction_percentage: float
    overall_uncertain_percentage: float
    behavior_distribution: Dict[str, float]
    students: List[StudentAttentionSummary]
    timeline: List[TimeIntervalSummary]
    annotated_video_path: Optional[str] = None
    report_markdown_path: Optional[str] = None
    report_json_path: Optional[str] = None
    summary_markdown: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["students"] = [s.to_dict() for s in self.students]
        data["timeline"] = [t.to_dict() for t in self.timeline]
        return data


class VideoAttentionAuditService:
    """Executes exhaustive attention audits on recorded multi-student lecture videos."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

    def audit_video(
        self,
        video_path: str,
        generate_annotated_video: bool = True,
        output_dir: Optional[str] = None,
        frame_stride: int = 1,
        interval_seconds: float = 15.0,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> VideoAuditResult:
        """Processes a video file from start to finish and produces an exhaustive audit result.

        Args:
            video_path: Path to recorded video file (MP4, AVI, MOV).
            generate_annotated_video: Whether to render and save an overlay video.
            output_dir: Directory where reports and annotated video are stored.
            frame_stride: Sample stride (1 = every frame; 2 = every 2nd frame, etc.).
            interval_seconds: Window size in seconds for timeline interval analysis.
            progress_callback: Callable(current_frame, total_frames, current_fps, status_msg).

        Returns:
            VideoAuditResult containing all aggregated metrics, student summaries, and report paths.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        reports_dir = output_dir or self.config.storage.reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        video_basename = os.path.basename(video_path)
        video_stem = os.path.splitext(video_basename)[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = orig_fps if (orig_fps and orig_fps > 0.0) else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if total_frames > 0 else 0.0

        logger.info(
            f"Starting attention audit on {video_basename}: {width}x{height}, "
            f"{total_frames} frames ({duration_s:.1f}s), stride={frame_stride}"
        )

        pipeline = VisionPipeline(self.config)
        pipeline.warmup(width=min(width, 1280), height=min(height, 720))
        rule_engine = BehaviorRuleEngine(self.config)
        smoother = TemporalBehaviorSmoother(max_gap_seconds=2.0)

        # Optional annotated video writer
        annotated_video_path: Optional[str] = None
        writer: Optional[cv2.VideoWriter] = None

        if generate_annotated_video:
            annotated_video_path = os.path.join(reports_dir, f"ANNOTATED_{video_basename}")
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            effective_out_fps = fps / frame_stride
            writer = cv2.VideoWriter(annotated_video_path, fourcc, effective_out_fps, (width, height))
            if not writer.isOpened():
                # Fallback to XVID
                annotated_video_path = os.path.join(reports_dir, f"ANNOTATED_{video_stem}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(annotated_video_path, fourcc, effective_out_fps, (width, height))
                if not writer.isOpened():
                    logger.warning("Failed to initialize VideoWriter. Skipping annotated video output.")
                    writer = None
                    annotated_video_path = None

        # Data structures for accumulation
        # track_id -> list of frame observation records
        track_observations: Dict[str, List[Dict[str, Any]]] = {}
        # frame_idx -> { 'concentrating': count, 'distracted': count, 'uncertain': count, 'total': count }
        frame_metrics: Dict[int, Dict[str, int]] = {}

        t_audit_start = time.time()
        processed_count = 0
        current_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame_idx += 1
            if (current_frame_idx - 1) % frame_stride != 0:
                continue

            processed_count += 1
            frame_elapsed_s = (current_frame_idx - 1) / fps
            iso_wall = now_utc_iso()

            pkt = FramePacket(
                frame=frame,
                frame_index=current_frame_idx,
                wall_timestamp=iso_wall,
                monotonic_ts=frame_elapsed_s,
                acquisition_latency_ms=0.0,
                width=width,
                height=height,
            )

            # Vision pipeline inference
            frame_result = pipeline.process_frame(pkt)

            frame_conc = 0
            frame_dist = 0
            frame_unc = 0
            visible_students_in_frame = len(frame_result.tracked_persons)

            # Annotate frame copy if writing
            annotated_frame = frame.copy() if writer is not None else None

            for tid, state in frame_result.tracked_persons.items():
                feats = extract_features_from_state(state, iso_wall, frame_elapsed_s)
                rule_eval = rule_engine.evaluate(feats)
                smoothed = smoother.process(rule_eval)

                beh = smoothed.smoothed_behavior
                conc = is_concentrating(beh)
                dist = is_distracted(beh)
                unc = (beh == BehaviorCategory.UNCERTAIN)

                if conc:
                    frame_conc += 1
                elif dist:
                    frame_dist += 1
                else:
                    frame_unc += 1

                if tid not in track_observations:
                    track_observations[tid] = []

                track_observations[tid].append({
                    "frame_index": current_frame_idx,
                    "elapsed_s": frame_elapsed_s,
                    "behavior": beh,
                    "raw_behavior": rule_eval.behavior,
                    "is_concentrating": conc,
                    "is_distracted": dist,
                    "is_uncertain": unc,
                    "face_quality": feats.face_quality,
                    "head_pitch": feats.head_pitch,
                    "head_yaw": feats.head_yaw,
                    "ear_avg": feats.ear_avg,
                    "bbox": state.bbox,
                })

                if annotated_frame is not None:
                    self._draw_student_annotation(annotated_frame, state.bbox, tid, beh, conc, dist)

            frame_metrics[current_frame_idx] = {
                "concentrating": frame_conc,
                "distracted": frame_dist,
                "uncertain": frame_unc,
                "total": visible_students_in_frame,
                "elapsed_s": frame_elapsed_s,
            }

            if annotated_frame is not None and writer is not None:
                self._draw_hud_banner(
                    annotated_frame,
                    frame_elapsed_s,
                    duration_s,
                    visible_students_in_frame,
                    frame_conc,
                    frame_dist,
                    frame_unc,
                )
                writer.write(annotated_frame)

            # Periodic progress reporting
            if progress_callback and (processed_count % 15 == 0 or current_frame_idx == total_frames):
                elapsed_now = time.time() - t_audit_start
                cur_fps = processed_count / elapsed_now if elapsed_now > 0 else 0.0
                status_msg = f"Auditing {video_basename}: frame {current_frame_idx}/{total_frames} ({cur_fps:.1f} FPS)"
                progress_callback(current_frame_idx, total_frames, cur_fps, status_msg)

        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Annotated video saved to: {annotated_video_path}")

        t_audit_end = time.time()
        total_audit_time = t_audit_end - t_audit_start
        overall_analysis_fps = processed_count / total_audit_time if total_audit_time > 0 else 0.0

        # Aggregate metrics
        result = self._aggregate_results(
            video_path=video_path,
            video_basename=video_basename,
            duration_s=duration_s,
            total_frames=total_frames,
            processed_frames=processed_count,
            frame_stride=frame_stride,
            fps=fps,
            resolution=f"{width}x{height}",
            total_audit_time=total_audit_time,
            overall_analysis_fps=overall_analysis_fps,
            track_observations=track_observations,
            frame_metrics=frame_metrics,
            interval_seconds=interval_seconds,
            annotated_video_path=annotated_video_path,
            reports_dir=reports_dir,
        )

        return result

    def _draw_student_annotation(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        tid: str,
        behavior: BehaviorCategory,
        is_conc: bool,
        is_dist: bool,
    ) -> None:
        """Renders bounding box and attention badge over a student."""
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        bw = x2 - x1

        # Color scheme (BGR)
        # Concentrating: Bright Green (46, 204, 113)
        # Distracted: Crimson Red (52, 73, 235)
        # Uncertain: Cool Gray (149, 165, 166)
        if is_conc:
            box_color = (46, 204, 113)
            status_text = "CONFIDENTLY WATCHING" if bw >= 140 else "WATCHING"
        elif is_dist:
            box_color = (52, 73, 235)
            if behavior == BehaviorCategory.LOOKING_AWAY:
                reason = "LOOKING AWAY"
            elif behavior == BehaviorCategory.DROWSY_SLEEP_LIKE:
                reason = "DROWSY/TIRED"
            elif behavior == BehaviorCategory.PHONE_USE:
                reason = "PHONE USE"
            elif behavior == BehaviorCategory.HEAD_DOWN:
                reason = "HEAD DOWN"
            else:
                reason = "DISTRACTED"
            status_text = f"DISTRACTED ({reason})" if bw >= 140 else f"DISTRACTED: {reason}"
        else:
            box_color = (149, 165, 166)
            status_text = "UNCERTAIN"

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # Responsive badge size
        font = cv2.FONT_HERSHEY_SIMPLEX
        if bw < 95:
            label = f"{tid}: {status_text.split(' ')[0]}"
            scale = 0.35
            thickness = 1
        elif bw < 140:
            label = f"{tid}: {status_text}"
            scale = 0.42
            thickness = 1
        else:
            label = f"{tid}: {status_text} [{behavior.value}]"
            scale = 0.50
            thickness = 1

        (lw, lh), base = cv2.getTextSize(label, font, scale, thickness)
        pill_y1 = max(0, y1 - lh - 8)
        pill_y2 = y1
        cv2.rectangle(img, (x1, pill_y1), (min(w - 1, x1 + lw + 6), pill_y2), box_color, -1)
        cv2.putText(img, label, (x1 + 3, pill_y2 - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_hud_banner(
        self,
        img: np.ndarray,
        elapsed_s: float,
        duration_s: float,
        students_count: int,
        conc_count: int,
        dist_count: int,
        unc_count: int,
    ) -> None:
        """Renders professional top HUD bar with video time and attention statistics."""
        w = img.shape[1]
        hud_h = 42

        # Draw semi-transparent dark banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Time formatting MM:SS
        cur_min, cur_sec = divmod(int(elapsed_s), 60)
        tot_min, tot_sec = divmod(int(duration_s), 60)
        time_str = f"Time: {cur_min:02d}:{cur_sec:02d} / {tot_min:02d}:{tot_sec:02d}"

        # Attention percentages
        active_cnt = conc_count + dist_count + unc_count
        conc_pct = (conc_count / active_cnt * 100.0) if active_cnt > 0 else 0.0
        dist_pct = (dist_count / active_cnt * 100.0) if active_cnt > 0 else 0.0

        font = cv2.FONT_HERSHEY_SIMPLEX

        if w >= 1100:
            stats_str = (
                f"Students: {students_count}  |  "
                f"Confidently Watching: {conc_count} ({conc_pct:.0f}%)  |  "
                f"Distracted: {dist_count} ({dist_pct:.0f}%)  |  "
                f"Uncertain: {unc_count}"
            )
            cv2.putText(img, "LECTURE ATTENTION AUDIT", (12, 26), font, 0.55, (52, 152, 219), 2, cv2.LINE_AA)
            cv2.putText(img, time_str, (280, 26), font, 0.55, (236, 240, 241), 1, cv2.LINE_AA)
            cv2.putText(img, stats_str, (470, 26), font, 0.55, (46, 204, 113) if conc_pct >= 50 else (52, 73, 235), 1, cv2.LINE_AA)
        else:
            # Compact responsive layout for 720p / 432p widths
            title_time = f"AUDIT | {time_str}"
            stats_str = f"Students: {students_count} | Watching: {conc_count} ({conc_pct:.0f}%) | Distracted: {dist_count} ({dist_pct:.0f}%)"
            cv2.putText(img, title_time, (8, 18), font, 0.42, (52, 152, 219), 1, cv2.LINE_AA)
            cv2.putText(img, stats_str, (8, 36), font, 0.42, (46, 204, 113) if conc_pct >= 50 else (52, 73, 235), 1, cv2.LINE_AA)

    def _aggregate_results(
        self,
        video_path: str,
        video_basename: str,
        duration_s: float,
        total_frames: int,
        processed_frames: int,
        frame_stride: int,
        fps: float,
        resolution: str,
        total_audit_time: float,
        overall_analysis_fps: float,
        track_observations: Dict[str, List[Dict[str, Any]]],
        frame_metrics: Dict[int, Dict[str, int]],
        interval_seconds: float,
        annotated_video_path: Optional[str],
        reports_dir: str,
    ) -> VideoAuditResult:
        """Aggregates student records, time epochs, and writes markdown & JSON audit artifacts."""
        total_unique_students = len(track_observations)

        # 1. Per-Student Summaries
        student_summaries: List[StudentAttentionSummary] = []
        all_behaviors_count: Dict[str, int] = {lbl: 0 for lbl in ALL_BEHAVIOR_LABELS}
        total_student_observations = 0
        total_conc_obs = 0
        total_dist_obs = 0
        total_unc_obs = 0

        # Sort tracks by ID
        sorted_tids = sorted(track_observations.keys())

        for tid in sorted_tids:
            obs_list = track_observations[tid]
            vis_frames = len(obs_list)
            if vis_frames == 0:
                continue

            vis_seconds = (vis_frames * frame_stride) / fps
            vis_pct = (vis_seconds / duration_s * 100.0) if duration_s > 0 else 0.0

            conc_frames = sum(1 for o in obs_list if o["is_concentrating"])
            dist_frames = sum(1 for o in obs_list if o["is_distracted"])
            unc_frames = sum(1 for o in obs_list if o["is_uncertain"])

            conc_s = (conc_frames * frame_stride) / fps
            dist_s = (dist_frames * frame_stride) / fps
            unc_s = (unc_frames * frame_stride) / fps

            conc_pct = (conc_frames / vis_frames) * 100.0
            dist_pct = (dist_frames / vis_frames) * 100.0
            unc_pct = (unc_frames / vis_frames) * 100.0

            total_student_observations += vis_frames
            total_conc_obs += conc_frames
            total_dist_obs += dist_frames
            total_unc_obs += unc_frames

            # Behavior frequencies
            beh_counts: Dict[str, int] = {lbl: 0 for lbl in ALL_BEHAVIOR_LABELS}
            fq_list: List[float] = []
            dist_breakdown = {
                "looking_away": 0,
                "phone_use": 0,
                "head_down": 0,
                "drowsy": 0,
            }

            prev_was_distracted = False
            distraction_events = 0

            for o in obs_list:
                b = o["behavior"].value if isinstance(o["behavior"], BehaviorCategory) else str(o["behavior"])
                if b in beh_counts:
                    beh_counts[b] += 1
                if b in all_behaviors_count:
                    all_behaviors_count[b] += 1

                if o["is_distracted"]:
                    if not prev_was_distracted:
                        distraction_events += 1
                        prev_was_distracted = True
                    if b == BehaviorCategory.LOOKING_AWAY.value:
                        dist_breakdown["looking_away"] += 1
                    elif b == BehaviorCategory.PHONE_USE.value:
                        dist_breakdown["phone_use"] += 1
                    elif b == BehaviorCategory.HEAD_DOWN.value:
                        dist_breakdown["head_down"] += 1
                    elif b == BehaviorCategory.DROWSY_SLEEP_LIKE.value:
                        dist_breakdown["drowsy"] += 1
                else:
                    prev_was_distracted = False

                if o.get("face_quality") is not None:
                    fq_list.append(float(o["face_quality"]))

            # Dominant behavior
            dominant_beh = max(beh_counts.items(), key=lambda kv: kv[1])[0]
            beh_breakdown = {lbl: round((cnt / vis_frames) * 100.0, 1) for lbl, cnt in beh_counts.items()}
            mean_fq = float(np.mean(fq_list)) if fq_list else 0.0

            # Attention rating
            if conc_pct >= 70.0:
                rating = "HIGH_ATTENTION"
            elif conc_pct >= 40.0:
                rating = "MODERATE_ATTENTION"
            else:
                rating = "INATTENTIVE"

            student_summaries.append(
                StudentAttentionSummary(
                    track_id=tid,
                    total_visible_frames=vis_frames,
                    total_visible_seconds=round(vis_seconds, 2),
                    visibility_percentage=round(vis_pct, 1),
                    concentrating_frames=conc_frames,
                    concentrating_seconds=round(conc_s, 2),
                    concentrating_percentage=round(conc_pct, 1),
                    distracted_frames=dist_frames,
                    distracted_seconds=round(dist_s, 2),
                    distracted_percentage=round(dist_pct, 1),
                    uncertain_frames=unc_frames,
                    uncertain_seconds=round(unc_s, 2),
                    uncertain_percentage=round(unc_pct, 1),
                    dominant_behavior=dominant_beh,
                    attention_rating=rating,
                    behavior_breakdown=beh_breakdown,
                    distraction_breakdown=dist_breakdown,
                    distraction_event_count=distraction_events,
                    mean_face_quality=round(mean_fq, 2),
                )
            )

        # 2. Overall Classroom Attention Distribution
        if total_student_observations > 0:
            overall_conc_pct = round((total_conc_obs / total_student_observations) * 100.0, 1)
            overall_dist_pct = round((total_dist_obs / total_student_observations) * 100.0, 1)
            overall_unc_pct = round((total_unc_obs / total_student_observations) * 100.0, 1)
            overall_beh_dist = {
                lbl: round((cnt / total_student_observations) * 100.0, 1)
                for lbl, cnt in all_behaviors_count.items()
            }
        else:
            overall_conc_pct = overall_dist_pct = overall_unc_pct = 0.0
            overall_beh_dist = {lbl: 0.0 for lbl in ALL_BEHAVIOR_LABELS}

        # Average students per frame
        active_student_counts = [fm["total"] for fm in frame_metrics.values()]
        avg_students_per_frame = float(np.mean(active_student_counts)) if active_student_counts else 0.0

        # 3. Timeline Interval Analysis
        timeline: List[TimeIntervalSummary] = []
        num_intervals = math.ceil(duration_s / interval_seconds) if duration_s > 0 else 1

        for i in range(num_intervals):
            t_start = i * interval_seconds
            t_end = min(duration_s, (i + 1) * interval_seconds)

            # Find all frames falling into this interval
            int_frames = [
                fm for fm in frame_metrics.values()
                if t_start <= fm["elapsed_s"] < t_end
            ]

            if int_frames:
                avg_present = int(round(np.mean([f["total"] for f in int_frames])))
                tot_conc = sum(f["concentrating"] for f in int_frames)
                tot_dist = sum(f["distracted"] for f in int_frames)
                tot_unc = sum(f["uncertain"] for f in int_frames)
                tot_all = tot_conc + tot_dist + tot_unc

                att_pct = (tot_conc / tot_all * 100.0) if tot_all > 0 else 0.0
                if att_pct >= 70.0:
                    status = "High Concentration"
                elif att_pct >= 45.0:
                    status = "Moderate"
                else:
                    status = "Attention Lapse"
            else:
                avg_present = 0
                tot_conc = tot_dist = tot_unc = 0
                att_pct = 0.0
                status = "No Students"

            start_m, start_s = divmod(int(t_start), 60)
            end_m, end_s = divmod(int(t_end), 60)
            label = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"

            timeline.append(
                TimeIntervalSummary(
                    interval_index=i + 1,
                    start_seconds=round(t_start, 1),
                    end_seconds=round(t_end, 1),
                    time_label=label,
                    students_present=avg_present,
                    students_concentrating=tot_conc,
                    students_distracted=tot_dist,
                    students_uncertain=tot_unc,
                    attention_percentage=round(att_pct, 1),
                    attention_status=status,
                )
            )

        timestamp_str = now_utc_iso().replace(":", "").replace("-", "")

        # Initial object without markdown/paths
        res = VideoAuditResult(
            video_path=video_path,
            video_filename=video_basename,
            analysis_timestamp=now_utc_iso(),
            video_duration_seconds=round(duration_s, 2),
            total_video_frames=total_frames,
            processed_frames=processed_frames,
            frame_stride=frame_stride,
            analysis_duration_seconds=round(total_audit_time, 2),
            analysis_fps=round(overall_analysis_fps, 1),
            fps=round(fps, 1),
            resolution=resolution,
            total_unique_students=total_unique_students,
            average_students_per_frame=round(avg_students_per_frame, 2),
            overall_concentration_percentage=overall_conc_pct,
            overall_distraction_percentage=overall_dist_pct,
            overall_uncertain_percentage=overall_unc_pct,
            behavior_distribution=overall_beh_dist,
            students=student_summaries,
            timeline=timeline,
            annotated_video_path=annotated_video_path,
        )

        # Generate markdown report
        md_content = self.generate_markdown_report(res)
        res.summary_markdown = md_content

        # Save artifacts to disk
        report_md_path = os.path.join(reports_dir, f"AUDIT_REPORT_{video_basename}_{timestamp_str}.md")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        res.report_markdown_path = report_md_path

        report_json_path = os.path.join(reports_dir, f"AUDIT_SUMMARY_{video_basename}_{timestamp_str}.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        res.report_json_path = report_json_path

        logger.info(f"Audit report saved: {report_md_path}")
        logger.info(f"Audit summary saved: {report_json_path}")

        return res

    def generate_markdown_report(self, res: VideoAuditResult) -> str:
        """Formats the audit metrics into an executive-ready Markdown Audit Report."""
        dur_min, dur_sec = divmod(int(res.video_duration_seconds), 60)

        # Determine overall classroom verdict
        if res.overall_concentration_percentage >= 70.0:
            verdict = "EXCELLENT ENGAGEMENT (Classroom demonstrates robust focus)"
            verdict_badge = "HIGH ATTENTION"
        elif res.overall_concentration_percentage >= 50.0:
            verdict = "SATISFACTORY ENGAGEMENT (Normal instructional interaction with minor distraction episodes)"
            verdict_badge = "MODERATE ATTENTION"
        else:
            verdict = "HIGH DISTRACTION OBSERVED (Multiple students exhibiting off-task behaviors)"
            verdict_badge = "LOW ATTENTION"

        lines = [
            f"# Classroom Attention & Behavioral Audit Report",
            f"",
            f"**Video Source:** `{res.video_filename}`  ",
            f"**Audit Execution Time:** `{res.analysis_timestamp}`  ",
            f"**Analysis Throughput:** `{res.processed_frames}` frames processed in `{res.analysis_duration_seconds:.1f}s` (`{res.analysis_fps:.1f} FPS`)  ",
            f"**Annotated Video Output:** `{res.annotated_video_path or 'Not generated'}`  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"| Metric | Value | Interpretation / Context |",
            f"| :--- | :--- | :--- |",
            f"| **Video Duration** | **{dur_min:02d}:{dur_sec:02d}** ({res.video_duration_seconds:.1f}s) | Total duration analyzed |",
            f"| **Resolution & Native FPS** | **{res.resolution} @ {res.fps:.1f} FPS** | Video frame dimensions and frame rate |",
            f"| **Total Students Detected** | **{res.total_unique_students} Students** | Unique anonymous tracks observed |",
            f"| **Average Students / Frame** | **{res.average_students_per_frame:.1f} Students** | Concurrently visible students |",
            f"| **Overall Attention / Concentration** | **{res.overall_concentration_percentage:.1f}%** | Forward-facing instructional engagement + Desk work |",
            f"| **Overall Distraction Rate** | **{res.overall_distraction_percentage:.1f}%** | Looking away, phone use, head down, or drowsiness |",
            f"| **Uncertain / Unobservable** | **{res.overall_uncertain_percentage:.1f}%** | Ambiguous posture or degraded landmarks |",
            f"| **Classroom Attention Verdict** | **{verdict_badge}** | {verdict} |",
            f"",
            f"---",
            f"",
            f"## 2. Observable Behavioral Taxonomy Distribution",
            f"",
            f"The system categorizes physical student actions into seven observable physical categories:",
            f"",
            f"| Observable Behavior | Time Fraction (%) | Physical Description / Educational Meaning |",
            f"| :--- | :--- | :--- |",
        ]

        tax_descriptions = {
            "ENGAGED_BEHAVIOR": "Attending to instruction: forward-facing posture, open eyes, upright head",
            "WRITING_READING": "Desk coursework: downward gaze with hands/wrists positioned near table",
            "LOOKING_AWAY": "Visual distraction: head turned significantly away from front (window, peer, ceiling)",
            "HEAD_DOWN": "Disengagement: downward head pitch without active writing or reading indicators",
            "PHONE_USE": "Digital distraction: electronic handheld device observed in hand/desk zone",
            "DROWSY_SLEEP_LIKE": "Fatigue/resting: prolonged eye closure or head resting on desk/arm",
            "UNCERTAIN": "Sensor boundary: occlusion, steep angle, or ambiguous visual evidence",
        }

        for beh in ALL_BEHAVIOR_LABELS:
            pct = res.behavior_distribution.get(beh, 0.0)
            desc = tax_descriptions.get(beh, "")
            lines.append(f"| **{beh}** | **{pct:.1f}%** | {desc} |")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Individual Student Attention Breakdown",
            f"",
            f"| Student ID | Visible Time | Concentrating (%) | Distracted (%) | Uncertain (%) | Dominant Action | Distraction Events | Attention Level |",
            f"| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |",
        ])

        for s in res.students:
            min_v, sec_v = divmod(int(s.total_visible_seconds), 60)
            vis_str = f"{min_v:02d}:{sec_v:02d} ({s.visibility_percentage:.0f}%)"
            lines.append(
                f"| **{s.track_id}** | {vis_str} | **{s.concentrating_percentage:.1f}%** ({s.concentrating_seconds:.1f}s) | "
                f"{s.distracted_percentage:.1f}% ({s.distracted_seconds:.1f}s) | {s.uncertain_percentage:.1f}% | "
                f"`{s.dominant_behavior}` | {s.distraction_event_count} | **{s.attention_rating}** |"
            )

        lines.extend([
            f"",
            f"### Per-Student Distraction Details",
            f"",
            f"| Student ID | Looking Away Frames | Phone Use Frames | Head Down Frames | Drowsy/Sleep Frames |",
            f"| :---: | :---: | :---: | :---: | :---: |",
        ])

        for s in res.students:
            db = s.distraction_breakdown
            lines.append(
                f"| **{s.track_id}** | {db.get('looking_away', 0)} | {db.get('phone_use', 0)} | "
                f"{db.get('head_down', 0)} | {db.get('drowsy', 0)} |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Lecture Timeline & Engagement Shifts",
            f"",
            f"Timeline sliced into {int(res.timeline[0].end_seconds - res.timeline[0].start_seconds) if res.timeline else 15}-second intervals:",
            f"",
            f"| Time Window | Students Present | Concentrating | Distracted | Uncertain | Attention % | Status |",
            f"| :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])

        for t in res.timeline:
            lines.append(
                f"| `{t.time_label}` | {t.students_present} | {t.students_concentrating} | "
                f"{t.students_distracted} | {t.students_uncertain} | **{t.attention_percentage:.1f}%** | {t.attention_status} |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Scientific Construct Validity & Methodological Safeguards",
            f"",
            f"> [!IMPORTANT]",
            f"> **Boundary of Measurement:** This system measures **observable physical behaviors** ",
            f"> (e.g. forward orientation, gaze openness, desk-level writing postures, phone detection). ",
            f"> It **does NOT directly measure internal cognitive states**, comprehension, or neurological attention. ",
            f"> Observable behaviors serve as empirical behavioral proxies for educational analytics.",
            f"",
            f"- **Zero-Biometric Provenance:** Anonymous IDs (`S001`, `S002`, etc.) are assigned per session. No facial embeddings or biometric profiles are computed or retained.",
            f"- **Explainable Rule Grounding:** Behavioral classifications are produced via an auditable rule engine and temporal smoother, ensuring full transparency without opaque black-box hallucination.",
            f"",
            f"*Generated by Lecture Engagement Pattern Analyzer — Automated Video Audit Engine.*",
            f"",
        ])

        return "\n".join(lines)

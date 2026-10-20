"""CLI entry point for Automated Video Attention Audit & Summary Generation.

Analyzes 2-5 minute classroom/lecture videos and generates:
1. Executive attention audit metrics (Concentrating vs. Distracted student counts and %).
2. Per-student engagement and distraction logs.
3. Time-sliced attention shift timeline.
4. Exported Markdown Audit Report and JSON summary.
5. Optional overlay-annotated MP4 video.

Usage:
    python scripts/analyze_video_audit.py --video data/recordings/classroom_benchmark.mp4
    python scripts/analyze_video_audit.py --video data/recordings/SES_20261019_054258_f420de.mp4 --stride 2
"""

import argparse
import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lecture_analyzer.analytics.video_audit_service import VideoAttentionAuditService
from lecture_analyzer.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Analyze 2-5 minute video and generate comprehensive attention audit.")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="data/recordings/classroom_benchmark.mp4",
        help="Path to recorded lecture/classroom video file (MP4, AVI, MOV).",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=1,
        help="Frame sampling stride (1 = every frame, 2 = every 2nd frame). Default: 1",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=15.0,
        help="Time window size in seconds for timeline interval analysis. Default: 15.0",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/reports",
        help="Directory to save audit report, JSON summary, and annotated video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video generation for maximum processing throughput.",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"[ERROR] Specified video file does not exist: {video_path}")
        sys.exit(1)

    config = load_config()
    service = VideoAttentionAuditService(config)

    print("=" * 72)
    print(" LECTURE ENGAGEMENT PATTERN ANALYZER — VIDEO ATTENTION AUDIT")
    print("=" * 72)
    print(f"Target Video:    {video_path}")
    print(f"Frame Stride:    {args.stride}")
    print(f"Annotated Video: {'Disabled' if args.no_video else 'Enabled'}")
    print(f"Output Dir:      {os.path.abspath(args.output_dir)}")
    print("-" * 72)

    def on_progress(cur, total, fps, msg):
        pct = (cur / total * 100.0) if total > 0 else 0.0
        sys.stdout.write(f"\r[AUDITING] Frame {cur:5d}/{total:5d} ({pct:5.1f}%) | Speed: {fps:4.1f} FPS")
        sys.stdout.flush()

    try:
        result = service.audit_video(
            video_path=video_path,
            generate_annotated_video=not args.no_video,
            output_dir=args.output_dir,
            frame_stride=args.stride,
            interval_seconds=args.interval,
            progress_callback=on_progress,
        )
    except Exception as e:
        print(f"\n[FATAL] Audit execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    print("\n" + "=" * 72)
    print(" AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 72)
    dur_m, dur_s = divmod(int(result.video_duration_seconds), 60)
    print(f"Video Duration:      {dur_m:02d}:{dur_s:02d} ({result.video_duration_seconds:.1f}s)")
    print(f"Processed Frames:    {result.processed_frames} / {result.total_video_frames} (Analysis FPS: {result.analysis_fps:.1f})")
    print(f"Total Students:      {result.total_unique_students} unique student tracks")
    print(f"Avg Students/Frame:  {result.average_students_per_frame:.1f}")
    print("-" * 72)
    print("CLASSROOM ATTENTION SUMMARY:")
    print(f"  [+] Concentrated / Paying Attention: {result.overall_concentration_percentage:5.1f}%")
    print(f"  [-] Distracted / Off-Task:          {result.overall_distraction_percentage:5.1f}%")
    print(f"  [?] Uncertain / Unobservable:        {result.overall_uncertain_percentage:5.1f}%")
    print("-" * 72)
    print("INDIVIDUAL STUDENT ROSTER:")
    for s in result.students:
        print(
            f"  * {s.track_id:4s} | Attn: {s.concentrating_percentage:5.1f}% ({s.concentrating_seconds:4.1f}s) | "
            f"Dist: {s.distracted_percentage:5.1f}% ({s.distracted_seconds:4.1f}s) | "
            f"Dominant: {s.dominant_behavior:18s} | Rating: {s.attention_rating}"
        )
    print("-" * 72)
    print(f"Markdown Audit Report: file:///{result.report_markdown_path.replace(os.sep, '/')}")
    print(f"JSON Machine Summary:  file:///{result.report_json_path.replace(os.sep, '/')}")
    if result.annotated_video_path:
        print(f"Annotated Output Video: file:///{result.annotated_video_path.replace(os.sep, '/')}")
    print("=" * 72)


if __name__ == "__main__":
    main()

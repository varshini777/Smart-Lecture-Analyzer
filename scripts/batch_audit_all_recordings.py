"""Batch Classroom Video Attention Auditor.

Scans all uploaded recordings in data/recordings, processes each video bit-by-bit,
generates second-by-second overlay annotated playback videos, individual audit reports,
and a consolidated multi-video classroom attention audit report.
"""

import os
from pathlib import Path
import sys
import time

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lecture_analyzer.analytics.video_audit_service import VideoAttentionAuditService, VideoAuditResult
from lecture_analyzer.config import load_config


def run_batch_audit(
    recordings_dir: str = "data/recordings",
    reports_dir: str = "data/reports",
    frame_stride: int = 2,
):
    rec_path = Path(recordings_dir)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        [p for p in rec_path.glob("*.mp4") if not p.name.startswith("ANNOTATED_")]
    )

    if not video_files:
        print(f"[ERROR] No video files found in {recordings_dir}")
        return []

    print("=" * 80)
    print(" BATCH CLASSROOM ATTENTION AUDITOR — MULTI-STUDENT SCANNER")
    print("=" * 80)
    print(f"Found {len(video_files)} video file(s) in {recordings_dir}:")
    for idx, vf in enumerate(video_files, 1):
        size_mb = vf.stat().st_size / (1024 * 1024)
        print(f"  {idx}. {vf.name} ({size_mb:.2f} MB)")
    print("-" * 80)

    config = load_config()
    # Ensure sensitive detection for small multi-student grid crops
    config.vision.person_confidence_threshold = 0.28
    service = VideoAttentionAuditService(config)

    all_results: list[VideoAuditResult] = []
    t_batch_start = time.time()

    for idx, vf in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Scanning: {vf.name}")
        print("  Status: Processing frames and rendering overlay playback video...")

        def on_progress(cur, total, fps, msg):
            pct = (cur / total * 100.0) if total > 0 else 0.0
            sys.stdout.write(f"\r  --> Progress: {cur:4d}/{total:4d} frames ({pct:5.1f}%) | {fps:4.1f} FPS")
            sys.stdout.flush()

        try:
            res = service.audit_video(
                video_path=str(vf),
                generate_annotated_video=True,
                output_dir=str(rep_path),
                frame_stride=frame_stride,
                interval_seconds=10.0,
                progress_callback=on_progress,
            )
            all_results.append(res)
            print()
            print(f"  --> Completed: {res.total_unique_students} students detected")
            print(f"  --> Confidently Watching: {res.overall_concentration_percentage:.1f}%")
            print(f"  --> Distracted:           {res.overall_distraction_percentage:.1f}%")
            print(f"  --> Uncertain:            {res.overall_uncertain_percentage:.1f}%")
            print(f"  --> Annotated Video:      {res.annotated_video_path}")
            print(f"  --> Audit Report:         {res.report_markdown_path}")
        except Exception as e:
            print(f"\n  [ERROR] Failed to audit {vf.name}: {e}")
            import traceback
            traceback.print_exc()

    t_batch_end = time.time()
    total_batch_time = t_batch_end - t_batch_start

    # Generate Consolidated Classroom Audit Report
    consolidated_path = rep_path / "CONSOLIDATED_CLASSROOM_AUDIT_REPORT.md"
    generate_consolidated_report(all_results, consolidated_path, total_batch_time)
    print("\n" + "=" * 80)
    print(f" ALL SCANS COMPLETED IN {total_batch_time:.1f}s")
    print(f" Master Report Generated: {consolidated_path}")
    print("=" * 80)

    return all_results


def generate_consolidated_report(results: list[VideoAuditResult], output_path: Path, total_time: float):
    lines = [
        "# Consolidated Multi-Video Classroom Attention Audit Report",
        "",
        f"**Audit Generated:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`  ",
        f"**Total Videos Audited:** `{len(results)} recordings`  ",
        f"**Total Audit Processing Time:** `{total_time:.1f} seconds`  ",
        "",
        "---",
        "",
        "## 1. Executive Multi-Video Comparison Dashboard",
        "",
        "| Video File | Duration | Students | Confidently Watching (%) | Distracted (%) | Uncertain (%) | Dominant Status | Annotated Video File |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in results:
        dur_m, dur_s = divmod(int(r.video_duration_seconds), 60)
        ann_name = os.path.basename(r.annotated_video_path) if r.annotated_video_path else "N/A"
        if r.overall_concentration_percentage >= 70.0:
            status = "HIGH ATTENTION"
        elif r.overall_concentration_percentage >= 45.0:
            status = "MODERATE"
        else:
            status = "LOW ATTENTION"

        lines.append(
            f"| `{r.video_filename}` | {dur_m:02d}:{dur_s:02d} | **{r.total_unique_students}** | "
            f"**{r.overall_concentration_percentage:.1f}%** | {r.overall_distraction_percentage:.1f}% | "
            f"{r.overall_uncertain_percentage:.1f}% | **{status}** | `{ann_name}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Per-Video Breakdown",
        "",
    ])

    for idx, r in enumerate(results, 1):
        dur_m, dur_s = divmod(int(r.video_duration_seconds), 60)
        lines.extend([
            f"### Video {idx}: `{r.video_filename}`",
            f"- **Resolution & Native FPS:** `{r.resolution} @ {r.fps:.1f} FPS`",
            f"- **Duration:** `{dur_m:02d}:{dur_s:02d}` ({r.video_duration_seconds:.1f}s, {r.total_video_frames} frames)",
            f"- **Unique Tracked Students:** `{r.total_unique_students}`",
            f"- **Concentration Rate (Confidently Watching):** **`{r.overall_concentration_percentage:.1f}%`**",
            f"- **Distraction Rate:** **`{r.overall_distraction_percentage:.1f}%`**",
            f"- **Uncertainty Rate:** **`{r.overall_uncertain_percentage:.1f}%`**",
            f"- **Annotated Video File:** `data/reports/{os.path.basename(r.annotated_video_path)}`" if r.annotated_video_path else "",
            "",
            "#### Student Roster (Sample)",
            "",
            "| Student ID | Visible Time | Concentrating (%) | Distracted (%) | Rating | Dominant Action |",
            "| :---: | :---: | :---: | :---: | :---: | :--- |",
        ])

        for s in r.students[:10]:
            vm, vs = divmod(int(s.total_visible_seconds), 60)
            lines.append(
                f"| **{s.track_id}** | {vm:02d}:{vs:02d} | **{s.concentrating_percentage:.1f}%** | "
                f"{s.distracted_percentage:.1f}% | `{s.attention_rating}` | `{s.dominant_behavior}` |"
            )
        if len(r.students) > 10:
            lines.append(f"| *... and {len(r.students) - 10} more students* | | | | | |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. How to Play Annotated Videos with Second-by-Second Scanned Results",
        "",
        "1. Open the **`data/reports`** directory.",
        "2. Locate the corresponding **`ANNOTATED_<video_name>.mp4`** file.",
        "3. Play the video in any media player (Windows Media Player, VLC, QuickTime) or within the Desktop Application under **Audit Video File**.",
        "4. **On-Screen Display Elements:**",
        "   - **Top HUD Banner:** Real-time elapsed time (`Time: MM:SS / MM:SS`), Total Students detected, Confidently Watching count and %, Distracted count and %.",
        "   - **Bounding Box Badges:**",
        "     - **Green Box:** `Sxx: CONFIDENTLY WATCHING` (attending to instructional space with alert posture).",
        "     - **Red Box:** `Sxx: DISTRACTED: [REASON]` (looking away, drowsy/resting, head down, or phone use).",
        "     - **Gray Box:** `Sxx: UNCERTAIN` (degraded landmarks or sensor boundary).",
        "",
        "*Generated by Lecture Engagement Pattern Analyzer — Automated Video Audit Engine.*",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_batch_audit(frame_stride=2)

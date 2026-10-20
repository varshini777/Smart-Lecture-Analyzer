"""CLI utility to audit recorded sessions."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_analyzer.config import load_config
from lecture_analyzer.dataset.quality_audit import QualityAuditor
from lecture_analyzer.storage.database import Database
from lecture_analyzer.storage.repositories import SessionRepository


def main():
    parser = argparse.ArgumentParser(description="Audit quality of recorded observation sessions.")
    parser.add_argument("--session-id", type=str, help="Specific session ID to audit. If omitted, audits all sessions.")
    args = parser.parse_args()

    cfg = load_config()
    db = Database(cfg.storage.database_path)
    auditor = QualityAuditor(db)
    session_repo = SessionRepository(db)

    if args.session_id:
        target_ids = [args.session_id]
    else:
        target_ids = [s.session_id for s in session_repo.list_all()]

    if not target_ids:
        print("[*] No sessions recorded in database to audit.")
        return 0

    print(f"[*] Auditing {len(target_ids)} session(s)...")
    for sid in target_ids:
        report = auditor.audit_session(sid, persist_result=True)
        print(f"\n=== Session: {sid} | Verdict: {report.verdict.value} ===")
        print(f"  Duration:            {report.duration_seconds:.1f}s")
        print(f"  Observations:        {report.observation_count}")
        print(f"  Effective Rate:      {report.effective_fps:.1f} FPS (mean {report.mean_interval_ms:.1f} ms, std {report.std_interval_ms:.1f} ms)")
        print(f"  Face Quality (mean): {report.face_quality_mean:.2f}")
        print(f"  Annotation Coverage: {report.annotation_coverage_fraction * 100.0:.1f}%")
        if report.descriptive_agreement_percentage is not None:
            print(f"  Descriptive Agreement: {report.descriptive_agreement_percentage:.1f}%")
        if report.warnings:
            print("  Warnings:")
            for w in report.warnings:
                print(f"    - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

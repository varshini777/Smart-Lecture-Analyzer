"""Database validation and schema integrity check utility."""

import logging
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_analyzer.config import load_config
from lecture_analyzer.storage.database import Database
from lecture_analyzer.utils.paths import resolve_path


def validate_database() -> bool:
    print("=== Database Integrity & Schema Validation ===")
    config = load_config()
    db_path = resolve_path(config.storage.database_path)
    print(f"[*] Database file: {db_path}")

    db = Database(db_path)
    conn = db.get_connection()

    try:
        # 1. PRAGMA integrity_check
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        res = cur.fetchall()
        if res and res[0][0] == "ok":
            print(" [PASS] SQLite integrity check: OK")
        else:
            print(f" [FAIL] SQLite integrity check returned: {res}")
            return False

        # 2. PRAGMA foreign_key_check
        cur.execute("PRAGMA foreign_key_check;")
        fk_violations = cur.fetchall()
        if not fk_violations:
            print(" [PASS] Foreign key constraints: Clean (0 violations)")
        else:
            print(f" [FAIL] Foreign key violations detected: {fk_violations}")
            return False

        # 3. Table verification
        tables = [
            "schema_migrations", "sessions", "session_tracks", "observations",
            "annotations", "quality_reports", "dataset_exports", "ml_models", "ml_evaluations"
        ]
        print("\n[*] Table Verification:")
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t};")
            cnt = cur.fetchone()[0]
            print(f"    [TABLE] {t:22s}: {cnt:6d} rows")

        print("\n[SUCCESS] Database is healthy and all schema constraints are satisfied.")
        return True
    finally:
        conn.close()


if __name__ == "__main__":
    ok = validate_database()
    sys.exit(0 if ok else 1)

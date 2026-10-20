"""Database migrations and initialization manager."""

import logging
import sqlite3
from pathlib import Path
from typing import Union

from lecture_analyzer.storage.schema import CREATE_INDICES_SQL, CREATE_TABLES_SQL, CURRENT_SCHEMA_VERSION
from lecture_analyzer.utils.paths import resolve_path
from lecture_analyzer.utils.timestamps import now_utc_iso

logger = logging.getLogger("lecture_analyzer.storage.migrations")


def initialize_database(db_path: Union[str, Path]) -> None:
    """Initializes the database schema and applies pending migrations cleanly."""
    path = resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing database schema at {path}...")
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    try:
        with conn:
            # Create migrations table first
            conn.execute(CREATE_TABLES_SQL[0])

            cur = conn.cursor()
            cur.execute("SELECT MAX(version) FROM schema_migrations;")
            row = cur.fetchone()
            current_ver = row[0] if (row and row[0] is not None) else 0

            if current_ver < CURRENT_SCHEMA_VERSION:
                logger.info(f"Applying schema migration from version {current_ver} to {CURRENT_SCHEMA_VERSION}...")
                for ddl in CREATE_TABLES_SQL[1:]:
                    conn.execute(ddl)

                for idx_ddl in CREATE_INDICES_SQL:
                    conn.execute(idx_ddl)

                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                    (CURRENT_SCHEMA_VERSION, now_utc_iso())
                )
                logger.info(f"Schema migration to v{CURRENT_SCHEMA_VERSION} completed successfully.")
            else:
                logger.debug(f"Database schema is already up to date (v{current_ver}).")
    finally:
        conn.close()

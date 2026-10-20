"""SQLite database connection manager and context provider."""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Generator, Optional, Union

from lecture_analyzer.storage.migrations import initialize_database
from lecture_analyzer.utils.paths import resolve_path

logger = logging.getLogger("lecture_analyzer.storage.database")


class Database:
    """Thread-safe SQLite database manager ensuring integrity and safe concurrency."""

    def __init__(self, db_path: Union[str, Path] = "data/database/lecture_analyzer.db"):
        self.db_path = resolve_path(db_path)
        # Initialize schema if not present
        initialize_database(self.db_path)

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with foreign keys and WAL mode enabled."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Provides a managed transaction block that commits on success or rolls back on exception."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            with conn:
                yield cur
        except Exception as e:
            logger.error(f"Database transaction error: {e}", exc_info=True)
            raise
        finally:
            cur.close()
            conn.close()

    def execute_query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Executes a SELECT query and returns all rows."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            conn.close()

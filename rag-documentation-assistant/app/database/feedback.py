import sqlite3
import threading
from datetime import datetime, timezone

from app import config

_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.FEEDBACK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the feedback table if it doesn't exist (idempotent)."""
    global _initialized
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
                    comment TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def save_feedback(question: str, rating: str, comment: str | None = None) -> int:
    """Insert one feedback row. Returns the new row id."""
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            cursor = conn.execute(
                "INSERT INTO feedback (question, rating, comment, created_at)"
                " VALUES (?, ?, ?, ?)",
                (question, rating, comment, created_at),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()


def list_feedback(limit: int = 100) -> list[dict]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, question, rating, comment, created_at FROM feedback"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

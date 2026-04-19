"""
SQLite-based violation logger — no external database required.

Schema
------
violations(id, timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit_kmh, snapshot_path)
"""

import sqlite3
import os
import datetime

# DB lives next to this package's parent directory (project root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "violations.db")


# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the violations table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                vehicle_id      INTEGER,
                vehicle_class   TEXT,
                speed_kmh       REAL,
                speed_limit_kmh REAL,
                snapshot_path   TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_violation(
    vehicle_id: int,
    vehicle_class: str,
    speed_kmh: float,
    speed_limit_kmh: float,
    snapshot_path: str,
) -> str:
    """Insert a violation record and return the timestamp string."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO violations
                (timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit_kmh, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, vehicle_id, vehicle_class,
             round(speed_kmh, 1), round(speed_limit_kmh, 1), snapshot_path),
        )
        conn.commit()
    finally:
        conn.close()
    return timestamp


def fetch_all_violations() -> list[tuple]:
    """
    Return all violations ordered newest-first.

    Each row: (id, timestamp, vehicle_class, speed_kmh, speed_limit_kmh, snapshot_path)
    """
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            """
            SELECT id, timestamp, vehicle_class, speed_kmh, speed_limit_kmh, snapshot_path
            FROM violations
            ORDER BY id DESC
            """
        )
        return cur.fetchall()
    finally:
        conn.close()


def clear_violations() -> None:
    """Delete all rows from the violations table."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM violations")
        conn.commit()
    finally:
        conn.close()


def get_violation_count() -> int:
    """Return total number of stored violations."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM violations")
        return cur.fetchone()[0]
    finally:
        conn.close()

"""
SQLite-based violation logger — no external database required.

Schema
------
violations(id, timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit_kmh,
           snapshot_path, violation_type, license_plate)

Supports:  overspeeding · no_helmet · wrong_lane · triple_riding
"""

import sqlite3
import os
import datetime

# DB lives next to this package's parent directory (project root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "violations.db")


# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the violations table if it doesn't already exist, and ensure
    the schema includes the newer columns (safe migration)."""
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
                snapshot_path   TEXT,
                violation_type  TEXT    DEFAULT 'overspeeding',
                license_plate   TEXT    DEFAULT ''
            )
        """)
        conn.commit()

        # ── Safe migration for existing DBs ──────────────────────────────
        # Add columns that might be missing in an older schema
        _ensure_column(conn, "violations", "violation_type", "TEXT DEFAULT 'overspeeding'")
        _ensure_column(conn, "violations", "license_plate",  "TEXT DEFAULT ''")
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Add *column* to *table* if it doesn't already exist (no-op otherwise)."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        conn.commit()


def log_violation(
    vehicle_id: int,
    vehicle_class: str,
    speed_kmh: float,
    speed_limit_kmh: float,
    snapshot_path: str,
    violation_type: str = "overspeeding",
    license_plate: str = "",
) -> str:
    """Insert a violation record and return the timestamp string."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO violations
                (timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit_kmh,
                 snapshot_path, violation_type, license_plate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, vehicle_id, vehicle_class,
             round(speed_kmh, 1), round(speed_limit_kmh, 1), snapshot_path,
             violation_type, license_plate),
        )
        conn.commit()
    finally:
        conn.close()
    return timestamp


def fetch_all_violations() -> list[tuple]:
    """
    Return all violations ordered newest-first.

    Each row: (id, timestamp, vehicle_class, speed_kmh, speed_limit_kmh,
               snapshot_path, violation_type, license_plate)
    """
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            """
            SELECT id, timestamp, vehicle_class, speed_kmh, speed_limit_kmh,
                   snapshot_path, violation_type, license_plate
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


def get_violation_count_by_type(violation_type: str) -> int:
    """Return count of violations of a specific type."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type = ?",
            (violation_type,),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()

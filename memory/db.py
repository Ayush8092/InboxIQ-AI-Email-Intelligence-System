"""
Database initialisation with migration support.
Adds `source` column to existing emails table if missing.
"""
import os
import sqlite3
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/aeoa.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialise DB schema and run migrations."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    schema_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "memory", "schema.sql",
    )
    # Try relative path
    if not os.path.exists(schema_path):
        schema_path = "memory/schema.sql"

    conn = get_connection()
    try:
        with open(schema_path) as f:
            conn.executescript(f.read())
        conn.commit()

        # ── Migration: add source column if not exists ────────────────────
        _migrate_add_source_column(conn)

        logger.info("Database initialised.")
    finally:
        conn.close()


def _migrate_add_source_column(conn: sqlite3.Connection):
    """
    Migration: add `source` column to emails table.
    Safe to run on existing databases — skips if column already exists.
    """
    try:
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(emails)").fetchall()
        ]
        if "source" not in cols:
            conn.execute(
                "ALTER TABLE emails ADD COLUMN source TEXT DEFAULT 'demo'"
            )
            conn.commit()
            logger.info("Migration: added `source` column to emails table")
    except Exception as e:
        logger.warning(f"Migration skipped: {type(e).__name__}")
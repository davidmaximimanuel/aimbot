"""
sync_files_to_supabase.py
─────────────────────────
Run once on every Railway deploy (add to your Procfile or start command).
Reads every .py file in the project directory and upserts it into the
`project_files` table in your AIM Supabase so AIM can read its own code.

Usage (add to Railway start command):
    python sync_files_to_supabase.py && gunicorn aimbot:app ...

Supabase table required (run once in SQL editor):
    CREATE TABLE IF NOT EXISTS project_files (
        filename    TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        file_size   INT,
        updated_at  TIMESTAMPTZ DEFAULT now()
    );
"""

import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sync")

# ── Config ─────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Files/dirs to skip
SKIP_EXTENSIONS = {".pyc", ".pyo", ".log", ".tmp", ".env"}
SKIP_DIRS       = {"__pycache__", ".git", "node_modules", ".venv", "venv", "env"}
SKIP_FILES      = {"sync_files_to_supabase.py"}  # don't need to sync the sync script itself

# Max file size to store (50KB — keeps Supabase row sizes sane)
MAX_FILE_SIZE = 50_000

def get_project_files(base_dir: str) -> list[dict]:
    """Walk the project dir and collect all readable .py files."""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        # Prune dirs we don't want to recurse into
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in filenames:
            if name in SKIP_FILES:
                continue
            _, ext = os.path.splitext(name)
            if ext not in {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml"}:
                continue
            if ext in SKIP_EXTENSIONS:
                continue
            full_path = os.path.join(root, name)
            rel_path  = os.path.relpath(full_path, base_dir)
            try:
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    logger.warning("⚠️  Skipping %s (too large: %d bytes)", rel_path, size)
                    continue
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                files.append({
                    "filename":   rel_path,
                    "content":    content,
                    "file_size":  size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning("⚠️  Could not read %s: %s", rel_path, e)
    return files


def sync_to_supabase(files: list[dict]) -> None:
    """Upsert all files into the project_files table."""
    try:
        from supabase import create_client
    except ImportError:
        logger.error("❌ supabase-py not installed — skipping sync")
        return

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("❌ SUPABASE_URL or SUPABASE_KEY not set — skipping sync")
        return

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error("❌ Could not connect to Supabase: %s", e)
        return

    success, failed = 0, 0
    for file in files:
        try:
            client.table("project_files").upsert(file, on_conflict="filename").execute()
            logger.info("✅ Synced: %s (%d bytes)", file["filename"], file["file_size"])
            success += 1
        except Exception as e:
            logger.error("❌ Failed to sync %s: %s", file["filename"], e)
            failed += 1

    logger.info("🎉 Sync complete — %d synced, %d failed", success, failed)

    # Clean up stale files (files deleted from repo but still in Supabase)
    try:
        current_names = [f["filename"] for f in files]
        existing = client.table("project_files").select("filename").execute()
        stale = [r["filename"] for r in (existing.data or []) if r["filename"] not in current_names]
        for name in stale:
            client.table("project_files").delete().eq("filename", name).execute()
            logger.info("🗑️  Removed stale file: %s", name)
    except Exception as e:
        logger.warning("⚠️  Could not clean stale files: %s", e)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info("🔍 Scanning project files in: %s", base_dir)
    files = get_project_files(base_dir)
    logger.info("📦 Found %d files to sync", len(files))
    sync_to_supabase(files)
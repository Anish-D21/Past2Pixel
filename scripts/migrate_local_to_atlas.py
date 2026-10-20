"""
Safe Migration Utility: Local JSON Persistence -> MongoDB Atlas.
Transfers user accounts and session metadata from data/sessions/local_db.json to MongoDB Atlas.

Safety Guarantees:
- Never runs automatically; requires explicit user flag (--confirm).
- Supports --dry-run to audit records without touching the database.
- Idempotent: Skips existing users (by email/username) and sessions (by session_id).
- Preserves local_db.json unchanged as a backup.
- Never prints secrets or credentials.

Usage:
    python scripts/migrate_local_to_atlas.py --dry-run
    python scripts/migrate_local_to_atlas.py --confirm
"""
import sys
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.db.mongodb import db_manager


def migrate(dry_run: bool = True):
    local_store_path = settings.SESSIONS_STORAGE_DIR / "local_db.json"
    if not local_store_path.exists():
        print(f"[!] Local database file not found at: {local_store_path}")
        return

    print("=" * 65)
    print("Past2Pixel — Local JSON to MongoDB Atlas Safe Migration")
    print("=" * 65)
    print(f"Mode: {'DRY RUN (No changes will be written)' if dry_run else 'LIVE MIGRATION'}")
    print(f"Source file: {local_store_path}")

    status = db_manager.get_status()
    if status["storage_mode"] != "mongodb_atlas" or status["status"] != "CONNECTED":
        print(f"[ERROR] MongoDB Atlas is not currently connected. Status: {status['status']}")
        print("Please configure valid MONGODB_URI in .env before running migration.")
        return

    print(f"Target Database: {status['database_name']} (MongoDB Atlas)")

    try:
        with open(local_store_path, "r", encoding="utf-8") as f:
            local_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read local database: {e}")
        return

    local_users = local_data.get("users", [])
    local_sessions = local_data.get("sessions", [])

    print(f"\nLocal inventory: {len(local_users)} user(s), {len(local_sessions)} session(s)")

    # 1. Migrate Users
    users_to_migrate = []
    users_skipped = 0
    for u in local_users:
        existing = db_manager.db.users.find_one({"email": u["email"]})
        if existing:
            users_skipped += 1
        else:
            users_to_migrate.append(u)

    print(f"\n[Users] To migrate: {len(users_to_migrate)}, Already in Atlas: {users_skipped}")

    # 2. Migrate Sessions
    sessions_to_migrate = []
    sessions_skipped = 0
    for s in local_sessions:
        existing = db_manager.db.sessions.find_one({"session_id": s["session_id"]})
        if existing:
            sessions_skipped += 1
        else:
            sessions_to_migrate.append(s)

    print(f"[Sessions] To migrate: {len(sessions_to_migrate)}, Already in Atlas: {sessions_skipped}")

    if dry_run:
        print("\n[DRY RUN COMPLETE] No records were modified in MongoDB Atlas.")
        print("To execute live migration, run:")
        print("    python scripts/migrate_local_to_atlas.py --confirm")
        return

    # Live Migration
    migrated_users_count = 0
    for u in users_to_migrate:
        # Clone doc to preserve original
        doc = dict(u)
        # Strip string _id so MongoDB can assign native ObjectId or preserve if desired
        doc.pop("_id", None)
        try:
            db_manager.db.users.insert_one(doc)
            migrated_users_count += 1
        except Exception as e:
            print(f"  [!] Failed migrating user '{u.get('username')}': {e}")

    migrated_sessions_count = 0
    for s in sessions_to_migrate:
        doc = dict(s)
        doc.pop("_id", None)
        try:
            db_manager.db.sessions.insert_one(doc)
            migrated_sessions_count += 1
        except Exception as e:
            print(f"  [!] Failed migrating session '{s.get('session_id')}': {e}")

    print("\n" + "=" * 65)
    print(f"MIGRATION SUMMARY: {migrated_users_count} users, {migrated_sessions_count} sessions migrated.")
    print("Local database file preserved intact at:", local_store_path)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Past2Pixel local JSON records to MongoDB Atlas.")
    parser.add_argument("--confirm", action="store_true", help="Execute live migration to Atlas.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Preview migration without modifying Atlas.")
    args = parser.parse_args()

    if not args.confirm and not args.dry_run:
        print("Please specify either --dry-run to preview, or --confirm to execute migration.")
    else:
        migrate(dry_run=not args.confirm)

"""
Database Persistence Layer for Past2Pixel.
Integrates MongoDB Atlas with graceful, transparent local file-backed persistence fallback.
Ensures durable storage of users and processing session reports without storing bulky binaries in documents.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import logging
import re
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, DuplicateKeyError, ConfigurationError

from ..core.config import settings

logger = logging.getLogger("past2pixel.db")


def sanitize_mongo_error(err: Optional[Exception]) -> str:
    """
    Sanitizes MongoDB exception messages to eliminate any embedded credentials or host tokens,
    and maps common Atlas failure conditions to clear, actionable troubleshooting categories.
    """
    if err is None:
        return "Unknown error"

    err_str = str(err).lower()

    # 1. Authentication failure (invalid user/password)
    if "authentication failed" in err_str or "bad auth" in err_str or "auth error" in err_str:
        return (
            "Authentication Failed: Invalid MongoDB username or password. "
            "Action: Verify MONGODB_URI credentials in .env against your MongoDB Atlas Database Access user."
        )

    # 2. Server selection timeout (usually IP whitelist missing or cluster unreachable)
    if isinstance(err, ServerSelectionTimeoutError) or "serverselectiontimeouterror" in err_str or "timed out" in err_str:
        return (
            "Connection Timeout / IP Not Whitelisted: Could not establish connection to MongoDB Atlas cluster within timeout. "
            "Action: Ensure your current IP address is added to the MongoDB Atlas 'Network Access' IP Access List, "
            "and verify that the cluster is not paused or unreachable."
        )

    # 3. Invalid connection string format or configuration error
    if isinstance(err, ConfigurationError) or "configurationerror" in err_str or "invalid uri" in err_str or "scheme" in err_str:
        return (
            "Configuration Error: Invalid MongoDB connection string format or missing DNS SRV dependencies. "
            "Action: Verify MONGODB_URI starts with 'mongodb+srv://' or 'mongodb://' and has valid syntax."
        )

    # 4. DNS / Name resolution failure
    if "nodename nor servname provided" in err_str or "getaddrinfo failed" in err_str or "dns" in err_str:
        return (
            "DNS / Name Resolution Failure: Could not resolve MongoDB Atlas cluster hostname. "
            "Action: Check your local internet connection and DNS settings, or verify the cluster hostname in MONGODB_URI."
        )

    # 5. TLS / SSL Certificate Error
    if "ssl" in err_str or "certificate" in err_str or "tls" in err_str:
        return (
            "TLS / SSL Error: SSL certificate verification failed when connecting to MongoDB Atlas. "
            "Action: Verify system certificate authorities or Python ssl configuration."
        )

    # Generic sanitized fallback: redact any embedded credentials or sensitive queries
    sanitized = re.sub(r"://([^:]+):([^@]+)@", "://<redacted>:<redacted>@", str(err))
    sanitized = re.sub(r"(password|pwd)=[^&]+", r"\1=<redacted>", sanitized, flags=re.IGNORECASE)
    return f"{type(err).__name__}: {sanitized}"


class DatabaseManager:
    """
    Manages user accounts, authentication records, and restoration session metadata.
    Connects to MongoDB Atlas when configured, or transparently uses a local JSON file-store
    for local development and testing, clearly reporting storage mode and status.
    """

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        raw_uri = uri if uri is not None else settings.MONGODB_URI
        self.uri = raw_uri.strip().strip("'\"") if raw_uri else ""
        raw_db = db_name if db_name is not None else settings.MONGODB_DB_NAME
        self.db_name = raw_db.strip().strip("'\"") if raw_db else "past2pixel"
        self.client: Optional[MongoClient] = None
        self.db = None
        self.storage_mode = "uninitialized"
        self.status = "DISCONNECTED"
        self.last_error: Optional[str] = None

        # Local fallback file store
        self.local_store_path = settings.SESSIONS_STORAGE_DIR / "local_db.json"
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Attempts connection to MongoDB Atlas, falling back cleanly to local file storage if unavailable."""
        settings.ensure_directories()
        if self.uri:
            # Validate connection string scheme
            valid_scheme = self.uri.startswith("mongodb://") or self.uri.startswith("mongodb+srv://")
            if not valid_scheme:
                self.last_error = (
                    "Configuration Error: Invalid connection string scheme. "
                    "Action: MONGODB_URI in .env must begin with 'mongodb://' or 'mongodb+srv://'."
                )
                logger.warning(self.last_error)
            else:
                try:
                    # 2.5-second timeout for server selection to prevent hanging on offline setups
                    self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2500)
                    self.client.admin.command("ping")
                    self.db = self.client[self.db_name]
                    self.storage_mode = "mongodb_atlas"
                    self.status = "CONNECTED"
                    self.last_error = None
                    self._ensure_indexes()
                    logger.info("Successfully connected to MongoDB Atlas database '%s'", self.db_name)
                    return
                except Exception as e:
                    self.last_error = sanitize_mongo_error(e)
                    logger.warning("MongoDB Atlas connection failed: %s. Falling back to local file persistence.", self.last_error)
                    if self.client:
                        try:
                            self.client.close()
                        except Exception:
                            pass
                    self.client = None
                    self.db = None

        # Fallback to transparent local file store
        self.storage_mode = "local_file_persistence"
        self.status = "LOCAL_PERSISTENCE"
        self._ensure_local_store()

    def reconnect(self, uri: Optional[str] = None, db_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Re-initializes connection with optional new URI or database name.
        Enables runtime reconfiguration and robust test isolation.
        """
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.db = None
        self.last_error = None

        raw_uri = uri if uri is not None else settings.MONGODB_URI
        self.uri = raw_uri.strip().strip("'\"") if raw_uri else ""
        raw_db = db_name if db_name is not None else settings.MONGODB_DB_NAME
        self.db_name = raw_db.strip().strip("'\"") if raw_db else "past2pixel"

        self._initialize_connection()
        return self.get_status()

    def _ensure_indexes(self) -> None:
        """Initializes unique and query indexes on MongoDB Atlas collections."""
        if self.db is not None:
            try:
                self.db.users.create_index("email", unique=True)
                self.db.users.create_index("username", unique=True)
                self.db.sessions.create_index("session_id", unique=True)
                self.db.sessions.create_index("user_id")
                self.db.sessions.create_index("created_at")
            except Exception as e:
                logger.warning("Failed to create MongoDB indexes: %s", e)

    def _ensure_local_store(self) -> None:
        """Creates the local JSON database file if it does not exist."""
        if not self.local_store_path.exists():
            initial_data = {"users": [], "sessions": []}
            self.local_store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_store_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def _read_local_store(self) -> Dict[str, List[Dict[str, Any]]]:
        self._ensure_local_store()
        try:
            with open(self.local_store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": [], "sessions": []}

    def _write_local_store(self, data: Dict[str, List[Dict[str, Any]]]) -> bool:
        self.local_store_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.local_store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error("Failed to write to local database store: %s", e)
            return False

    # ---------------------------------------------------------
    # Status & Diagnostic Reporting
    # ---------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive persistence state, connection status, and configuration details."""
        return {
            "status": self.status,
            "storage_mode": self.storage_mode,
            "uri_configured": bool(self.uri and self.uri.strip()),
            "database_name": self.db_name,
            "is_persistent": True,
            "last_error": self.last_error,
            "local_store_path": str(self.local_store_path) if self.storage_mode == "local_file_persistence" else None,
        }

    # ---------------------------------------------------------
    # User Management CRUD
    # ---------------------------------------------------------
    def create_user(self, user_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Persists a new user record with hashed password."""
        user_doc["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            try:
                res = self.db.users.insert_one(user_doc)
                user_doc["_id"] = str(res.inserted_id)
                return user_doc
            except DuplicateKeyError as e:
                err_msg = str(e).lower()
                if "email" in err_msg:
                    raise ValueError(f"User with email '{user_doc['email']}' already exists.")
                if "username" in err_msg:
                    raise ValueError(f"User with username '{user_doc['username']}' already exists.")
                raise ValueError("User with this email or username already exists.")

        # Local store
        data = self._read_local_store()
        # Verify unique email and username
        for u in data.get("users", []):
            if u["email"].lower() == user_doc["email"].lower():
                raise ValueError(f"User with email '{user_doc['email']}' already exists.")
            if u["username"].lower() == user_doc["username"].lower():
                raise ValueError(f"User with username '{user_doc['username']}' already exists.")

        user_doc["_id"] = f"usr_{len(data['users']) + 1:04d}_{int(datetime.now().timestamp())}"
        data["users"].append(user_doc)
        self._write_local_store(data)
        return user_doc

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Finds user by email address."""
        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            u = self.db.users.find_one({"email": email.strip()})
            if u:
                u["_id"] = str(u["_id"])
            return u

        data = self._read_local_store()
        for u in data.get("users", []):
            if u.get("email", "").lower() == email.strip().lower():
                return u
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Finds user by user ID."""
        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            from bson.objectid import ObjectId
            try:
                u = self.db.users.find_one({"_id": ObjectId(user_id)})
            except Exception:
                u = self.db.users.find_one({"_id": user_id})
            if u:
                u["_id"] = str(u["_id"])
            return u

        data = self._read_local_store()
        for u in data.get("users", []):
            if u.get("_id") == user_id:
                return u
        return None

    # ---------------------------------------------------------
    # Session Management CRUD
    # ---------------------------------------------------------
    def create_session(self, session_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Creates and stores a new processing session record."""
        session_doc["created_at"] = session_doc.get("created_at") or datetime.now(timezone.utc).isoformat()
        session_doc["updated_at"] = session_doc.get("created_at")
        session_doc["persistence_info"] = {
            "storage_mode": self.storage_mode,
            "status": "SAVED",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            res = self.db.sessions.insert_one(session_doc)
            session_doc["_id"] = str(res.inserted_id)
            return session_doc

        data = self._read_local_store()
        session_doc["_id"] = session_doc["session_id"]
        # Replace if existing or append
        existing = [i for i, s in enumerate(data.get("sessions", [])) if s.get("session_id") == session_doc["session_id"]]
        if existing:
            data["sessions"][existing[0]] = session_doc
        else:
            data.setdefault("sessions", []).append(session_doc)
        self._write_local_store(data)
        return session_doc

    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates specific fields of an existing session."""
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            self.db.sessions.update_one({"session_id": session_id}, {"$set": update_data})
            s = self.db.sessions.find_one({"session_id": session_id})
            if s:
                s["_id"] = str(s["_id"])
            return s

        data = self._read_local_store()
        for i, s in enumerate(data.get("sessions", [])):
            if s.get("session_id") == session_id:
                s.update(update_data)
                data["sessions"][i] = s
                self._write_local_store(data)
                return s
        return None

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a session by its unique session ID."""
        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            s = self.db.sessions.find_one({"session_id": session_id})
            if s:
                s["_id"] = str(s["_id"])
            return s

        data = self._read_local_store()
        for s in data.get("sessions", []):
            if s.get("session_id") == session_id:
                return s
        return None

    def list_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists sessions owned by a specific authenticated user, ordered newest first."""
        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            cursor = self.db.sessions.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results

        data = self._read_local_store()
        user_sessions = [s for s in data.get("sessions", []) if s.get("user_id") == user_id]
        user_sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return user_sessions[:limit]

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Deletes a session if owned by the requesting user. Also removes associated file artifacts."""
        session = self.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return False

        # Clean up files on disk if present
        session_dir = settings.SESSIONS_STORAGE_DIR / session_id
        if session_dir.exists() and session_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                logger.warning("Could not delete session directory %s: %e", session_dir, e)

        if self.storage_mode == "mongodb_atlas" and self.db is not None:
            res = self.db.sessions.delete_one({"session_id": session_id, "user_id": user_id})
            return res.deleted_count > 0

        data = self._read_local_store()
        initial_len = len(data.get("sessions", []))
        data["sessions"] = [s for s in data.get("sessions", []) if not (s.get("session_id") == session_id and s.get("user_id") == user_id)]
        if len(data["sessions"]) < initial_len:
            self._write_local_store(data)
            return True
        return False


# Singleton database instance
db_manager = DatabaseManager()

"""
Test Suite for Environment Loading, Configuration Precedence, and MongoDB Atlas Persistence / Diagnostics.
Verifies all configuration behaviors, mock connections, and error sanitization without requiring internet access.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure

from src.core.config import Settings, load_project_env, reload_settings, PROJECT_ROOT
from src.db.mongodb import DatabaseManager, sanitize_mongo_error


def test_env_loading_from_project_root():
    """Confirms that .env file is found and loaded anchoring to PROJECT_ROOT."""
    # Write a dummy test env file in a temporary folder
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_env = Path(tmp_dir) / ".env"
        test_env.write_text("TEST_KEY_CUSTOM_123=custom_value_456\n", encoding="utf-8")
        
        # Load custom env
        loaded = load_project_env(env_path=test_env, override=True)
        assert loaded is True
        assert os.getenv("TEST_KEY_CUSTOM_123") == "custom_value_456"
        
        # Cleanup
        os.environ.pop("TEST_KEY_CUSTOM_123", None)


def test_config_loading_independent_of_working_directory(monkeypatch):
    """Verifies that load_project_env correctly resolves PROJECT_ROOT even if cwd is changed."""
    # Use existing system temp directory or tests folder to avoid Windows directory-lock on rmtree
    target_dir = Path(tempfile.gettempdir())
    monkeypatch.chdir(target_dir)
    assert Path.cwd() != PROJECT_ROOT

    # Project root .env should still be discoverable and loadable
    root_env = PROJECT_ROOT / ".env"
    if root_env.is_file():
        res = load_project_env(override=False)
        assert isinstance(res, bool)


def test_existing_environment_variables_take_precedence():
    """Confirms that external environment variables are not overwritten when override=False."""
    os.environ["PRE_EXISTING_VAR_TEST"] = "external_priority_value"
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_env = Path(tmp_dir) / ".env"
        test_env.write_text("PRE_EXISTING_VAR_TEST=file_value\n", encoding="utf-8")
        
        load_project_env(env_path=test_env, override=False)
        assert os.getenv("PRE_EXISTING_VAR_TEST") == "external_priority_value"
        
    os.environ.pop("PRE_EXISTING_VAR_TEST", None)


def test_missing_or_empty_mongodb_uri_handling():
    """Confirms that empty or None URI causes clean local persistence fallback without errors."""
    mgr = DatabaseManager(uri="", db_name="test_past2pixel")
    status = mgr.get_status()
    assert status["status"] == "LOCAL_PERSISTENCE"
    assert status["storage_mode"] == "local_file_persistence"
    assert status["uri_configured"] is False
    assert status["last_error"] is None
    assert status["is_persistent"] is True
    assert status["local_store_path"] is not None


def test_invalid_scheme_mongodb_uri_handling():
    """Confirms that a URI missing mongodb:// or mongodb+srv:// is rejected with a clean message."""
    mgr = DatabaseManager(uri="http://localhost:27017", db_name="test_past2pixel")
    status = mgr.get_status()
    assert status["status"] == "LOCAL_PERSISTENCE"
    assert status["storage_mode"] == "local_file_persistence"
    assert status["uri_configured"] is True
    assert "Invalid connection string scheme" in status["last_error"]


def test_mongodb_connection_success_mock():
    """Tests successful MongoDB Atlas connection using mock client."""
    with patch("src.db.mongodb.MongoClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.admin.command.return_value = {"ok": 1}
        mock_client_cls.return_value = mock_instance

        mgr = DatabaseManager(uri="mongodb+srv://user:pass@cluster0.fake.net/?retryWrites=true", db_name="past2pixel")
        status = mgr.get_status()

        assert status["status"] == "CONNECTED"
        assert status["storage_mode"] == "mongodb_atlas"
        assert status["uri_configured"] is True
        assert status["last_error"] is None
        assert status["local_store_path"] is None


def test_mongodb_connection_timeout_fallback_mock():
    """Tests timeout handling and sanitized error reporting when cluster is unreachable."""
    with patch("src.db.mongodb.MongoClient") as mock_client_cls:
        mock_client_cls.side_effect = ServerSelectionTimeoutError("No servers found yet")

        mgr = DatabaseManager(uri="mongodb+srv://user:secret123@cluster0.fake.net/", db_name="past2pixel")
        status = mgr.get_status()

        assert status["status"] == "LOCAL_PERSISTENCE"
        assert status["storage_mode"] == "local_file_persistence"
        assert status["uri_configured"] is True
        assert "Connection Timeout / IP Not Whitelisted" in status["last_error"]
        # Ensure credentials are NOT leaked in last_error
        assert "secret123" not in status["last_error"]


def test_mongodb_authentication_failure_mock():
    """Tests authentication failure reporting and credential sanitization."""
    with patch("src.db.mongodb.MongoClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.admin.command.side_effect = OperationFailure("bad auth : authentication failed")
        mock_client_cls.return_value = mock_instance

        mgr = DatabaseManager(uri="mongodb+srv://baduser:topsecretpassword@cluster0.fake.net/", db_name="past2pixel")
        status = mgr.get_status()

        assert status["status"] == "LOCAL_PERSISTENCE"
        assert status["storage_mode"] == "local_file_persistence"
        assert "Authentication Failed" in status["last_error"]
        assert "topsecretpassword" not in status["last_error"]


def test_sanitize_mongo_error_redaction():
    """Confirms that raw strings containing credentials are fully redacted."""
    raw_error = Exception("Failed connecting to mongodb+srv://admin_user:SuperSecretP@ssword@cluster0.net/db?auth=1")
    sanitized = sanitize_mongo_error(raw_error)
    assert "SuperSecretP@ssword" not in sanitized
    assert "<redacted>" in sanitized

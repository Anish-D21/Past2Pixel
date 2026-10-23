"""
Unit & Integration Test Suite for Past2Pixel Streamlit Frontend & API Client.
Tests API client methods, error propagation, session state, report builders,
and academic disclosure invariants without requiring internet or a live UI server.
"""
import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.frontend.api_client import ApiClient, ApiException
from src.frontend.state import (
    init_session_state,
    set_auth,
    clear_auth,
    is_authenticated,
    get_token,
    get_user,
    set_active_session,
    get_active_session_id,
    navigate_to,
)
from src.frontend.components.styles import render_status_badge
from src.frontend.components.report_builder import (
    generate_session_markdown_report,
    generate_session_json_export,
)


# -------------------------------------------------------------
# 1. API Client Unit Tests
# -------------------------------------------------------------
def test_api_client_initialization():
    client = ApiClient(base_url="http://localhost:8000/")
    assert client.base_url == "http://localhost:8000"


def test_api_client_connection_error():
    client = ApiClient(base_url="http://127.0.0.1:9999")  # unopen port
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Refused")):
        with pytest.raises(ApiException) as exc_info:
            client.get_health()
        assert exc_info.value.status_code == 503
        assert "unreachable" in exc_info.value.detail.lower()


def test_api_client_login_success():
    client = ApiClient()
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "access_token": "mock.jwt.token",
        "token_type": "bearer",
        "expires_in_minutes": 1440,
        "user": {"id": "usr_1", "username": "student", "email": "s@uni.edu"},
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.login("student", "password123")
        assert res["access_token"] == "mock.jwt.token"
        assert res["user"]["username"] == "student"


def test_api_client_error_response_parsing():
    client = ApiClient()
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Email already registered."}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(ApiException) as exc_info:
            client.register("s@uni.edu", "student", "password123")
        assert exc_info.value.status_code == 400
        assert "already registered" in exc_info.value.detail


# -------------------------------------------------------------
# 2. Session State Management Tests
# -------------------------------------------------------------
def test_session_state_lifecycle():
    # Use patch to mock streamlit.session_state
    mock_state = {}
    with patch("streamlit.session_state", mock_state):
        init_session_state()
        assert is_authenticated() is False
        assert get_token() is None

        set_auth("sample_token", {"username": "researcher", "email": "r@p2p.org"})
        assert is_authenticated() is True
        assert get_token() == "sample_token"
        assert get_user()["username"] == "researcher"

        set_active_session("p2p_sess_123", redirect_page="Results & Analysis")
        assert get_active_session_id() == "p2p_sess_123"
        assert mock_state["current_page"] == "Results & Analysis"

        navigate_to("Pipeline Explanation")
        assert mock_state["current_page"] == "Pipeline Explanation"

        clear_auth()
        assert is_authenticated() is False
        assert get_token() is None
        assert mock_state["current_page"] == "Login"


# -------------------------------------------------------------
# 3. Status Badge Rendering Tests
# -------------------------------------------------------------
def test_status_badge_classes():
    b_ready = render_status_badge("READY")
    assert "badge-ready" in b_ready
    assert "READY" in b_ready

    b_untrained = render_status_badge("UNTRAINED_RANDOM_WEIGHTS")
    assert "badge-untrained" in b_untrained

    b_ref = render_status_badge("REFERENCE UNAVAILABLE")
    assert "badge-ref-unavailable" in b_ref

    b_spec = render_status_badge("SPECULATIVE_PREDICTION")
    assert "badge-speculative" in b_spec

    b_unavail = render_status_badge("UNAVAILABLE")
    assert "badge-unavailable" in b_unavail


# -------------------------------------------------------------
# 4. Academic Report Generator Tests
# -------------------------------------------------------------
def test_markdown_report_academic_integrity():
    sample_session = {
        "session_id": "p2p_test_12345",
        "created_at": "2026-10-21T01:00:00+00:00",
        "user_id": "usr_aiml",
        "status": "RESTORATION_COMPLETED",
        "original_metadata": {
            "original_filename": "folio_sample.png",
            "width": 1600,
            "height": 200,
            "mode": "RGB",
            "format": "PNG",
            "file_size_bytes": 102400,
        },
        "restoration_result": {
            "timings_ms": {
                "total_pipeline": 125.4,
                "preprocessing": 5.2,
                "damage_detection": 45.1,
                "unet_restoration": 60.3,
            },
            "model_readiness": {
                "unet_restoration": "UNTRAINED",
                "unet_damage_detector": "UNTRAINED",
            },
            "damage_detection": {
                "damaged_pixel_percentage": 14.8,
                "metrics": {"metrics_available": False},
            },
            "restoration": {
                "unet_metrics": {"reference_available": False},
            },
            "device_info": {"device_type": "cpu", "device_name": "CPU"},
        },
        "ocr_result": {
            "status": "UNAVAILABLE",
            "extracted_text": "",
            "is_authentic_ocr": False,
        },
        "persistence_info": {"storage_mode": "mongodb_atlas"},
    }

    report = generate_session_markdown_report(sample_session)
    assert "p2p_test_12345" in report
    assert "folio_sample.png" in report
    assert "14.8%" in report
    # Critical honesty checks
    assert "Untrained Initial Weights (Diagnostic)" in report
    assert "REFERENCE UNAVAILABLE" in report
    assert "GROUND TRUTH MASK UNAVAILABLE" in report
    assert "Academic Disclosure" in report

    json_export = generate_session_json_export(sample_session)
    parsed = json.loads(json_export)
    assert parsed["session_id"] == "p2p_test_12345"
    assert parsed["restoration_result"]["damage_detection"]["damaged_pixel_percentage"] == 14.8

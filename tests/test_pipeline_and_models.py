"""
Test Suite for AI Pipeline Execution, Model Readiness Status, and Partial Failure Handling.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.model_registry import model_registry

client = TestClient(app)


def test_model_readiness_untrained_status():
    """Verifies that uninitialized checkpoints are reported as UNTRAINED, never fabricated as trained."""
    statuses = model_registry.get_all_statuses()
    for model_name, info in statuses.items():
        assert info["checkpoint_exists"] is False
        assert "UNTRAINED" in str(info["status"])


def test_restoration_execution_and_artifacts():
    # 1. Upload an image
    img = Image.new("RGB", (256, 256), color=(230, 215, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_res = client.post(
        "/api/upload",
        files={"file": ("test_folio.png", buf, "image/png")},
    )
    assert upload_res.status_code == 201
    session_id = upload_res.json()["session_id"]

    # 2. Run restoration
    res = client.post("/api/restore", json={"session_id": session_id, "device_preference": "cpu"})
    assert res.status_code == 200
    data = res.json()

    assert data["session_id"] == session_id
    assert "timings_ms" in data
    assert "damage_detection" in data
    assert "restoration" in data
    assert "artifacts" in data
    assert data["artifacts"]["restored"] is not None
    assert data["artifacts"]["damage_overlay"] is not None
    assert data["artifacts"]["damage_mask"] is not None

    # Verify model readiness explicitly flags untrained status
    assert "UNTRAINED" in data["model_readiness"]["unet_restoration"]


def test_restoration_missing_session():
    res = client.post("/api/restore", json={"session_id": "p2p_missing_12345"})
    assert res.status_code == 404

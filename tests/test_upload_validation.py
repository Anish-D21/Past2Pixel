"""
Test Suite for Upload Validation, Content Security, File Sizes, and Image Decodability.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_valid_image_upload():
    img = Image.new("RGB", (300, 200), color=(235, 220, 190))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/upload",
        files={"file": ("manuscript_folio.png", buf, "image/png")},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["session_id"].startswith("p2p_")
    assert data["width"] == 300
    assert data["height"] == 200
    assert data["image_url"].startswith("/static/sessions/")


def test_unsupported_file_extension_rejected():
    fake_txt = io.BytesIO(b"Hello world, this is a text file.")
    res = client.post(
        "/api/upload",
        files={"file": ("malicious_script.txt", fake_txt, "text/plain")},
    )
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]


def test_empty_file_upload_rejected():
    empty_buf = io.BytesIO(b"")
    res = client.post(
        "/api/upload",
        files={"file": ("empty.png", empty_buf, "image/png")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_corrupt_image_bytes_rejected():
    corrupt_buf = io.BytesIO(b"\x89PNG\r\n\x1a\nThis is corrupt junk data instead of IHDR chunks")
    res = client.post(
        "/api/upload",
        files={"file": ("corrupt.png", corrupt_buf, "image/png")},
    )
    assert res.status_code == 400
    assert "corrupted" in res.json()["detail"].lower() or "not a valid image" in res.json()["detail"].lower()


def test_oversized_file_rejected(monkeypatch):
    from src.core.config import settings
    # Temporarily set max upload to 1 MB for testing
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    large_buf = io.BytesIO(b"0" * (1024 * 1024 + 100))
    res = client.post(
        "/api/upload",
        files={"file": ("oversized.png", large_buf, "image/png")},
    )
    assert res.status_code == 413
    assert "exceeds maximum allowed limit" in res.json()["detail"]

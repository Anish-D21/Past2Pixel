"""
Comprehensive Test Suite for OCR Engine, Availability Reporting, Language Checking,
Endpoints (GET/POST /api/ocr, POST /api/ocr/direct), Pipeline Integration, and CER/WER Metrics.
"""
import io
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import settings
from src.db.mongodb import db_manager
from src.ocr.ocr_engine import ocr_engine, OCREngine

client = TestClient(app)


def test_ocr_status_structure():
    status_info = ocr_engine.get_status(force_refresh=True)
    assert "status" in status_info
    assert "is_installed" in status_info
    assert "devanagari_support" in status_info
    assert "reason" in status_info
    assert isinstance(status_info["available_languages"], list)
    if not status_info["is_installed"]:
        assert status_info["status"] == "UNAVAILABLE"
        assert status_info["setup_instructions"] is not None
        assert "download_url" in status_info["setup_instructions"]


def test_get_ocr_endpoints():
    """Verify that GET /api/ocr and GET /api/ocr/status return 200 with engine metadata."""
    res1 = client.get("/api/ocr")
    assert res1.status_code == 200
    data1 = res1.json()
    assert "status" in data1
    assert "is_installed" in data1
    assert "available_languages" in data1

    res2 = client.get("/api/ocr/status")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == data1["status"]


def test_ocr_session_not_found():
    res = client.post("/api/ocr", json={"session_id": "p2p_non_existent_999", "language": "san"})
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert "not found" in detail.lower()


def test_direct_ocr_upload():
    """Test POST /api/ocr/direct with an in-memory test image."""
    img = Image.new("RGB", (300, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "Past2Pixel", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/ocr/direct",
        files={"file": ("direct_test.png", buf, "image/png")},
        data={"language": "eng"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "character_count" in body
    assert body["image_source"] == "direct_upload"


def test_direct_ocr_empty_file():
    """Test POST /api/ocr/direct with empty payload returns 400 Bad Request."""
    empty_buf = io.BytesIO(b"")
    res = client.post(
        "/api/ocr/direct",
        files={"file": ("empty.png", empty_buf, "image/png")},
        data={"language": "eng"},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_ocr_pipeline_restored_image_priority(tmp_path):
    """
    Test that OCR on a session prefers restored_unet.png over original.png by default,
    and falls back to original.png when requested.
    """
    session_id = "test_ocr_pipeline_session"
    session_dir = settings.SESSIONS_STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create dummy original and restored images
        orig_img = Image.new("RGB", (100, 50), color=(200, 200, 200))
        rest_img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        orig_img.save(session_dir / "original.png")
        rest_img.save(session_dir / "restored_unet.png")

        # Create session in db
        db_manager.create_session({
            "session_id": session_id,
            "user_id": "anonymous",
            "status": "UPLOADED",
            "original_metadata": {},
        })

        # 1. Default (restored target)
        res_rest = client.post("/api/ocr", json={"session_id": session_id, "language": "eng"})
        assert res_rest.status_code == 200
        assert res_rest.json()["image_source"] == "restored_unet.png"

        # 2. Explicit original target
        res_orig = client.post(
            "/api/ocr",
            json={"session_id": session_id, "language": "eng", "target_image": "original"},
        )
        assert res_orig.status_code == 200
        assert res_orig.json()["image_source"] == "original.png"
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
        db_manager.delete_session(session_id, "anonymous")


def test_engine_uninstalled_diagnostic():
    """Verify OCREngine handles non-existent paths gracefully with UNAVAILABLE diagnostic."""
    dummy_engine = OCREngine(tesseract_cmd="C:/non_existent_path/tesseract.exe")
    status_rep = dummy_engine.get_status(force_refresh=True)
    assert status_rep["status"] == "UNAVAILABLE"
    assert status_rep["setup_instructions"] is not None

    # Test extract_text on uninstalled engine
    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    res = dummy_engine.extract_text(img)
    assert res["status"] == "UNAVAILABLE"
    assert res["is_authentic_ocr"] is False
    assert "not installed" in res["error"].lower()


def test_ocr_cer_wer_evaluation():
    engine = OCREngine()
    gt = "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्"
    pred_exact = "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्"
    eval_exact = engine.evaluate_ocr(pred_exact, gt)
    assert eval_exact["character_error_rate"] == 0.0
    assert eval_exact["word_error_rate"] == 0.0
    assert eval_exact["is_exact_match"] is True

    # Test with one word difference
    pred_diff = "अग्निमीळे पुरोहितम् यज्ञस्य देवमृत्विजम्"
    eval_diff = engine.evaluate_ocr(pred_diff, gt)
    assert eval_diff["character_error_rate"] > 0.0
    assert eval_diff["is_exact_match"] is False


def test_missing_language_pack_behavior():
    """Verify that requesting an uninstalled language pack returns MISSING_LANGUAGE_PACK."""
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 'xyz' is guaranteed not to exist
    res = client.post(
        "/api/ocr/direct",
        files={"file": ("test.png", buf, "image/png")},
        data={"language": "xyz_nonexistent_pack"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "MISSING_LANGUAGE_PACK"
    assert "xyz_nonexistent_pack" in data["error"]
    assert data["setup_instructions"] is not None


def test_supported_configurations_returned():
    """Verify status response includes supported_configurations list."""
    res = client.get("/api/ocr/status")
    assert res.status_code == 200
    data = res.json()
    assert "supported_configurations" in data
    assert isinstance(data["supported_configurations"], list)


def test_direct_ocr_with_preprocessing():
    """Verify that preprocessing options are accepted and reported in direct OCR."""
    img = Image.new("RGB", (150, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/ocr/direct",
        files={"file": ("test.png", buf, "image/png")},
        data={"language": "eng", "preprocessing": "contrast"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data.get("preprocessing_applied") == "contrast"


def test_unicode_nfc_preservation():
    """Verify Devanagari text is preserved in Unicode canonical NFC form."""
    import unicodedata
    # Construct NFD (decomposed) string
    nfd_text = unicodedata.normalize("NFD", "अग्निमीळे पुरोहितं")
    nfc_text = unicodedata.normalize("NFC", nfd_text)
    
    # Engine evaluation must normalize consistently
    engine = OCREngine()
    eval_res = engine.evaluate_ocr(nfd_text, nfc_text)
    assert eval_res["character_error_rate"] == 0.0
    assert eval_res["is_exact_match"] is True

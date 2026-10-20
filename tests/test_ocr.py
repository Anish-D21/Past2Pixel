"""
Test Suite for OCR Engine, Availability Reporting, Language Checking, and CER/WER Metrics.
"""
from fastapi.testclient import TestClient

from src.api.app import app
from src.ocr.ocr_engine import ocr_engine, OCREngine

client = TestClient(app)


def test_ocr_status_structure():
    status_info = ocr_engine.get_status()
    assert "status" in status_info
    assert "is_installed" in status_info
    assert "devanagari_support" in status_info
    assert "reason" in status_info
    if not status_info["is_installed"]:
        assert status_info["status"] == "UNAVAILABLE"
        assert status_info["setup_instructions"] is not None
        assert "download_url" in status_info["setup_instructions"]


def test_ocr_session_not_found():
    res = client.post("/api/ocr", json={"session_id": "p2p_non_existent_999", "language": "san"})
    assert res.status_code == 404


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

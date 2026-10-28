"""
Test Suite for Context-Aware Text Prediction, Uncertainty Labeling, and Scientific Integrity.
Verifies English and Devanagari missing words, incorrect words, damaged spans, and edge cases.
"""
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_prediction_output_and_labeling():
    payload = {
        "left_context": "धर्मक्षेत्रे",
        "right_context": "समवेता",
        "max_candidates": 3,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Mandatory Academic Integrity Assertions
    assert data["is_speculative"] is True
    assert data["uncertainty_label"] == "SPECULATIVE_CONTEXTUAL_COMPLETION"
    assert "never be presented as verified historical content" in data["disclaimer"]
    assert "status" in data
    assert isinstance(data["candidates"], list)


def test_prediction_empty_context_rejected():
    payload = {
        "left_context": "   ",
        "right_context": "",
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 400
    assert "At least one context span" in res.json()["detail"]


def test_prediction_with_full_text_extraction():
    """Verify that providing full_text and uncertain_span extracts context automatically."""
    full_text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
    payload = {
        "full_text": full_text,
        "uncertain_span": "कुरुक्षेत्रे",
        "max_candidates": 3,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["PREDICTED", "COMPLETED"]
    assert "धर्मक्षेत्रे" in data["surrounding_context"]["left_context"]
    assert "समवेता" in data["surrounding_context"]["right_context"]
    assert len(data["candidates"]) > 0
    first_cand = data["candidates"][0]
    assert "word" in first_cand
    assert "score" in first_cand
    assert first_cand.get("evidence") is not None


def test_prediction_english_missing_word():
    """Test Case 1: Missing word placeholder 'The king [MISSING] the temple.'"""
    payload = {
        "sentence": "The king [MISSING] the temple.",
        "max_candidates": 4,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PREDICTED"
    assert data["prediction"] in ["entered", "visited", "built", "protected"]
    assert data["reconstructed_sentence"] is not None
    assert "[MISSING]" not in data["reconstructed_sentence"]
    assert data["reconstructed_sentence"].startswith("The king")
    assert data["reconstructed_sentence"].endswith("the temple.")
    assert data["confidence_score"] > 0.0


def test_prediction_incorrect_extracted_word():
    """Test Case 2: Incorrect extracted word marked uncertain 'The king visited the market.'"""
    payload = {
        "sentence": "The king visited the market.",
        "uncertain_span": "visited",
        "max_candidates": 4,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PREDICTED"
    assert len(data["candidates"]) > 0
    words = [c["word"] for c in data["candidates"]]
    assert any(w in words for w in ["visited", "entered", "inspected", "arrived"])
    assert data["reconstructed_sentence"] is not None
    assert "the market." in data["reconstructed_sentence"]


def test_prediction_multi_word_damaged_span():
    """Test Case 3: Missing multiple words placeholder 'The king [DAMAGED] before sunrise.'"""
    payload = {
        "sentence": "The king [DAMAGED] before sunrise.",
        "uncertain_span": "[DAMAGED]",
        "max_candidates": 4,
        "allow_multi_word": True,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PREDICTED"
    assert data["prediction"] is not None
    assert "[DAMAGED]" not in data["reconstructed_sentence"]
    assert "before sunrise." in data["reconstructed_sentence"]


def test_prediction_devanagari_missing_span():
    """Test Case 4: Devanagari text with missing span 'धर्मक्षेत्रे [MISSING] समवेता युयुत्सवः'"""
    payload = {
        "sentence": "धर्मक्षेत्रे [MISSING] समवेता युयुत्सवः",
        "max_candidates": 4,
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PREDICTED"
    assert data["prediction"] == "कुरुक्षेत्रे"
    assert data["reconstructed_sentence"] == "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
    assert data["confidence_score"] > 0.5


def test_prediction_unsupported_sparse_input():
    """Test Case 5: Empty/unsupported sparse words with no reliable candidate."""
    payload = {
        "left_context": "qwerty12345nonexistent",
        "right_context": "zxcvbn98765unseen",
        "uncertain_span": "[MISSING]",
    }
    res = client.post("/api/predict-text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["INSUFFICIENT_CONTEXT", "LOW_CONFIDENCE"]
    # Does not claim false certainty
    if data["status"] == "INSUFFICIENT_CONTEXT":
        assert data["prediction"] is None
        assert data["confidence_score"] == 0.0

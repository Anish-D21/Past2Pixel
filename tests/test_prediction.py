"""
Test Suite for Context-Aware Text Prediction, Uncertainty Labeling, and Scientific Integrity.
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
    # Verify evidence structure
    first_cand = data["candidates"][0]
    assert "word" in first_cand
    assert "score" in first_cand
    assert first_cand.get("evidence") is not None

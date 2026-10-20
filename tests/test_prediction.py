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

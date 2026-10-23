"""
Test Suite for Text & Ink Protection, Conservative Traditional Baseline,
Reference-Free Quality Metrics, and Candidate Selection.
"""
import io
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from src.api.app import app
from src.pipeline.text_protection import text_protection_engine
from src.pipeline.traditional_baseline import traditional_restoration_engine
from src.pipeline.restoration_engine import restoration_engine
from src.metrics.restoration_metrics import (
    calculate_text_contrast,
    calculate_stroke_sharpness,
    calculate_stroke_preservation,
    calculate_background_variance,
    evaluate_candidate_quality,
)

client = TestClient(app)


def test_text_protection_segmentation():
    """Verify that text_protection_engine correctly segments dark text from parchment background."""
    img = Image.new("RGB", (300, 100), color=(240, 230, 210))
    d = ImageDraw.Draw(img)
    # Draw dark text strokes
    d.text((30, 40), "Manuscript Text Preservation", fill=(20, 15, 10))

    seg = text_protection_engine.segment_text_and_ink(img)
    assert "confident_ink_mask" in seg
    assert "background_mask" in seg
    assert "alpha_weight" in seg
    assert seg["ink_pixel_count"] > 0
    assert seg["ink_pixel_ratio"] > 0.001

    # Check non-overlap between confident ink and confident background
    assert not np.any(seg["confident_ink_mask"] & seg["background_mask"])


def test_traditional_baseline_restoration():
    """Verify that traditional_restoration_engine flattens uneven illumination without erasing text."""
    # Create image with uneven gradient
    w, h = 200, 100
    x = np.linspace(150, 240, w)
    grad = np.tile(x, (h, 1)).astype(np.uint8)
    img_arr = np.stack([grad, grad, grad], axis=-1)
    img = Image.fromarray(img_arr)
    d = ImageDraw.Draw(img)
    d.text((20, 40), "Rigveda Line", fill=(10, 10, 10))

    res = traditional_restoration_engine.restore_image(img)
    assert res["restored_pil"].size == (w, h)
    assert "mean_background_gain" in res
    assert res["method"] == "traditional_illumination_normalization"


def test_reference_free_quality_metrics():
    """Verify that contrast and stroke metrics reflect image quality correctly."""
    clean = Image.new("RGB", (200, 80), color=(250, 245, 235))
    d = ImageDraw.Draw(clean)
    d.text((20, 30), "Vedic Sanskrit", fill=(10, 10, 10))

    seg = text_protection_engine.segment_text_and_ink(clean)
    ink_mask = seg["confident_ink_mask"]

    eval_clean = evaluate_candidate_quality(clean, ink_mask)
    assert eval_clean["composite_score"] > 50.0
    assert eval_clean["text_contrast"] > 0.35
    assert eval_clean["stroke_preservation_ratio"] == 1.0

    # Test that a severely blurred version gets lower score and contrast
    blurred = clean.resize((30, 15), Image.Resampling.BILINEAR).resize((200, 80), Image.Resampling.BILINEAR)
    eval_blur = evaluate_candidate_quality(blurred, ink_mask)
    assert eval_blur["composite_score"] < eval_clean["composite_score"]
    assert eval_blur["text_contrast"] < eval_clean["text_contrast"]


def test_candidate_selection_in_restoration_engine():
    """Verify that process_image evaluates candidates and returns evidence-based selection."""
    img = Image.new("RGB", (160, 80), color=(235, 225, 205))
    d = ImageDraw.Draw(img)
    d.text((20, 30), "Devanagari Test", fill=(15, 10, 10))

    report = restoration_engine.process_image(img)
    assert "candidate_selection" in report
    sel = report["candidate_selection"]
    assert sel["selected_candidate"] in ["model_guided_restoration", "traditional_baseline", "original_preserved"]
    assert len(sel["selection_reason"]) > 10
    assert "candidates_evaluated" in sel
    assert "original" in sel["candidates_evaluated"]
    assert "traditional_baseline" in sel["candidates_evaluated"]
    assert "model_guided_restoration" in sel["candidates_evaluated"]


def test_api_restoration_candidate_and_artifacts():
    """Verify that /api/restore endpoint returns candidate selection and all artifacts."""
    img = Image.new("RGB", (128, 128), color=(240, 230, 215))
    d = ImageDraw.Draw(img)
    d.text((15, 50), "Sample Folio", fill=(20, 15, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_res = client.post("/api/upload", files={"file": ("candidate_test.png", buf, "image/png")})
    assert upload_res.status_code == 201
    sid = upload_res.json()["session_id"]

    res = client.post("/api/restore", json={"session_id": sid, "device_preference": "cpu"})
    assert res.status_code == 200
    data = res.json()

    assert data["artifacts"]["restored"] is not None
    assert data["artifacts"]["baseline"] is not None
    assert data["artifacts"]["model"] is not None
    assert data["artifacts"]["text_mask"] is not None
    assert "candidate_selection" in data
    assert data["candidate_selection"]["selected_candidate"] is not None


def test_ocr_compare_all_variants():
    """Verify that /api/ocr with target_image='compare_all' returns comparative OCR results."""
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Rigveda Folio", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    up = client.post("/api/upload", files={"file": ("ocr_comp_test.png", buf, "image/png")})
    sid = up.json()["session_id"]
    client.post("/api/restore", json={"session_id": sid, "device_preference": "cpu"})

    ocr_res = client.post("/api/ocr", json={"session_id": sid, "language": "eng", "target_image": "compare_all"})
    assert ocr_res.status_code == 200
    body = ocr_res.json()
    assert body["comparison_results"] is not None
    assert "original" in body["comparison_results"]
    assert "selected_restoration" in body["comparison_results"]

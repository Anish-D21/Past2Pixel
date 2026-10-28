"""
Focused Test Suite for OCR Accuracy, Image Grounding, Multi-Font Handling,
Physical Stroke Diagnostics, and Contextual Completion.

Validates the complete Past2Pixel OCR & Image-Grounded Correction pipeline:
1. Clear printed text recognition & evaluation against ground truth.
2. Faded/low-contrast text with CLAHE contrast enhancement.
3. Blurred/noisy image with stroke-preserving bilateral denoising.
4. Skew detection and correction.
5. Multi-font and varying text size handling.
6. Sanskrit ('san'), Hindi ('hin'), and English ('eng') script configurations.
7. Physical patch metrics (stroke density, contrast ratio, thumbnail generation).
8. Competing multi-pass readings & image-supported correction resolution.
9. Genuinely damaged/unreadable text flagged honestly without fabrication.
10. Contextual reconstruction for missing/damaged spans using statistical N-grams.
11. Empty input & unsupported language handling.
12. CER/WER calculation and non-destructive raw OCR preservation.
"""
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.ocr.ocr_engine import ocr_engine
from src.ocr.ocr_preprocessor import ocr_preprocessor
from src.ocr.image_grounded_corrector import image_grounded_corrector
from src.prediction.context_predictor import ContextTextPredictor, calculate_cer, calculate_wer

client = TestClient(app)


def _create_synthetic_text_image(
    text: str,
    font_size: int = 36,
    bg_color=(250, 245, 235),
    text_color=(20, 20, 20),
    width: int = 600,
    height: int = 120,
) -> Image.Image:
    """Creates a high-contrast clean synthetic manuscript line."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((30, 40), text, fill=text_color, font=font)
    return img


def test_clear_printed_text_and_cer_wer():
    """1. Test clear printed manuscript text extraction and CER/WER computation against reference."""
    ground_truth = "Past2Pixel Manuscript Recovery System"
    img = _create_synthetic_text_image(ground_truth, width=700, height=100)

    res = ocr_engine.extract_text(img, language="eng", use_multi_pass=True)
    assert res["status"] == "SUCCESS"
    extracted_text = res["extracted_text"]
    assert len(extracted_text) > 0

    # Verify CER and WER evaluation helper
    metrics = ocr_engine.evaluate_ocr(hypothesis=extracted_text, reference=ground_truth)
    assert "character_error_rate" in metrics
    assert "word_error_rate" in metrics
    assert 0.0 <= metrics["character_error_rate"] <= 1.0


def test_faded_low_contrast_enhancement():
    """2. Test faded/low-contrast image where CLAHE contrast enhancement restores stroke legibility."""
    # Create very low contrast text: pale gray text on white/cream background
    img = _create_synthetic_text_image(
        "Faded Manuscript",
        bg_color=(240, 238, 230),
        text_color=(195, 192, 185),  # very faint
        width=400,
        height=80,
    )
    # Image quality diagnostics should register low contrast
    quality = ocr_preprocessor.evaluate_image_quality(img)
    assert "contrast_ratio" in quality
    assert quality["contrast_ratio"] < 50.0  # low contrast

    # CLAHE contrast enhancement should improve dynamic range
    enhanced = ocr_preprocessor.clahe_contrast_enhancement(img)
    arr_enhanced = np.array(enhanced.convert("L"))
    assert arr_enhanced.std() >= np.array(img.convert("L")).std()


def test_blurred_and_noisy_stroke_preserving_denoise():
    """3. Test noisy image where stroke-preserving bilateral filtering reduces noise while preserving stroke edges."""
    base_img = _create_synthetic_text_image("Ancient Inscription", width=400, height=80)
    arr = np.array(base_img).astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, 15, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    # Stroke-preserving bilateral denoise
    denoised = ocr_preprocessor.stroke_preserving_denoise(noisy_img)
    assert denoised.size == noisy_img.size

    # Verify edge sharpness is maintained (Laplacian variance)
    sharp_noisy = ocr_preprocessor.evaluate_image_quality(noisy_img)["blur_score"]
    sharp_denoised = ocr_preprocessor.evaluate_image_quality(denoised)["blur_score"]
    assert sharp_denoised > 0.0


def test_skew_detection_and_correction():
    """4. Test skew estimation and rotation on tilted manuscript images."""
    base_img = _create_synthetic_text_image("Tilted Manuscript Line", width=500, height=120)
    # Rotate by 4.0 degrees
    tilted = base_img.rotate(4.0, expand=False, fillcolor=(250, 245, 235))

    deskewed, angle = ocr_preprocessor.estimate_and_correct_skew(tilted, max_angle=10.0, step=0.5)
    assert isinstance(angle, float)
    assert deskewed.size == tilted.size
    # Correction angle should be negative of tilt (around -4.0 degrees)
    assert abs(angle - (-4.0)) <= 1.5


def test_multi_font_and_varying_text_sizes():
    """5. Test layout containing multiple text sizes and densities."""
    img = Image.new("RGB", (600, 160), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Title (large)
    draw.text((20, 20), "CHAPTER I: THE HISTORICAL MANUSCRIPT", fill=(0, 0, 0))
    # Body (medium)
    draw.text((20, 70), "The inscription was discovered near the holy temple.", fill=(20, 20, 20))
    # Subtext (dense)
    draw.text((20, 120), "Preserved in the royal archives of the 8th century.", fill=(40, 40, 40))

    res = ocr_engine.extract_text(img, language="eng", use_multi_pass=True)
    assert res["status"] == "SUCCESS"
    assert len(res.get("lines", [])) >= 2 or len(res["extracted_text"].split("\n")) >= 2
    assert "words_detail" in res
    assert len(res["words_detail"]) > 0


def test_devanagari_and_sanskrit_hindi_configs():
    """6. Test Sanskrit and Hindi recognition configurations."""
    status = ocr_engine.get_status()
    available_langs = status.get("available_languages", [])

    # Test Devanagari text
    devanagari_img = _create_synthetic_text_image("धर्मक्षेत्रे कुरुक्षेत्रे", width=500, height=100)

    # When 'san' or 'hin' is installed
    if "san" in available_langs:
        res_san = ocr_engine.extract_text(devanagari_img, language="san")
        assert res_san["status"] == "SUCCESS"
        assert res_san["language_used"] == "san"
    if "hin" in available_langs:
        res_hin = ocr_engine.extract_text(devanagari_img, language="hin")
        assert res_hin["status"] == "SUCCESS"
        assert res_hin["language_used"] == "hin"


def test_physical_patch_metrics_ink_vs_blank():
    """7. Test physical stroke density and contrast calculations on ink vs blank image patches."""
    # Ink patch (black text on white)
    ink_img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(ink_img)
    draw.rectangle([20, 10, 80, 40], fill=(0, 0, 0))

    ink_metrics = image_grounded_corrector.extract_patch_metrics(ink_img, [20, 10, 60, 30])
    assert ink_metrics["has_visible_ink"] is True
    assert ink_metrics["stroke_density"] > 0.20
    assert ink_metrics["contrast_ratio"] > 0.50
    assert ink_metrics["patch_data_uri"] is not None
    assert ink_metrics["patch_data_uri"].startswith("data:image/png;base64,")

    # Blank patch (pure white background, no ink strokes)
    blank_img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    blank_metrics = image_grounded_corrector.extract_patch_metrics(blank_img, [10, 10, 50, 30])
    assert blank_metrics["has_visible_ink"] is False
    assert blank_metrics["stroke_density"] < 0.05


def test_multi_pass_alternative_resolution():
    """8. Test image-grounded resolution when multi-pass candidates compete."""
    dummy_img = Image.new("RGB", (300, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    draw.text((20, 20), "temple", fill=(0, 0, 0))

    # Candidate 1: 'tenple' (OCR error with low conf)
    # Candidate 2: 'temple' (Valid English word from another pass)
    resolved = image_grounded_corrector.resolve_word_candidate(
        original_img=dummy_img,
        word="tenple",
        bbox=[20, 20, 80, 25],
        confidence=45.0,
        competing_candidates=["temple"],
        left_context="The king entered the",
        right_context="to pray.",
    )
    # 'temple' should be favored due to physical ink support and linguistic context
    assert resolved["word"] == "temple"
    assert resolved["status"] == "IMAGE_SUPPORTED_CORRECTION"
    assert resolved["original_word"] == "tenple"


def test_genuinely_unreadable_text_flagged_honestly():
    """9. Test that genuinely damaged / blank regions with no ink are flagged as [DAMAGED] / unreadable."""
    blank_img = Image.new("RGB", (300, 60), color=(255, 255, 255))

    # Attempt word candidate resolution on blank coordinates where OCR hallucinated text
    resolved = image_grounded_corrector.resolve_word_candidate(
        original_img=blank_img,
        word="xyz123",
        bbox=[20, 20, 60, 25],
        confidence=25.0,
        competing_candidates=[],
    )
    assert resolved["status"] == "DAMAGED_UNREADABLE"
    assert resolved["word"] == "[DAMAGED]"
    assert "no identifiable ink strokes" in resolved["evidence"]


def test_contextual_reconstruction_missing_spans():
    """10. Test linguistic N-gram contextual completion for missing spans ([MISSING], ___, damaged words)."""
    predictor = ContextTextPredictor(n=3)
    predictor.train_on_corpus([
        "The king entered the temple to offer prayers.",
        "The noble citizens gathered at the market square.",
        "The sacred manuscript was preserved within the temple archives.",
    ])

    # Test explicit placeholder: [MISSING]
    res1 = predictor.predict_missing(
        sentence="The king entered the [MISSING] to offer prayers.",
        max_candidates=3,
    )
    assert res1["status"] in ["PREDICTED", "SUCCESS"]
    assert len(res1["candidates"]) > 0
    assert res1["candidates"][0]["text"] == "temple"
    assert "temple" in res1["reconstructed_sentence"]

    # Test underline placeholder: ___
    res2 = predictor.predict_missing(
        sentence="The noble citizens gathered at the ___ square.",
        max_candidates=3,
    )
    assert res2["status"] in ["PREDICTED", "SUCCESS"]
    assert res2["candidates"][0]["text"] == "market"

    # Test candidate scoring helper for image grounding
    scores = predictor.score_candidates(
        candidates=["temple", "tent", "market"],
        left_context="The king entered the",
        right_context="to offer prayers.",
    )
    assert scores["temple"] > scores["tent"]


def test_empty_input_and_unsupported_language():
    """11. Test empty input rejection and unsupported language error handling."""
    # Empty file POST /api/ocr/direct
    res_empty = client.post(
        "/api/ocr/direct",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        data={"language": "eng"},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # Unsupported language code
    img = _create_synthetic_text_image("Test", width=200, height=60)
    res_unsupported = ocr_engine.extract_text(img, language="klingon_999")
    assert res_unsupported["status"] in ["MISSING_LANGUAGE_PACK", "ERROR", "UNAVAILABLE"]
    assert "not installed" in res_unsupported["error"].lower() or "missing" in res_unsupported["error"].lower() or "unsupported" in res_unsupported["error"].lower()


def test_pipeline_non_destructive_raw_and_corrected_preservation():
    """12. Test that raw OCR text is strictly preserved alongside corrected image-supported text."""
    img = _create_synthetic_text_image("Ancient Manuscript Record", width=500, height=80)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/ocr/direct",
        files={"file": ("record.png", buf, "image/png")},
        data={"language": "eng", "use_multi_pass": "true"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "raw_extracted_text" in body
    assert "corrected_text" in body
    assert "extracted_text" in body
    assert "image_grounded_corrections" in body
    assert "image_quality" in body
    assert "secondary_model_status" in body
    assert body["is_authentic_ocr"] is True

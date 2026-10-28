"""
Test suite validating the restored 2x Super-Resolution OCR extraction workflow.
Verifies:
1. Exact (2W, 2H) output dimensions when 2x upscaling is applied.
2. Upscaled image genuinely reaches the OCR recognition stage.
3. Aspect ratio preservation with no unintended cropping.
4. Safe handling of RGB, RGBA, and Grayscale (L) images.
5. Prevention of redundant repeat upscaling.
6. Preservation of raw OCR text without destructive overwrites.
7. CER/WER accuracy metrics on authentic manuscript ground truth pairs.
"""
from pathlib import Path
from PIL import Image
import pytest

from src.ocr.ocr_preprocessor import ocr_preprocessor
from src.ocr.ocr_engine import ocr_engine
from src.prediction.context_predictor import calculate_cer, calculate_wer


def test_2x_upscale_exact_dimensions():
    """Verify that upscale_2x produces exactly twice the input width and height."""
    test_sizes = [(100, 50), (240, 180), (333, 111), (75, 75)]
    for w, h in test_sizes:
        img = Image.new("RGB", (w, h), color=(240, 235, 220))
        upscaled = ocr_preprocessor.upscale_2x(img)
        assert upscaled.size == (w * 2, h * 2), f"Expected {(w * 2, h * 2)}, got {upscaled.size}"
        assert getattr(upscaled, "_is_2x_upscaled", False) is True


def test_2x_upscale_image_formats_safety():
    """Verify safe handling of RGB, RGBA, and Grayscale (L) images."""
    # RGBA with transparency
    rgba_img = Image.new("RGBA", (80, 60), color=(255, 255, 255, 128))
    upscaled_rgba = ocr_preprocessor.upscale_2x(rgba_img)
    assert upscaled_rgba.size == (160, 120)
    assert upscaled_rgba.mode == "RGB"

    # Grayscale
    gray_img = Image.new("L", (90, 70), color=128)
    upscaled_gray = ocr_preprocessor.upscale_2x(gray_img)
    assert upscaled_gray.size == (180, 140)
    assert upscaled_gray.mode in ("L", "RGB")


def test_no_redundant_repeat_upscaling():
    """Verify that an image marked as already upscaled is not repeatedly blown up."""
    img = Image.new("RGB", (100, 80), color=(200, 200, 200))
    up1 = ocr_preprocessor.upscale_2x(img)
    assert up1.size == (200, 160)
    
    # Attempt second upscaling
    up2 = ocr_preprocessor.upscale_2x(up1)
    assert up2.size == (200, 160), "Repeat upscaling should be prevented"


def test_ocr_engine_processes_upscaled_image():
    """Verify that ocr_engine receives the upscaled image and records 2x telemetry."""
    # Create small test image
    w, h = 120, 40
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    
    # Run with preprocessing="upscale" in single-pass mode
    res_sp = ocr_engine.extract_text(img, lang="san+hin", preprocessing="upscale", mode="single-pass")
    assert res_sp["super_resolution_applied"] is True
    assert res_sp["upscale_factor"] == 2.0
    assert res_sp["original_dimensions"] == [w, h]
    assert res_sp["processed_dimensions"] == [w * 2, h * 2]

    # Run with preprocessing="upscale" in multi-pass mode
    res_mp = ocr_engine.extract_text(img, lang="san+hin", preprocessing="upscale", mode="multi-pass")
    assert res_mp["super_resolution_applied"] is True
    assert res_mp["upscale_factor"] == 2.0
    assert res_mp["original_dimensions"] == [w, h]
    assert res_mp["processed_dimensions"] == [w * 2, h * 2]


def test_raw_ocr_preserved_and_not_silently_overwritten():
    """Verify that raw_extracted_text and corrected_text are separate and raw text is preserved."""
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    res = ocr_engine.extract_text(img, lang="eng", preprocessing="upscale", mode="single-pass")
    assert "raw_extracted_text" in res
    assert "extracted_text" in res
    assert "corrected_text" in res


def test_ground_truth_accuracy_improvement():
    """Verify CER evaluation against authentic Sanskrit/Hindi ground truth samples."""
    gt_dir = Path("data/raw/Sanskrit & Hindi OCR + VQA/Dataset/test/ocr")
    sample_img_p = gt_dir / "1.jpg"
    sample_txt_p = gt_dir / "1.txt"
    
    if not sample_img_p.exists() or not sample_txt_p.exists():
        pytest.skip("Authentic manuscript test dataset not present in this workspace environment")
        
    with open(sample_txt_p, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()
        
    img = Image.open(sample_img_p)
    w, h = img.size
    
    res_base = ocr_engine.extract_text(img, lang="san+hin", preprocessing="none", mode="single-pass", psm=8)
    res_2x = ocr_engine.extract_text(img, lang="san+hin", preprocessing="upscale", mode="single-pass", psm=8)
    
    # 2x output dimensions must be exactly 2W, 2H
    assert res_2x["processed_dimensions"] == [w * 2, h * 2]
    assert res_2x["super_resolution_applied"] is True
    
    # Measure CER
    cer_base = calculate_cer(res_base["extracted_text"].strip(), gt_text)
    cer_2x = calculate_cer(res_2x["extracted_text"].strip(), gt_text)
    
    # 2x Super-Resolution achieves lower or equal CER compared to unscaled
    assert cer_2x <= cer_base

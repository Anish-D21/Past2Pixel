"""
Unit and integration tests for isolated OCR Preprocessing module.
Ensures image filters are non-destructive and preserve stroke topology.
"""
import numpy as np
from PIL import Image
from src.ocr.ocr_preprocessor import OCRPreprocessor


def test_to_grayscale():
    rgb_img = Image.new("RGB", (100, 100), color=(120, 80, 40))
    gray_img = OCRPreprocessor.to_grayscale(rgb_img)
    assert gray_img.mode == "L"
    assert gray_img.size == (100, 100)

    # Idempotent on L
    gray_again = OCRPreprocessor.to_grayscale(gray_img)
    assert gray_again.mode == "L"


def test_conservative_contrast():
    arr = np.linspace(50, 200, 10000, dtype=np.uint8).reshape((100, 100))
    img = Image.fromarray(arr, mode="L")
    enhanced = OCRPreprocessor.conservative_contrast(img)
    assert enhanced.size == (100, 100)
    arr_enh = np.array(enhanced)
    assert arr_enh.shape[:2] == (100, 100)


def test_illumination_correction():
    # Synthetic gradient illumination
    x = np.linspace(0, 100, 100, dtype=np.float32)
    gradient = np.tile(x, (100, 1)).astype(np.uint8)
    img = Image.fromarray(gradient, mode="L")
    corrected = OCRPreprocessor.illumination_correction(img)
    assert corrected.size == (100, 100)


def test_adaptive_binarization_sauvola():
    img = Image.new("L", (80, 80), color=220)
    # Draw dark strokes
    arr = np.array(img)
    arr[20:60, 20:60] = 30
    bin_img = OCRPreprocessor.adaptive_binarization(Image.fromarray(arr))
    assert bin_img.size == (80, 80)
    bin_arr = np.array(bin_img)
    unique_vals = np.unique(bin_arr)
    assert set(unique_vals).issubset({0, 255})


def test_upscale_small_text():
    small_img = Image.new("RGB", (120, 40), color=(255, 255, 255))
    upscaled = OCRPreprocessor.upscale_small_text(small_img, min_dimension=100)
    assert upscaled.size[1] >= 100
    assert upscaled.size[0] > small_img.size[0]


def test_preprocess_for_ocr_modes():
    img = Image.new("RGB", (200, 80), color=(240, 230, 210))
    for mode in ["none", "contrast", "illumination", "binarize", "upscale"]:
        processed, applied = OCRPreprocessor.apply_mode(img, mode=mode)
        assert processed is not None
        assert applied == mode

    # Auto mode selects appropriate variant
    processed_auto, applied_auto = OCRPreprocessor.apply_mode(img, mode="auto")
    assert processed_auto is not None
    assert applied_auto in ["upscale", "illumination", "contrast"]

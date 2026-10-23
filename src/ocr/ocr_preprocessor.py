"""
src/ocr/ocr_preprocessor.py
Controlled, isolated image preprocessing pipeline for Tesseract OCR.
Generates conservative candidate variants to optimize character recognition without
modifying, overwriting, or destabilizing manuscript restoration artifacts.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, grey_dilation
import unicodedata
import logging

logger = logging.getLogger(__name__)


class OCRPreprocessor:
    """
    Image preprocessing specifically designed for Vedic Sanskrit, Hindi, and English manuscript OCR.
    Applies non-destructive, conservative transforms to improve glyph separation and contrast.
    """

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """Converts RGB/RGBA image safely to single-channel 8-bit luminance."""
        if image.mode != "L":
            return image.convert("L")
        return image

    @staticmethod
    def illumination_correction(image: Image.Image) -> Image.Image:
        """
        Removes uneven background illumination, parchment shadows, and stains
        using morphological background surface estimation and division normalization.
        """
        gray = OCRPreprocessor.to_grayscale(image)
        arr = np.array(gray, dtype=np.float32)

        # Estimate illumination background surface
        struct_size = max(15, min(arr.shape) // 12)
        dilated = grey_dilation(arr, size=(struct_size, struct_size))
        background = gaussian_filter(dilated, sigma=struct_size / 2.0)
        background = np.maximum(background, 1.0)

        # Division normalization: flatten non-uniform background to high intensity (240)
        normalized = (arr / background) * 240.0
        normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(normalized, mode="L").convert("RGB")

    @staticmethod
    def conservative_contrast(image: Image.Image) -> Image.Image:
        """
        Enhances ink-to-background contrast using histogram stretching and mild unsharp masking.
        Preserves thin strokes without clipping fine Devanagari matras.
        """
        gray = OCRPreprocessor.to_grayscale(image)
        arr = np.array(gray, dtype=np.float32)

        # Percentile stretching (2nd to 98th percentile)
        p2 = np.percentile(arr, 2.0)
        p98 = np.percentile(arr, 98.0)
        if p98 > p2:
            stretched = (arr - p2) / (p98 - p2) * 255.0
            stretched = np.clip(stretched, 0.0, 255.0).astype(np.uint8)
        else:
            stretched = arr.astype(np.uint8)

        pil_stretched = Image.fromarray(stretched, mode="L")
        # Mild unsharp mask for stroke edge definition
        sharpened = pil_stretched.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        return sharpened.convert("RGB")

    @staticmethod
    def adaptive_binarization(image: Image.Image) -> Image.Image:
        """
        Applies Sauvola adaptive thresholding specifically calibrated for manuscript ink strokes.
        Calculates local mean and standard deviation: T = m * (1 + k * (s / R - 1)).
        """
        gray = OCRPreprocessor.to_grayscale(image)
        arr = np.array(gray, dtype=np.float32)

        window_size = max(11, min(arr.shape) // 20)
        if window_size % 2 == 0:
            window_size += 1

        sigma = window_size / 3.0
        local_mean = gaussian_filter(arr, sigma=sigma)
        local_sq_mean = gaussian_filter(arr ** 2, sigma=sigma)
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        local_std = np.sqrt(local_var)

        # Sauvola formula: k=0.18, R=128
        k = 0.18
        R = 128.0
        threshold = local_mean * (1.0 + k * (local_std / R - 1.0))

        binarized = np.where(arr < threshold, 0, 255).astype(np.uint8)
        return Image.fromarray(binarized, mode="L").convert("RGB")

    @staticmethod
    def upscale_small_text(image: Image.Image, min_dimension: int = 120) -> Image.Image:
        """
        Upscales small line crops or low-DPI scans where stroke width < 2px.
        Tesseract performs significantly better when x-height >= 30-35 pixels.
        """
        w, h = image.size
        if min(w, h) < min_dimension:
            scale = float(min_dimension) / max(min(w, h), 1)
            scale = min(scale, 2.5)  # Cap scale at 2.5x to prevent excessive memory
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Gentle unsharp mask to restore edge crispness after interpolation
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=2))
            return upscaled
        return image

    @classmethod
    def generate_candidate_variants(cls, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Generates controlled set of preprocessing candidates for a given manuscript image.
        """
        rgb_img = image.convert("RGB")
        candidates = {
            "none": rgb_img,
            "contrast": cls.conservative_contrast(rgb_img),
            "illumination": cls.illumination_correction(rgb_img),
            "binarize": cls.adaptive_binarization(rgb_img),
        }

        # If dimensions are small, also include upscaled variant
        if min(image.size) < 140:
            candidates["upscale"] = cls.upscale_small_text(candidates["contrast"])

        return candidates

    @classmethod
    def apply_mode(
        cls,
        image: Image.Image,
        mode: str = "auto",
    ) -> Tuple[Image.Image, str]:
        """
        Applies requested preprocessing mode or selects the safest variant automatically.
        Supported modes: 'none', 'contrast', 'illumination', 'binarize', 'upscale', 'auto'.
        """
        clean_mode = (mode or "auto").lower().strip()
        rgb_img = image.convert("RGB")

        if clean_mode == "none":
            return rgb_img, "none"
        elif clean_mode == "contrast":
            return cls.conservative_contrast(rgb_img), "contrast"
        elif clean_mode == "illumination":
            return cls.illumination_correction(rgb_img), "illumination"
        elif clean_mode == "binarize":
            return cls.adaptive_binarization(rgb_img), "binarize"
        elif clean_mode == "upscale":
            return cls.upscale_small_text(cls.conservative_contrast(rgb_img)), "upscale"
        elif clean_mode == "auto":
            # Heuristic selection based on image properties:
            # 1. If very small height (< 120), upscale with contrast enhancement
            if min(rgb_img.size) < 120:
                return cls.upscale_small_text(cls.conservative_contrast(rgb_img)), "upscale"

            # 2. Measure background variance to detect uneven staining
            gray_arr = np.array(cls.to_grayscale(rgb_img), dtype=np.float32)
            bg_mask = gray_arr > np.median(gray_arr)
            bg_std = float(gray_arr[bg_mask].std()) if bg_mask.any() else 0.0

            if bg_std > 28.0:
                # Strong uneven staining: illumination flattening is most beneficial
                return cls.illumination_correction(rgb_img), "illumination"
            else:
                # Moderate/clean background: conservative contrast enhancement
                return cls.conservative_contrast(rgb_img), "contrast"
        else:
            return rgb_img, "none"


# Global singleton instance
ocr_preprocessor = OCRPreprocessor()

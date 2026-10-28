"""
src/ocr/ocr_preprocessor.py
Controlled, isolated image preprocessing pipeline for Tesseract OCR.
Generates conservative candidate variants to optimize character recognition without
modifying, overwriting, or destabilizing manuscript restoration artifacts.
Includes deskewing, stroke-preserving CLAHE contrast, bilateral denoising,
illumination flattening, and physical image quality diagnostics.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, grey_dilation
import unicodedata
import logging

logger = logging.getLogger(__name__)


class OCRPreprocessor:
    """
    Image preprocessing specifically designed for Vedic Sanskrit, Hindi, and English manuscript OCR.
    Applies non-destructive, conservative transforms to improve glyph separation and contrast
    without damaging fine Devanagari matras, conjuncts, or historical ink strokes.
    """

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """Converts RGB/RGBA image safely to single-channel 8-bit luminance."""
        if image.mode != "L":
            return image.convert("L")
        return image

    @staticmethod
    def clahe_contrast_enhancement(image: Image.Image, clip_limit: float = 2.2, grid_size: int = 8) -> Image.Image:
        """
        Enhances local contrast of faded ink strokes using Contrast Limited Adaptive Histogram Equalization.
        Operates in LAB color space on luminance channel, preserving natural parchment color and stroke fidelity.
        """
        rgb_arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)

    @staticmethod
    def stroke_preserving_denoise(image: Image.Image) -> Image.Image:
        """
        Applies bilateral filtering to suppress background parchment grain and speckle
        while strictly preserving high-gradient ink stroke boundaries and fine matras.
        """
        rgb_arr = np.array(image.convert("RGB"))
        # Bilateral filter smooths flat textures while preserving sharp stroke edges
        denoised = cv2.bilateralFilter(rgb_arr, d=5, sigmaColor=35, sigmaSpace=35)
        return Image.fromarray(denoised)

    @staticmethod
    def estimate_and_correct_skew(
        image: Image.Image, max_angle: float = 12.0, step: float = 0.5
    ) -> Tuple[Image.Image, float]:
        """
        Estimates skew angle of manuscript text lines using horizontal projection profile variance,
        and deskews the image using high-fidelity bicubic interpolation with white border fill.
        Returns the deskewed PIL image and the detected skew angle in degrees.
        """
        gray = OCRPreprocessor.to_grayscale(image)
        w, h = gray.size
        # Downscale for fast and robust skew search if image is very large
        search_scale = min(1.0, 800.0 / max(w, h))
        if search_scale < 1.0:
            sw, sh = int(w * search_scale), int(h * search_scale)
            search_gray = gray.resize((sw, sh), Image.Resampling.BILINEAR)
        else:
            search_gray = gray

        arr = np.array(search_gray, dtype=np.uint8)
        # Invert so ink strokes are foreground
        thresh = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )

        # If image has negligible ink content, no skew rotation
        ink_pixels = int(np.count_nonzero(thresh))
        if ink_pixels < 30:
            return image.convert("RGB"), 0.0

        angles = np.arange(-max_angle, max_angle + step, step)
        best_angle = 0.0
        max_variance = -1.0
        center = (thresh.shape[1] // 2, thresh.shape[0] // 2)

        for a in angles:
            M = cv2.getRotationMatrix2D(center, float(a), 1.0)
            rotated = cv2.warpAffine(
                thresh, M, (thresh.shape[1], thresh.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            # Horizontal projection profile: sum across columns
            proj = np.sum(rotated, axis=1)
            var = float(np.var(proj))
            if var > max_variance:
                max_variance = var
                best_angle = float(a)

        # Only apply rotation if detected skew is significant (>= 0.5 deg)
        if abs(best_angle) >= 0.5 and max_variance > 10.0:
            full_rgb = np.array(image.convert("RGB"))
            fh, fw = full_rgb.shape[:2]
            f_center = (fw // 2, fh // 2)
            M_full = cv2.getRotationMatrix2D(f_center, best_angle, 1.0)
            corrected_rgb = cv2.warpAffine(
                full_rgb, M_full, (fw, fh),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
            return Image.fromarray(corrected_rgb), best_angle

        return image.convert("RGB"), 0.0

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
    def upscale_2x(image: Image.Image, method: str = "lanczos") -> Image.Image:
        """
        Genuinely upscales the input manuscript image to exactly 2x width and 2x height (2W x 2H).
        Uses high-fidelity Lanczos/Bicubic resampling with calibrated unsharp masking for stroke definition.
        Handles RGB, RGBA (composite over white background), and L (grayscale) safely.
        Avoids excessive ringing or halo artifacts that distort fine character strokes.
        """
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            img_to_scale = bg
        elif image.mode not in ("RGB", "L"):
            img_to_scale = image.convert("RGB")
        else:
            img_to_scale = image

        w, h = img_to_scale.size
        # Guard against redundant multiple 2x upscaling if image is already huge (>= 3500px)
        if getattr(image, "_is_2x_upscaled", False) or max(w, h) >= 3500:
            return img_to_scale

        target_w = w * 2
        target_h = h * 2

        resample_mode = Image.Resampling.LANCZOS if method == "lanczos" else Image.Resampling.BICUBIC
        upscaled = img_to_scale.resize((target_w, target_h), resample_mode)

        # Gentle unsharp mask calibrated to sharpen character strokes without ringing/halos
        # Radius 1.0, 75% unsharp, threshold 2 preserves fine Sanskrit/Devanagari matras
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.0, percent=75, threshold=2))
        try:
            sharpened._is_2x_upscaled = True  # type: ignore[attr-defined]
        except Exception:
            pass
        return sharpened

    @staticmethod
    def upscale_small_text(image: Image.Image, min_dimension: int = 140) -> Image.Image:
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
    def evaluate_image_quality(cls, image: Image.Image) -> Dict[str, Any]:
        """
        Computes physical image quality metrics:
        - Contrast ratio (p95 - p5) / 255
        - Blur score (Laplacian variance)
        - Skew angle (in degrees)
        - Background non-uniformity standard deviation
        - Recommended preprocessing mode based on measured physical characteristics
        """
        gray = cls.to_grayscale(image)
        arr = np.array(gray, dtype=np.uint8)

        # 1. Blur score via Laplacian variance
        lap = cv2.Laplacian(arr, cv2.CV_64F)
        blur_score = round(float(lap.var()), 2)

        # 2. Contrast ratio
        p5 = float(np.percentile(arr, 5.0))
        p95 = float(np.percentile(arr, 95.0))
        contrast_ratio = round((p95 - p5) / 255.0, 3)

        # 3. Background non-uniformity
        bg_mask = arr > np.median(arr)
        bg_std = round(float(arr[bg_mask].std()), 2) if bg_mask.any() else 0.0

        # 4. Skew angle
        _, skew_deg = cls.estimate_and_correct_skew(image, max_angle=10.0)

        # Determine recommendation
        if min(image.size) < 140:
            rec_mode = "upscale"
        elif abs(skew_deg) >= 2.0:
            rec_mode = "deskew"
        elif bg_std > 30.0:
            rec_mode = "illumination"
        elif contrast_ratio < 0.35:
            rec_mode = "clahe"
        else:
            rec_mode = "contrast"

        return {
            "blur_score": blur_score,
            "contrast_ratio": contrast_ratio,
            "skew_angle_deg": skew_deg,
            "background_std": bg_std,
            "width": image.width,
            "height": image.height,
            "recommended_mode": rec_mode,
        }

    @classmethod
    def generate_candidate_variants(cls, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Generates controlled set of preprocessing candidates for a given manuscript image.
        Never overwrites the original image.
        """
        rgb_img = image.convert("RGB")
        deskewed_img, _ = cls.estimate_and_correct_skew(rgb_img)

        candidates = {
            "none": rgb_img,
            "deskew": deskewed_img,
            "contrast": cls.conservative_contrast(deskewed_img),
            "clahe": cls.clahe_contrast_enhancement(deskewed_img),
            "illumination": cls.illumination_correction(deskewed_img),
            "binarize": cls.adaptive_binarization(deskewed_img),
            "denoised": cls.stroke_preserving_denoise(deskewed_img),
        }

        # Always include genuine 2x Super-Resolution candidate
        candidates["upscale"] = cls.upscale_2x(deskewed_img)

        return candidates

    @classmethod
    def apply_mode(
        cls,
        image: Image.Image,
        mode: str = "auto",
    ) -> Tuple[Image.Image, str]:
        """
        Applies requested preprocessing mode or selects the safest variant automatically.
        Supported modes: 'none', 'contrast', 'clahe', 'illumination', 'binarize', 'upscale', 'deskew', 'auto'.
        """
        clean_mode = (mode or "auto").lower().strip()
        rgb_img = image.convert("RGB")

        if clean_mode == "none":
            return rgb_img, "none"
        elif clean_mode == "contrast":
            return cls.conservative_contrast(rgb_img), "contrast"
        elif clean_mode == "clahe":
            return cls.clahe_contrast_enhancement(rgb_img), "clahe"
        elif clean_mode == "deskew":
            deskewed, _ = cls.estimate_and_correct_skew(rgb_img)
            return deskewed, "deskew"
        elif clean_mode == "illumination":
            return cls.illumination_correction(rgb_img), "illumination"
        elif clean_mode == "binarize":
            return cls.adaptive_binarization(rgb_img), "binarize"
        elif clean_mode == "upscale":
            return cls.upscale_2x(rgb_img), "upscale"
        elif clean_mode == "auto":
            quality = cls.evaluate_image_quality(rgb_img)
            rec = quality["recommended_mode"]
            return cls.apply_mode(rgb_img, mode=rec)
        else:
            return rgb_img, "none"


# Global singleton instance
ocr_preprocessor = OCRPreprocessor()

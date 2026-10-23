"""
Text & Ink Protection Engine for Ancient Manuscript Restoration.
Prevents deep learning models and denoising filters from erasing, blurring,
or distorting historical handwritten strokes and Devanagari ligatures.

Employs:
- Adaptive Multi-Scale Sauvola Thresholding
- Sobel Edge Gradient Magnitude Confirmation
- Confident Ink vs Background vs Uncertain Region Partitioning
- Sub-pixel Gaussian Boundary Alpha Blending
"""
from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
from scipy import ndimage


class TextProtectionEngine:
    """
    Identifies and protects likely ink strokes in ancient manuscripts.
    Ensures that high-resolution original text evidence is never overwritten by blurry model approximations.
    """

    def __init__(
        self,
        sauvola_k: float = 0.22,
        dynamic_range_r: float = 128.0,
        gradient_threshold: float = 22.0,
    ):
        self.k = sauvola_k
        self.R = dynamic_range_r
        self.grad_threshold = gradient_threshold

    def segment_text_and_ink(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Segments manuscript image into:
        - confident_ink_mask: binary boolean mask of confirmed ink strokes
        - background_mask: binary boolean mask of paper/parchment background
        - uncertain_mask: boundary transition zones or faint strokes
        - alpha_weight: smooth float32 [0.0, 1.0] blending weight (1.0 = ink, 0.0 = background)
        - visual_text_mask_pil: PIL Image visualizer for scholars
        """
        if isinstance(image, Image.Image):
            pil_rgb = image.convert("RGB")
            arr_rgb = np.array(pil_rgb, dtype=np.float32)
        else:
            arr_rgb = image.astype(np.float32)
            if arr_rgb.max() <= 1.0:
                arr_rgb = arr_rgb * 255.0
            pil_rgb = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8))

        h, w = arr_rgb.shape[:2]

        # 1. Luminance computation
        gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]

        # 2. Adaptive Multi-Scale Sauvola Thresholding
        win_size = max(15, int(min(w, h) * 0.12) | 1)
        mean = ndimage.uniform_filter(gray, size=win_size)
        mean_sq = ndimage.uniform_filter(gray ** 2, size=win_size)
        var = np.maximum(mean_sq - (mean ** 2), 0.0)
        std = np.sqrt(var)

        sauvola_threshold = mean * (1.0 + self.k * ((std / self.R) - 1.0))
        dark_candidate = gray < sauvola_threshold

        # 3. Directional Sobel Gradient Confirmation
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        grad_mag = np.hypot(gx, gy)
        grad_dilated = ndimage.grey_dilation(grad_mag, size=(3, 3))

        # True ink strokes must be both darker than local paper and possess crisp stroke boundaries
        confident_ink = dark_candidate & (grad_dilated > self.grad_threshold)

        # Morphological cleanup: remove isolated 1-pixel noise specks
        confident_ink = ndimage.binary_opening(confident_ink, structure=np.ones((2, 2)))

        # 4. Background and Uncertain Region Partitioning
        # Background: noticeably brighter than threshold and with low gradient
        confident_bg = (gray > (sauvola_threshold + 15.0)) & (grad_dilated < (self.grad_threshold * 0.75))

        # Uncertain: boundary transition zone between ink and background, or faint ink
        uncertain = (~confident_ink) & (~confident_bg)

        # 5. Smooth Alpha Blending Weight for Sub-pixel Anti-aliasing
        # Dilate ink slightly and smooth with gentle Gaussian filter to avoid sharp halo artifacts
        dilated_ink = ndimage.binary_dilation(confident_ink, iterations=1)
        soft_weight = ndimage.gaussian_filter(dilated_ink.astype(np.float32), sigma=0.8)
        # Boost confidence in core ink centers
        alpha_weight = np.clip(soft_weight * 1.4, 0.0, 1.0)

        # 6. Scholar Visualization Mask
        # Renders text in deep indigo (#1E3A8A) over light cream parchment (#FDFBF7)
        vis_arr = np.ones((h, w, 3), dtype=np.uint8) * 250
        vis_arr[:, :, 0] = 248
        vis_arr[:, :, 1] = 246
        vis_arr[:, :, 2] = 240
        # Color ink pixels
        vis_arr[confident_ink] = [30, 58, 138]  # Deep Indigo ink
        vis_arr[uncertain & ~confident_ink] = [180, 140, 100]  # Amber/brown boundary
        vis_mask_pil = Image.fromarray(vis_arr)

        ink_ratio = float(np.mean(confident_ink))

        return {
            "confident_ink_mask": confident_ink,
            "background_mask": confident_bg,
            "uncertain_mask": uncertain,
            "alpha_weight": alpha_weight,
            "ink_pixel_ratio": round(ink_ratio, 4),
            "ink_pixel_count": int(np.sum(confident_ink)),
            "visual_mask_pil": vis_mask_pil,
            "visual_text_mask_pil": vis_mask_pil,
        }


# Singleton engine instance
text_protection_engine = TextProtectionEngine()

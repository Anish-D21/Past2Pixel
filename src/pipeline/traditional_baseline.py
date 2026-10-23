"""
Conservative Traditional Image-Processing Restoration Baseline for Past2Pixel.
Provides a non-neural, evidence-preserving enhancement baseline:
1. Low-Frequency Background Illumination Estimation & Equalization
2. Restrained Background Denoising (without blurring text boundaries)
3. Historical Ink Contrast Normalization & Stroke Deepening
4. Complete Preservation of Original Stroke Geometry and Ligatures
"""
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image
from scipy import ndimage


class TraditionalRestorationEngine:
    """
    Executes traditional computer vision restoration on manuscripts.
    Acts as a safe, conservative baseline that does not hallucinate or over-smooth.
    """

    def __init__(self, target_bg_luminance: float = 238.0):
        self.target_bg = target_bg_luminance

    def restore_image(
        self,
        image: Union[Image.Image, np.ndarray],
        text_segmentation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes conservative traditional restoration.

        Returns:
            restored_pil: PIL.Image of the enhanced manuscript
            restored_arr: np.ndarray (H, W, 3) in [0, 255]
            metadata: Diagnostic information on background gain and stroke contrast
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
        gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]

        # 1. Obtain text segmentation (or compute if not supplied)
        if text_segmentation is None:
            from .text_protection import text_protection_engine
            text_segmentation = text_protection_engine.segment_text_and_ink(arr_rgb)

        alpha_weight = text_segmentation["alpha_weight"][:, :, np.newaxis]
        confident_ink = text_segmentation["confident_ink_mask"]

        # 2. Background Illumination Surface Estimation
        # Use large dilation kernel to bridge over text strokes and estimate paper surface
        kernel_dim = max(21, int(min(w, h) * 0.15) | 1)
        bg_surface = ndimage.grey_dilation(gray, size=(kernel_dim, kernel_dim))
        bg_smooth = ndimage.gaussian_filter(bg_surface, sigma=kernel_dim * 0.4)

        # Calculate illumination gain to flatten uneven lighting and broad stains
        gain = np.clip(self.target_bg / np.maximum(bg_smooth, 30.0), 0.65, 1.9)
        gain_3d = gain[:, :, np.newaxis]

        # 3. Background Restoration: Brighten and flatten paper surface
        flattened_bg = np.clip(arr_rgb * gain_3d, 0.0, 255.0)

        # Mild edge-preserving background smoothing (only applied in non-ink areas)
        smoothed_bg = ndimage.gaussian_filter(flattened_bg, sigma=[0.6, 0.6, 0.0])
        clean_bg = np.where(alpha_weight > 0.1, flattened_bg, smoothed_bg)

        # 4. Ink Preservation & Gentle Contrast Normalization
        # Keep original stroke colors, deepening the darkest tones slightly to enhance legibility
        enhanced_ink = np.clip(arr_rgb * 0.88, 0.0, 255.0)

        # 5. Seamless Compositing via Text-Aware Alpha Blending
        restored_arr = (1.0 - alpha_weight) * clean_bg + alpha_weight * enhanced_ink
        restored_arr = np.clip(restored_arr, 0.0, 255.0).astype(np.uint8)
        restored_pil = Image.fromarray(restored_arr)

        return {
            "restored_pil": restored_pil,
            "restored_arr": restored_arr,
            "mean_background_gain": round(float(np.mean(gain)), 3),
            "method": "traditional_illumination_normalization",
        }


# Singleton engine instance
traditional_restoration_engine = TraditionalRestorationEngine()

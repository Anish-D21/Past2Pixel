"""
Controlled Synthetic Degradation Engine.
Generates degraded manuscript training pairs (Clean Image, Degraded Image, Ground-Truth Damage Mask)
with reproducible degradation parameters.
"""
import random
from typing import Tuple, Dict, Any, List
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


class SyntheticDegrader:
    """
    Simulates physical and optical degradation commonly found in ancient historical manuscripts:
    - Uneven illumination & non-uniform background fading
    - Surface stains (water spots, tea/ink bleed-through)
    - Physical scratches, cracks, and surface noise
    - Localized tears, holes, and ink occlusions
    - Contrast loss, overall fading, and optical blur
    
    Crucially, tracks affected pixel coordinates to output an exact binary Damage Mask.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def apply_pipeline(
        self,
        image: Image.Image,
        apply_stains: bool = True,
        apply_scratches: bool = True,
        apply_fading: bool = True,
        apply_blur: bool = True,
        apply_noise: bool = True,
        apply_holes: bool = True,
    ) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
        """
        Applies a random combination of degradations to an input clean manuscript PIL Image.

        Returns:
            degraded_img: PIL.Image of the degraded manuscript
            damage_mask: np.ndarray (H, W), dtype float32, in range [0.0, 1.0], where 1.0 indicates damage
            metadata: Dict detailing applied transformations and intensities
        """
        img = image.convert("RGB")
        w, h = img.size
        # Damage mask: 0 = clean, 1 = damaged
        mask = np.zeros((h, w), dtype=np.float32)
        history: List[str] = []

        # 1. Physical Holes / Tears / Localized Missing Regions
        if apply_holes and self.rng.random() > 0.3:
            img, mask = self._apply_missing_regions(img, mask)
            history.append("missing_regions")

        # 2. Surface Stains (watermarks, moisture damage, discoloration)
        if apply_stains and self.rng.random() > 0.2:
            img, mask = self._apply_stains(img, mask)
            history.append("stains")

        # 3. Surface Scratches & Cracks
        if apply_scratches and self.rng.random() > 0.25:
            img, mask = self._apply_scratches(img, mask)
            history.append("scratches")

        # 4. Fading & Contrast Reduction (global and non-uniform lighting)
        if apply_fading and self.rng.random() > 0.2:
            img, mask = self._apply_fading_and_uneven_light(img, mask)
            history.append("fading_uneven_light")

        # 5. Noise (salt & pepper, grain)
        if apply_noise and self.rng.random() > 0.3:
            img, mask = self._apply_noise(img, mask)
            history.append("surface_noise")

        # 6. Optical Blur (defocus / motion)
        if apply_blur and self.rng.random() > 0.35:
            img = self._apply_blur(img)
            history.append("optical_blur")

        metadata = {
            "applied_degradations": history,
            "damage_pixel_ratio": float(np.mean(mask > 0.2)),
            "width": w,
            "height": h,
        }

        return img, mask.astype(np.float32), metadata

    def _apply_missing_regions(
        self, img: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Creates irregular tears or holes simulating physically missing parchment/paper."""
        w, h = img.size
        num_holes = self.rng.randint(1, 3)
        draw_img = ImageDraw.Draw(img)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        draw_mask = ImageDraw.Draw(mask_pil)

        for _ in range(num_holes):
            cx = self.rng.randint(int(w * 0.1), int(w * 0.9))
            cy = self.rng.randint(int(h * 0.1), int(h * 0.9))
            rx = self.rng.randint(10, max(15, int(w * 0.12)))
            ry = self.rng.randint(10, max(15, int(h * 0.12)))

            # Irregular polygon for natural tear look
            num_points = self.rng.randint(6, 10)
            points = []
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                jitter_r = self.rng.uniform(0.6, 1.4)
                px = int(cx + rx * jitter_r * np.cos(angle))
                py = int(cy + ry * jitter_r * np.sin(angle))
                points.append((px, py))

            # Fill with dark/dirty background or parchment edge color
            bg_color = (
                self.rng.randint(25, 60),
                self.rng.randint(20, 50),
                self.rng.randint(15, 40),
            )
            draw_img.polygon(points, fill=bg_color)
            draw_mask.polygon(points, fill=255)

        updated_mask = np.array(mask_pil, dtype=np.float32) / 255.0
        return img, updated_mask

    def _apply_stains(
        self, img: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Creates semi-transparent brown/sepia water stains."""
        w, h = img.size
        num_stains = self.rng.randint(1, 4)
        img_np = np.array(img, dtype=np.float32)

        for _ in range(num_stains):
            cx = self.rng.randint(0, w)
            cy = self.rng.randint(0, h)
            rx = self.rng.randint(max(20, int(w * 0.08)), max(40, int(w * 0.3)))
            ry = self.rng.randint(max(20, int(h * 0.08)), max(40, int(h * 0.3)))

            y_grid, x_grid = np.ogrid[:h, :w]
            dist_sq = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
            stain_intensity = np.clip(1.0 - dist_sq, 0.0, 1.0)
            stain_intensity = stain_intensity ** 1.5

            # Stain coloration (sepia / brownish darkening)
            stain_alpha = self.rng.uniform(0.35, 0.65) * stain_intensity
            stain_r = self.rng.uniform(0.65, 0.85)
            stain_g = self.rng.uniform(0.55, 0.75)
            stain_b = self.rng.uniform(0.35, 0.55)

            img_np[:, :, 0] = img_np[:, :, 0] * (1.0 - stain_alpha * (1.0 - stain_r))
            img_np[:, :, 1] = img_np[:, :, 1] * (1.0 - stain_alpha * (1.0 - stain_g))
            img_np[:, :, 2] = img_np[:, :, 2] * (1.0 - stain_alpha * (1.0 - stain_b))

            mask = np.maximum(mask, stain_intensity * 0.8)

        img_out = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        return img_out, mask

    def _apply_scratches(
        self, img: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Simulates linear scratches and cracks across the manuscript."""
        w, h = img.size
        num_scratches = self.rng.randint(2, 6)
        draw_img = ImageDraw.Draw(img)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        draw_mask = ImageDraw.Draw(mask_pil)

        for _ in range(num_scratches):
            x1 = self.rng.randint(0, w)
            y1 = self.rng.randint(0, h)
            angle = self.rng.uniform(0, 2 * np.pi)
            length = self.rng.randint(int(min(w, h) * 0.2), int(min(w, h) * 0.7))
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            width = self.rng.randint(1, 3)

            scratch_color = self.rng.choice(
                [(220, 215, 200), (40, 35, 30), (180, 170, 150)]
            )
            draw_img.line([(x1, y1), (x2, y2)], fill=scratch_color, width=width)
            draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=width + 1)

        updated_mask = np.array(mask_pil, dtype=np.float32) / 255.0
        return img, updated_mask

    def _apply_fading_and_uneven_light(
        self, img: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Simulates non-uniform lighting and ink fading across one side of the manuscript."""
        w, h = img.size
        img_np = np.array(img, dtype=np.float32)

        # Linear or radial gradient for illumination
        direction = self.rng.choice(["horizontal", "vertical", "radial"])
        if direction == "horizontal":
            grad = np.linspace(
                self.rng.uniform(0.5, 0.8), self.rng.uniform(1.0, 1.2), w
            )
            illumination = np.tile(grad, (h, 1))
        elif direction == "vertical":
            grad = np.linspace(
                self.rng.uniform(0.5, 0.8), self.rng.uniform(1.0, 1.2), h
            )
            illumination = np.tile(grad[:, np.newaxis], (1, w))
        else:
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
            dist_norm = dist / dist.max()
            illumination = 1.0 - (dist_norm * self.rng.uniform(0.2, 0.5))

        for c in range(3):
            img_np[:, :, c] = np.clip(img_np[:, :, c] * illumination, 0, 255)

        # Marked as subtle background degradation in regions where illumination dropped below 0.7
        fade_mask = np.clip((0.8 - illumination) / 0.8, 0.0, 0.5)
        mask = np.maximum(mask, fade_mask)

        return Image.fromarray(img_np.astype(np.uint8)), mask

    def _apply_noise(
        self, img: Image.Image, mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Adds Gaussian and salt-and-pepper grain noise."""
        img_np = np.array(img, dtype=np.float32)
        std = self.rng.uniform(8.0, 22.0)
        noise = self.np_rng.normal(0, std, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255)

        # Mark severe noise locations in mask
        severe_noise = (np.abs(noise).mean(axis=2) > (std * 2.0)).astype(np.float32)
        mask = np.maximum(mask, severe_noise * 0.4)

        return Image.fromarray(noisy.astype(np.uint8)), mask

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Applies gentle optical blur simulating out-of-focus capture."""
        radius = self.rng.uniform(0.8, 1.8)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

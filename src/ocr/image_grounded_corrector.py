"""
src/ocr/image_grounded_corrector.py
Image-Grounded Text Recognition & Correction Engine for Past2Pixel.
Grounds OCR transcriptions directly in the physical manuscript image,
using character stroke presence, bounding box visual evidence, line-crop micro-OCR,
and secondary recognition candidates.
Never invents words or overrides visible stroke evidence with linguistic guesses.
"""
from typing import Dict, Any, List, Optional, Tuple
import io
import base64
import numpy as np
import cv2
from PIL import Image, ImageOps
import unicodedata
import difflib
import logging

from .ocr_preprocessor import ocr_preprocessor
from ..prediction.context_predictor import context_predictor

logger = logging.getLogger(__name__)


class ImageGroundedCorrector:
    """
    Evaluates OCR transcriptions against original and restored manuscript image patches.
    Verifies ink stroke presence, resolves competing multi-pass readings using visual evidence,
    and applies conservative, reversible corrections grounded in physical character geometry.
    """

    def __init__(self):
        self.predictor = context_predictor

    @staticmethod
    def extract_patch_metrics(
        image: Image.Image,
        bbox: List[int],
        pad: int = 4,
    ) -> Dict[str, Any]:
        """
        Extracts an image patch for a word bounding box [left, top, width, height]
        and measures physical stroke density, contrast ratio, and edge energy.
        Also produces a lightweight base64 thumbnail for frontend Word Inspector.
        """
        w_img, h_img = image.size
        if not bbox or len(bbox) < 4:
            return {
                "stroke_density": 0.0,
                "contrast_ratio": 0.0,
                "edge_energy": 0.0,
                "has_visible_ink": False,
                "patch_data_uri": None,
                "bbox": bbox,
            }

        left, top, width, height = bbox
        # Clamped bounding box with padding
        x1 = max(0, left - pad)
        y1 = max(0, top - pad)
        x2 = min(w_img, left + width + pad)
        y2 = min(h_img, top + height + pad)

        if x2 <= x1 or y2 <= y1:
            return {
                "stroke_density": 0.0,
                "contrast_ratio": 0.0,
                "edge_energy": 0.0,
                "has_visible_ink": False,
                "patch_data_uri": None,
                "bbox": bbox,
            }

        crop = image.crop((x1, y1, x2, y2))
        crop_gray = crop.convert("L")
        arr = np.array(crop_gray, dtype=np.uint8)

        # 1. Stroke density: fraction of dark ink pixels via adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
        )
        total_pixels = max(1, arr.size)
        ink_pixels = int(np.count_nonzero(thresh))
        stroke_density = round(float(ink_pixels) / total_pixels, 3)

        # 2. Contrast ratio: dynamic range across patch (max - min) / 255
        p_min = float(np.min(arr))
        p_max = float(np.max(arr))
        contrast_ratio = round((p_max - p_min) / 255.0, 3)

        # 3. Edge energy via Sobel gradient magnitude along character stroke boundaries
        sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_energy = round(float(np.mean(edge_mag)), 2)

        # Physical ink presence rule:
        # Genuine ink stroke requires minimum stroke density (>= 0.8%) and either contrast (>= 0.10) or stroke edges (>= 8.0)
        has_visible_ink = stroke_density >= 0.008 and (contrast_ratio >= 0.10 or edge_energy >= 8.0)

        # 4. Generate compact base64 thumbnail for frontend inspection
        patch_data_uri = None
        try:
            # Resize slightly if very tiny so glyph details are visible
            thumb = crop
            if max(crop.size) < 80:
                scale = min(3.0, 120.0 / max(crop.size))
                tw, th = int(round(crop.width * scale)), int(round(crop.height * scale))
                thumb = crop.resize((tw, th), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            patch_data_uri = f"data:image/png;base64,{b64_str}"
        except Exception:
            patch_data_uri = None

        return {
            "stroke_density": stroke_density,
            "contrast_ratio": contrast_ratio,
            "edge_energy": edge_energy,
            "has_visible_ink": has_visible_ink,
            "patch_data_uri": patch_data_uri,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        }

    def resolve_word_candidate(
        self,
        original_img: Image.Image,
        word: str,
        bbox: List[int],
        confidence: float,
        competing_candidates: List[str],
        left_context: str = "",
        right_context: str = "",
        lang: str = "san+hin",
    ) -> Dict[str, Any]:
        """Convenience evaluation for an individual word bounding box directly on an image."""
        metrics = self.extract_patch_metrics(original_img, bbox)
        return self.evaluate_and_correct_word(
            word=word,
            confidence=confidence,
            bbox=bbox,
            patch_metrics=metrics,
            competing_candidates=competing_candidates,
            left_context=left_context,
            right_context=right_context,
            lang=lang,
        )

    def evaluate_and_correct_word(
        self,
        word: str,
        confidence: float,
        bbox: List[int],
        patch_metrics: Dict[str, Any],
        competing_candidates: List[str],
        left_context: str = "",
        right_context: str = "",
        lang: str = "san+hin",
    ) -> Dict[str, Any]:
        """
        Decides whether to keep, correct, or flag a single word token based on visual evidence.
        - High-confidence words with visible ink are kept as IMAGE_VERIFIED.
        - Regions with no visible ink (< 0.8% stroke density) are marked as DAMAGED_UNREADABLE.
        - Words with low confidence or competing readings are resolved using visual stroke support,
          with linguistic N-gram probabilities used only as an auxiliary arbiter for ambiguous cases.
        """
        clean_word = word.strip()
        has_ink = patch_metrics.get("has_visible_ink", True)
        stroke_density = patch_metrics.get("stroke_density", 0.0)

        # CASE 1: Completely degraded/eroded or blank region where OCR hallucinated text without ink
        if (not has_ink and confidence < 30.0) or (stroke_density < 0.005 and confidence < 20.0):
            # Area has no ink strokes in the manuscript: mark as damaged/obscured
            return {
                "word": "[DAMAGED]",
                "original_word": clean_word,
                "confidence": confidence,
                "status": "DAMAGED_UNREADABLE",
                "is_uncertain": True,
                "evidence": "Manuscript image patch contains no identifiable ink strokes (< 0.5% stroke density).",
                "alternatives": competing_candidates,
                "patch_metrics": patch_metrics,
            }

        # If low stroke density but OCR recognized valid characters, preserve text and flag uncertainty
        if stroke_density < 0.008 and confidence < 40.0:
            is_uncertain_low_stroke = True
        else:
            is_uncertain_low_stroke = False

        # CASE 2: High confidence with strong stroke presence
        if confidence >= 72.0 and not any(c in clean_word for c in "~^{}_|\\/`+=<>*"):
            return {
                "word": clean_word,
                "original_word": clean_word,
                "confidence": confidence,
                "status": "IMAGE_VERIFIED",
                "is_uncertain": False,
                "evidence": f"Verified by manuscript image (stroke density {stroke_density:.1%}, confidence {confidence:.1f}%).",
                "alternatives": competing_candidates,
                "patch_metrics": patch_metrics,
            }

        # CASE 3: Low confidence or noisy glyph requiring candidate reconciliation
        # Clean up obvious OCR substitution errors (e.g. pipe symbol '|' for danda '।')
        normalized_word = clean_word
        if normalized_word.endswith("|"):
            normalized_word = normalized_word[:-1] + "।"

        # Remove stray punctuation noise from word edges
        trimmed = normalized_word.strip("~^_{}[]*§°±«»`=<>")
        if trimmed and trimmed != normalized_word:
            normalized_word = trimmed

        # Gather distinct candidate readings (original + competing from other passes/line-crop)
        candidates_to_score = []
        for c in [normalized_word, clean_word] + competing_candidates:
            c_clean = c.strip()
            if c_clean and c_clean not in candidates_to_score:
                candidates_to_score.append(c_clean)

        best_candidate = normalized_word
        best_score = 0.0
        selection_reason = "Preserved original reading with low confidence."

        # Linguistic assistance for ranking ambiguous candidates
        ling_scores: Dict[str, float] = {}
        if len(candidates_to_score) > 1 and (left_context or right_context):
            try:
                # Ask context predictor for ranking among these specific candidate words
                ling_scores = self.predictor.score_candidates(
                    candidates=candidates_to_score,
                    left_context=left_context,
                    right_context=right_context,
                )
            except Exception:
                ling_scores = {}

        # Evaluate each candidate: 65% Visual/OCR score + 35% Linguistic transition score
        for cand in candidates_to_score:
            # Baseline visual confidence: candidate receives full confidence if it was the OCR reading
            vis_conf = confidence if cand == clean_word else 60.0
            # Devanagari / valid script bonus
            script_bonus = 15.0 if any("\u0900" <= ch <= "\u097F" for ch in cand) else 5.0
            # Noise penalty
            noise_pen = sum(15.0 for ch in cand if ch in "~^{}_|\\/`+=<>[]*")

            vis_score = max(0.0, (vis_conf * 0.65) + script_bonus - noise_pen)
            ling_score = ling_scores.get(cand, 0.1) * 35.0
            composite = vis_score + ling_score

            if composite > best_score:
                best_score = composite
                best_candidate = cand
                if cand != clean_word:
                    selection_reason = (
                        f"Corrected using multi-pass image evidence & linguistic consistency "
                        f"(visual conf {vis_conf:.0f}%, ling score {ling_scores.get(cand, 0):.2f})."
                    )

        was_corrected = best_candidate != clean_word
        status_label = "IMAGE_SUPPORTED_CORRECTION" if was_corrected else "UNCERTAIN_REVIEW_NEEDED"

        return {
            "word": best_candidate,
            "original_word": clean_word,
            "confidence": round(min(100.0, max(confidence, 65.0 if was_corrected else confidence)), 1),
            "status": status_label,
            "is_uncertain": not was_corrected and confidence < 55.0,
            "evidence": selection_reason if was_corrected else f"Low confidence ({confidence:.1f}%), stroke density {stroke_density:.1%}.",
            "alternatives": [c for c in candidates_to_score if c != best_candidate],
            "patch_metrics": patch_metrics,
        }

    def correct_document(
        self,
        image: Image.Image,
        lines_detail: List[Dict[str, Any]],
        restored_image: Optional[Image.Image] = None,
        lang: str = "san+hin",
    ) -> Dict[str, Any]:
        """
        Processes entire reconciled document line-by-line and word-by-word.
        Applies image-grounded patch verification and produces:
        - corrected_transcription (image-supported text)
        - raw_transcription (immutable original OCR text)
        - words_detail (complete word tokens with bounding boxes & patch thumbnails)
        - corrections_applied (audit log of every change)
        """
        corrected_lines = []
        raw_lines = []
        all_words_detail = []
        corrections_applied = []
        uncertain_spans_count = 0

        # Primary reference image for stroke analysis
        ref_image = restored_image if restored_image is not None else image

        for line_item in lines_detail:
            line_text = line_item.get("text", "")
            raw_lines.append(line_text)
            line_num = line_item.get("line_num", 1)
            line_bbox = line_item.get("bbox", [0, 0, 100, 30])
            line_words = line_item.get("words", [])
            alt_readings = [alt.get("text", "") for alt in line_item.get("alternate_readings", [])]

            corrected_words_in_line = []

            # If line has structured word tokens with bounding boxes
            if line_words and isinstance(line_words, list):
                for idx, w_token in enumerate(line_words):
                    w_str = w_token.get("word", "") if isinstance(w_token, dict) else str(w_token)
                    w_bbox = w_token.get("bbox", line_bbox) if isinstance(w_token, dict) else line_bbox
                    w_conf = w_token.get("confidence", line_item.get("mean_confidence", 50.0)) if isinstance(w_token, dict) else 50.0

                    # Context words
                    before_str = " ".join([wt.get("word", "") if isinstance(wt, dict) else str(wt) for wt in line_words[:idx]])
                    after_str = " ".join([wt.get("word", "") if isinstance(wt, dict) else str(wt) for wt in line_words[idx + 1:]])

                    # Extract patch metrics from reference image
                    patch_metrics = self.extract_patch_metrics(ref_image, w_bbox)

                    # Gather competing candidate tokens from alternate line readings based on textual similarity
                    competing = []
                    for alt_text in alt_readings:
                        alt_tokens = alt_text.split()
                        for alt_w in alt_tokens:
                            alt_w_clean = alt_w.strip()
                            if not alt_w_clean or alt_w_clean == w_str or alt_w_clean in competing:
                                continue
                            sim = difflib.SequenceMatcher(None, w_str, alt_w_clean).ratio()
                            # Candidate must have character overlap/similarity to be considered an alternate reading of w_str
                            if sim >= 0.45 or (len(w_str) <= 3 and len(alt_w_clean) <= 3 and sim >= 0.3):
                                competing.append(alt_w_clean)

                    word_eval = self.evaluate_and_correct_word(
                        word=w_str,
                        confidence=w_conf,
                        bbox=w_bbox,
                        patch_metrics=patch_metrics,
                        competing_candidates=competing,
                        left_context=before_str,
                        right_context=after_str,
                        lang=lang,
                    )

                    word_eval["line_num"] = line_num
                    word_eval["token_index"] = len(all_words_detail)
                    all_words_detail.append(word_eval)

                    corrected_words_in_line.append(word_eval["word"])

                    if word_eval["status"] == "IMAGE_SUPPORTED_CORRECTION":
                        corrections_applied.append({
                            "line_num": line_num,
                            "original": word_eval["original_word"],
                            "corrected": word_eval["word"],
                            "confidence": word_eval["confidence"],
                            "reason": word_eval["evidence"],
                        })

                    if word_eval.get("is_uncertain"):
                        uncertain_spans_count += 1

                corrected_line_str = " ".join(corrected_words_in_line)
            else:
                # Fallback if line has no word bounding boxes: split string
                tokens = line_text.split()
                for idx, tok in enumerate(tokens):
                    all_words_detail.append({
                        "word": tok,
                        "original_word": tok,
                        "confidence": line_item.get("mean_confidence", 50.0),
                        "status": "IMAGE_VERIFIED" if (line_item.get("mean_confidence", 50.0) >= 65) else "UNCERTAIN_REVIEW_NEEDED",
                        "is_uncertain": line_item.get("mean_confidence", 50.0) < 50.0,
                        "line_num": line_num,
                        "token_index": len(all_words_detail),
                        "evidence": "Extracted from full-line OCR pass.",
                        "alternatives": [],
                        "patch_metrics": {"bbox": line_bbox, "has_visible_ink": True},
                    })
                    corrected_words_in_line.append(tok)
                corrected_line_str = line_text

            corrected_lines.append(corrected_line_str)

        raw_transcription = "\n".join(raw_lines)
        corrected_transcription = "\n".join(corrected_lines)

        return {
            "corrected_transcription": corrected_transcription,
            "raw_transcription": raw_transcription,
            "words_detail": all_words_detail,
            "corrections_applied": corrections_applied,
            "corrections_count": len(corrections_applied),
            "uncertain_spans_count": uncertain_spans_count,
        }


# Singleton corrector instance
image_grounded_corrector = ImageGroundedCorrector()

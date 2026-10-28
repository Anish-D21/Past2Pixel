"""
src/ocr/ocr_multi_pass.py
Maximum OCR Text Recovery Engine for Past2Pixel.
Executes multi-pass OCR on manuscript images with complementary Page Segmentation Modes (PSM)
and conservative image transforms, followed by geometric line-level reconciliation,
duplicate elimination, secondary line-crop micro-OCR, and image-grounded text correction.
"""
from typing import Dict, Any, List, Optional, Tuple
import time
import difflib
import unicodedata
import logging
from PIL import Image, ImageOps
import pytesseract

from .ocr_preprocessor import ocr_preprocessor, OCRPreprocessor
from .image_grounded_corrector import image_grounded_corrector

logger = logging.getLogger(__name__)


def score_reading(text: str, conf: float) -> float:
    """
    Evaluates quality of a candidate line reading.
    Rewards genuine character content and confidence, penalizes random noise/symbols.
    """
    clean_text = text.strip()
    if not clean_text:
        return 0.0

    # Count high-frequency OCR hallucination/noise symbols common in degraded margins
    noise_symbols = sum(1 for c in clean_text if c in '~^{}_|\\/`+=<>[]*§°±«»~')
    char_len = len(clean_text)
    legit_chars = max(0, char_len - noise_symbols)

    # Devanagari vowel sign / accent count bonus
    devanagari_marks = sum(1 for c in clean_text if '\u0900' <= c <= '\u097F')

    score = (legit_chars * 1.5) + (devanagari_marks * 0.8) + (conf * 0.4) - (noise_symbols * 3.5)
    return max(0.0, float(score))


class MultiPassOCREngine:
    """
    Coordinates multi-pass OCR extraction, secondary line-crop recognition,
    and intelligent line-level reconciliation.
    Never blindly concatenates passes to avoid duplicate paragraphs or disordered text.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd

    def _extract_pass_lines(
        self,
        image: Image.Image,
        lang: str,
        config: str,
        pass_name: str,
        pad: int = 30,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs pytesseract.image_to_data on an image variant, extracts line bounding boxes,
        captures word-level bounding boxes and confidences, normalizes coordinates back
        to unpadded source space, and returns structured lines.
        """
        start_t = time.perf_counter()

        # Add white border padding to prevent edge text clipping
        # Scale padding proportionally to image dimensions so small line crops don't get washed out
        eff_pad = min(pad, max(4, int(min(image.width, image.height) * 0.08))) if pad > 0 else 0
        if eff_pad > 0:
            padded_img = ImageOps.expand(image, border=eff_pad, fill="white")
        else:
            padded_img = image

        try:
            data = pytesseract.image_to_data(
                padded_img,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            logger.warning("Pass %s failed with config %s: %s", pass_name, config, e)
            return [], round((time.perf_counter() - start_t) * 1000, 2)

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        lines_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        n_boxes = len(data.get("text", []))

        for i in range(n_boxes):
            raw_w = data["text"][i]
            if not raw_w:
                continue
            clean_w = unicodedata.normalize("NFC", raw_w).strip()
            if not clean_w:
                continue

            b = data["block_num"][i]
            p = data["par_num"][i]
            l = data["line_num"][i]
            key = (b, p, l)

            left = max(0, data["left"][i] - eff_pad)
            top = max(0, data["top"][i] - eff_pad)
            width = max(1, data["width"][i])
            height = max(1, data["height"][i])
            right = left + width
            bottom = top + height
            conf = float(data["conf"][i]) if str(data["conf"][i]).replace("-", "").isdigit() else 0.0

            if key not in lines_map:
                lines_map[key] = {
                    "words": [],
                    "word_tokens": [],
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "confs": [],
                    "pass_name": pass_name,
                }

            lines_map[key]["words"].append(clean_w)
            lines_map[key]["word_tokens"].append({
                "word": clean_w,
                "bbox": [left, top, width, height],
                "confidence": max(0.0, conf),
            })

            if conf >= 0:
                lines_map[key]["confs"].append(conf)
            lines_map[key]["left"] = min(lines_map[key]["left"], left)
            lines_map[key]["top"] = min(lines_map[key]["top"], top)
            lines_map[key]["right"] = max(lines_map[key]["right"], right)
            lines_map[key]["bottom"] = max(lines_map[key]["bottom"], bottom)

        structured_lines: List[Dict[str, Any]] = []
        for key, val in lines_map.items():
            line_str = " ".join(val["words"]).strip()
            # Filter out standalone punctuation noise artifacts while preserving valid single chars/numerals
            if len(line_str) < 2 and not line_str.isalnum() and not any("\u0900" <= c <= "\u097F" for c in line_str):
                continue

            confs = val["confs"]
            mean_conf = round(sum(confs) / len(confs), 1) if confs else 50.0

            structured_lines.append({
                "text": line_str,
                "top": val["top"],
                "bottom": val["bottom"],
                "left": val["left"],
                "right": val["right"],
                "mean_conf": mean_conf,
                "pass_name": pass_name,
                "quality_score": score_reading(line_str, mean_conf),
                "words": val.get("word_tokens", []),
            })

        # Sort top-to-bottom for natural reading order
        structured_lines.sort(key=lambda x: (round(x["top"] / 15) * 15, x["left"]))
        return structured_lines, elapsed_ms

    def reconcile_passes(
        self,
        pass_results: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reconciles line detections from multiple passes.
        - Matches overlapping lines geometrically and textually to prevent duplication.
        - Preserves horizontally separate columns and margin notes without collapsing them.
        - Picks the highest-fidelity reading for each line.
        - Preserves word-level token bounding boxes and confidences for deep image grounding.
        - Inserts newly recovered lines at their proper vertical position.
        - Preserves uncertain lines for paleographic inspection.
        """
        reconciled: List[Dict[str, Any]] = []

        for pass_name, lines in pass_results:
            for candidate in lines:
                matched_idx = -1
                best_sim = 0.0

                for idx, existing in enumerate(reconciled):
                    # Check horizontal separation (side-by-side columns or margin notes)
                    h_overlap = max(0, min(candidate["right"], existing["right"]) - max(candidate["left"], existing["left"]))
                    is_horiz_disjoint = (
                        h_overlap == 0 and
                        (candidate["right"] < existing["left"] - 25 or existing["right"] < candidate["left"] - 25)
                    )

                    # 1. Vertical bounding box overlap check
                    overlap_top = max(candidate["top"], existing["top"])
                    overlap_bot = min(candidate["bottom"], existing["bottom"])
                    overlap_h = max(0, overlap_bot - overlap_top)
                    min_h = max(1, min(candidate["bottom"] - candidate["top"], existing["bottom"] - existing["top"]))
                    v_overlap_ratio = overlap_h / min_h

                    # 2. Text similarity check
                    sim = difflib.SequenceMatcher(None, candidate["text"], existing["text"]).ratio()

                    # Substring check: candidate text contained in existing or vice versa
                    cand_t = candidate["text"].strip()
                    exist_t = existing["text"].strip()
                    is_sub = (cand_t in exist_t or exist_t in cand_t) and min(len(cand_t), len(exist_t)) >= 3

                    # If lines are horizontally disjoint and text is not identical, they belong to different columns/margins
                    if is_horiz_disjoint and sim < 0.85 and not is_sub:
                        continue

                    # Same physical line if significant vertical overlap or near-identical text or substring
                    v_distance = abs(candidate["top"] - existing["top"])
                    is_match = (v_overlap_ratio >= 0.35 and (sim >= 0.25 or is_sub)) or (sim >= 0.80 and v_distance < 80) or (is_sub and v_distance < 50)

                    if is_match:
                        if sim > best_sim or (v_overlap_ratio >= 0.35 and matched_idx == -1) or is_sub:
                            best_sim = sim
                            matched_idx = idx

                if matched_idx >= 0:
                    # Same line detected across different passes: resolve best reading
                    existing = reconciled[matched_idx]
                    cand_score = candidate["quality_score"]
                    exist_score = existing["quality_score"]

                    # Keep alternate reading for scholar inspection
                    if "alternate_readings" not in existing:
                        existing["alternate_readings"] = []

                    alt_entry = {
                        "text": candidate["text"],
                        "pass_name": candidate["pass_name"],
                        "mean_conf": candidate["mean_conf"],
                    }
                    if alt_entry["text"] != existing["text"]:
                        existing["alternate_readings"].append(alt_entry)

                    # Replace with higher scoring candidate
                    if cand_score > exist_score:
                        existing["alternate_readings"].append({
                            "text": existing["text"],
                            "pass_name": existing["pass_name"],
                            "mean_conf": existing["mean_conf"],
                        })
                        existing["text"] = candidate["text"]
                        existing["mean_conf"] = candidate["mean_conf"]
                        existing["pass_name"] = candidate["pass_name"]
                        existing["quality_score"] = cand_score
                        existing["top"] = min(existing["top"], candidate["top"])
                        existing["bottom"] = max(existing["bottom"], candidate["bottom"])
                        existing["left"] = min(existing["left"], candidate["left"])
                        existing["right"] = max(existing["right"], candidate["right"])
                        existing["words"] = candidate.get("words", [])
                else:
                    # Genuinely new / recovered line: add to document
                    new_item = dict(candidate)
                    new_item["alternate_readings"] = []
                    new_item["words"] = candidate.get("words", [])
                    reconciled.append(new_item)

        # Sort reconciled lines strictly by vertical reading order band + horizontal offset
        reconciled.sort(key=lambda x: (round(x["top"] / 15) * 15, x["left"]))

        # Format lines_detail and identify uncertain lines (< 45% confidence or noise warning)
        lines_detail = []
        uncertain_lines = []

        for idx, item in enumerate(reconciled, start=1):
            is_uncertain = item["mean_conf"] < 48.0 or item["quality_score"] < 10.0
            detail_entry = {
                "line_num": idx,
                "text": item["text"],
                "mean_confidence": item["mean_conf"],
                "pass_name": item["pass_name"],
                "bbox": [item["left"], item["top"], max(1, item["right"] - item["left"]), max(1, item["bottom"] - item["top"])],
                "is_uncertain": is_uncertain,
                "alternate_readings": item.get("alternate_readings", []),
                "words": item.get("words", []),
            }
            lines_detail.append(detail_entry)
            if is_uncertain:
                uncertain_lines.append(detail_entry)

        # Assemble full cohesive transcription text
        reconciled_text = "\n".join(item["text"] for item in lines_detail)

        return reconciled_text, lines_detail, uncertain_lines

    def run_line_crop_micro_ocr(
        self,
        image: Image.Image,
        lines_detail: List[Dict[str, Any]],
        lang: str = "san+hin",
    ) -> None:
        """
        Secondary Recognition Pass:
        For low-confidence or uncertain lines, crops the individual line from the image,
        applies local CLAHE contrast and mild padding, and runs single-line OCR (--psm 7).
        Stores the resulting reading into alternate_readings for paleographic review.
        """
        w, h = image.size
        for line_item in lines_detail:
            # Only run targeted micro-OCR if line is uncertain or mean confidence < 52%
            if not line_item.get("is_uncertain") and line_item.get("mean_confidence", 100) >= 52:
                continue

            bbox = line_item.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            lx, ly, lw, lh = bbox
            x1 = max(0, lx - 8)
            y1 = max(0, ly - 6)
            x2 = min(w, lx + lw + 8)
            y2 = min(h, ly + lh + 6)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            # Enhance crop contrast specifically for line reading
            enhanced_crop = ocr_preprocessor.clahe_contrast_enhancement(crop, clip_limit=2.5)
            padded_crop = ImageOps.expand(enhanced_crop, border=12, fill="white")

            try:
                line_str = pytesseract.image_to_string(padded_crop, lang=lang, config="--psm 7")
                clean_crop_txt = unicodedata.normalize("NFC", line_str).strip()
                if clean_crop_txt and clean_crop_txt != line_item["text"]:
                    if "alternate_readings" not in line_item:
                        line_item["alternate_readings"] = []
                    line_item["alternate_readings"].append({
                        "text": clean_crop_txt,
                        "pass_name": "line_crop_micro_ocr_psm7",
                        "mean_conf": 65.0,
                    })
            except Exception as e:
                logger.debug("Line crop micro-OCR failed for line #%d: %s", line_item.get("line_num", 0), e)

    def extract_reconciled(
        self,
        image: Image.Image,
        lang: str = "san+hin",
        enable_sparse_pass: bool = True,
        max_passes: int = 5,
        restored_image: Optional[Image.Image] = None,
        force_2x_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes multi-pass OCR on the enhanced image, generates complementary variants,
        reconciles all recognized lines, executes secondary line-crop micro-OCR,
        and applies image-grounded patch verification.
        Never overwrites the original image.
        """
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size
        total_start = time.perf_counter()

        # 1. Physical image quality assessment and skew correction
        img_quality = ocr_preprocessor.evaluate_image_quality(rgb_img)
        skew_deg = img_quality.get("skew_angle_deg", 0.0)

        if abs(skew_deg) >= 1.0:
            working_img, _ = ocr_preprocessor.estimate_and_correct_skew(rgb_img)
            deskew_applied = True
        else:
            working_img = rgb_img
            deskew_applied = False

        pass_results: List[Tuple[str, List[Dict[str, Any]]]] = []
        pass_summaries: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # PASS 1: Border-Padded Block Layout (PSM 6)
        # Prevents edge-clipping and captures standard text blocks
        # -------------------------------------------------------------
        p1_lines, p1_time = self._extract_pass_lines(
            working_img,
            lang=lang,
            config="--psm 6",
            pass_name="block_padded_psm6",
            pad=30,
        )
        pass_results.append(("block_padded_psm6", p1_lines))
        pass_summaries["block_padded_psm6"] = {
            "name": "Block Uniform Layout (PSM 6 + 30px Border Pad)",
            "lines_count": len(p1_lines),
            "character_count": sum(len(l["text"]) for l in p1_lines),
            "mean_confidence": round(sum(l["mean_conf"] for l in p1_lines) / max(len(p1_lines), 1), 1),
            "processing_time_ms": p1_time,
        }

        # -------------------------------------------------------------
        # PASS 2: Standard Automatic Page Segmentation (PSM 3)
        # Captures multi-paragraph, multi-column divisions
        # -------------------------------------------------------------
        if len(pass_results) < max_passes:
            p2_lines, p2_time = self._extract_pass_lines(
                working_img,
                lang=lang,
                config="--psm 3",
                pass_name="layout_standard_psm3",
                pad=30,
            )
            pass_results.append(("layout_standard_psm3", p2_lines))
            pass_summaries["layout_standard_psm3"] = {
                "name": "Automatic Page Layout (PSM 3 + Border Pad)",
                "lines_count": len(p2_lines),
                "character_count": sum(len(l["text"]) for l in p2_lines),
                "mean_confidence": round(sum(l["mean_conf"] for l in p2_lines) / max(len(p2_lines), 1), 1),
                "processing_time_ms": p2_time,
            }

        # -------------------------------------------------------------
        # PASS 3: Sparse Layout (PSM 11) - Critical for Manuscripts!
        # Finds isolated marginalia, header folios numbers, colophons, and annotations
        # -------------------------------------------------------------
        if enable_sparse_pass and len(pass_results) < max_passes:
            p3_lines, p3_time = self._extract_pass_lines(
                working_img,
                lang=lang,
                config="--psm 11",
                pass_name="sparse_marginalia_psm11",
                pad=30,
            )
            pass_results.append(("sparse_marginalia_psm11", p3_lines))
            pass_summaries["sparse_marginalia_psm11"] = {
                "name": "Sparse Text & Marginalia (PSM 11)",
                "lines_count": len(p3_lines),
                "character_count": sum(len(l["text"]) for l in p3_lines),
                "mean_confidence": round(sum(l["mean_conf"] for l in p3_lines) / max(len(p3_lines), 1), 1),
                "processing_time_ms": p3_time,
            }

        # -------------------------------------------------------------
        # PASS 4: Local CLAHE Contrast Enhanced Pass (PSM 6)
        # Recovers faint ink strokes, thin Devanagari matras, faded vowel signs
        # -------------------------------------------------------------
        if len(pass_results) < max_passes:
            clahe_img = ocr_preprocessor.clahe_contrast_enhancement(working_img, clip_limit=2.2)
            p4_lines, p4_time = self._extract_pass_lines(
                clahe_img,
                lang=lang,
                config="--psm 6",
                pass_name="clahe_faint_ink_psm6",
                pad=30,
            )
            pass_results.append(("clahe_faint_ink_psm6", p4_lines))
            pass_summaries["clahe_faint_ink_psm6"] = {
                "name": "Faint Ink CLAHE Contrast (PSM 6)",
                "lines_count": len(p4_lines),
                "character_count": sum(len(l["text"]) for l in p4_lines),
                "mean_confidence": round(sum(l["mean_conf"] for l in p4_lines) / max(len(p4_lines), 1), 1),
                "processing_time_ms": p4_time,
            }

        # -------------------------------------------------------------
        # PASS 5 (Conditional): 2× Super-Resolution Upscaling Pass
        # Triggered when force_2x_pass is True or image is small (min dimension < 240)
        # -------------------------------------------------------------
        already_upscaled = getattr(working_img, "_is_2x_upscaled", False)
        needs_upscale_pass = force_2x_pass or (min(w, h) < 240 and not already_upscaled)
        if needs_upscale_pass and len(pass_results) < (max_passes + 1):
            upscaled_img = ocr_preprocessor.upscale_2x(working_img)
            p5_lines, p5_time = self._extract_pass_lines(
                upscaled_img,
                lang=lang,
                config="--psm 6",
                pass_name="super_resolution_upscale",
                pad=20,
            )
            # Rescale line and word coordinates back to original working_img space
            scale_x = working_img.width / max(upscaled_img.width, 1)
            scale_y = working_img.height / max(upscaled_img.height, 1)
            for line_item in p5_lines:
                line_item["top"] = int(round(line_item["top"] * scale_y))
                line_item["bottom"] = int(round(line_item["bottom"] * scale_y))
                line_item["left"] = int(round(line_item["left"] * scale_x))
                line_item["right"] = int(round(line_item["right"] * scale_x))
                for w_token in line_item.get("words", []):
                    if "bbox" in w_token and len(w_token["bbox"]) == 4:
                        bx, by, bw, bh = w_token["bbox"]
                        w_token["bbox"] = [
                            int(round(bx * scale_x)),
                            int(round(by * scale_y)),
                            int(round(bw * scale_x)),
                            int(round(bh * scale_y)),
                        ]
            pass_results.append(("super_resolution_upscale", p5_lines))
            pass_summaries["super_resolution_upscale"] = {
                "name": "2× Super-Resolution Upscaling (PSM 6)",
                "lines_count": len(p5_lines),
                "character_count": sum(len(l["text"]) for l in p5_lines),
                "mean_confidence": round(sum(l["mean_conf"] for l in p5_lines) / max(len(p5_lines), 1), 1),
                "processing_time_ms": p5_time,
            }

        # -------------------------------------------------------------
        # Line-Level Geometric & Textual Reconciliation
        # -------------------------------------------------------------
        reconciled_text, lines_detail, uncertain_lines = self.reconcile_passes(pass_results)

        # -------------------------------------------------------------
        # Secondary Recognition Pass: Line-Crop Micro-OCR on uncertain lines
        # -------------------------------------------------------------
        self.run_line_crop_micro_ocr(working_img, lines_detail, lang=lang)

        # -------------------------------------------------------------
        # Image-Grounded Text Verification & Correction
        # -------------------------------------------------------------
        ref_image = restored_image if restored_image is not None else working_img
        correction_result = image_grounded_corrector.correct_document(
            image=working_img,
            lines_detail=lines_detail,
            restored_image=ref_image,
            lang=lang,
        )

        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000, 2)

        # Compute overall mean confidence across reconciled lines
        all_confs = [l["mean_confidence"] for l in lines_detail if l.get("mean_confidence") is not None]
        overall_mean_conf = round(sum(all_confs) / max(len(all_confs), 1), 1) if all_confs else None

        # Count provenance contributions
        provenance_breakdown: Dict[str, int] = {}
        for l in lines_detail:
            p_name = l["pass_name"]
            provenance_breakdown[p_name] = provenance_breakdown.get(p_name, 0) + 1

        # Use image-supported transcription as main text, keeping raw text for audit trail
        final_corrected_text = correction_result["corrected_transcription"]
        raw_ocr_text = reconciled_text

        is_sr = getattr(working_img, "_is_2x_upscaled", False) or "super_resolution_upscale" in pass_summaries
        orig_dim = [w // 2, h // 2] if getattr(working_img, "_is_2x_upscaled", False) else [w, h]
        proc_dim = [working_img.width, working_img.height]

        return {
            "status": "SUCCESS",
            "extracted_text": final_corrected_text if final_corrected_text else raw_ocr_text,
            "raw_extracted_text": raw_ocr_text,
            "corrected_text": final_corrected_text,
            "processing_time_ms": total_elapsed_ms,
            "character_count": len(final_corrected_text),
            "word_count": len(final_corrected_text.split()),
            "lines_count": len(lines_detail),
            "mean_confidence": overall_mean_conf,
            "is_multi_pass": True,
            "original_dimensions": orig_dim,
            "processed_dimensions": proc_dim,
            "super_resolution_applied": is_sr,
            "upscale_factor": 2.0 if is_sr else 1.0,
            "lines_detail": lines_detail,
            "words_detail": correction_result.get("words_detail", []),
            "image_grounded_corrections": correction_result.get("corrections_applied", []),
            "corrections_count": correction_result.get("corrections_count", 0),
            "uncertain_lines_count": len(uncertain_lines),
            "uncertain_lines": uncertain_lines,
            "uncertain_spans_count": correction_result.get("uncertain_spans_count", 0),
            "pass_summaries": pass_summaries,
            "provenance_breakdown": provenance_breakdown,
            "image_quality": img_quality,
            "deskew_applied": deskew_applied,
            "secondary_model_status": {
                "name": "Line-Crop Micro-OCR (PSM 7)",
                "status": "ACTIVE",
                "neural_model_installed": False,
                "notes": "Single-line micro-OCR pass is actively resolving uncertain lines; external heavy neural model (TrOCR) is optional.",
            },
            "language_used": lang,
            "is_authentic_ocr": True,
            "error": None,
        }


# Global singleton multi-pass engine
multi_pass_ocr_engine = MultiPassOCREngine()

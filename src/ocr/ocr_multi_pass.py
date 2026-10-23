"""
src/ocr/ocr_multi_pass.py
Maximum OCR Text Recovery Engine for Past2Pixel.
Executes multi-pass OCR on manuscript images with complementary Page Segmentation Modes (PSM)
and conservative image transforms, followed by geometric line-level reconciliation,
duplicate elimination, and reading-order reconstruction.
"""
from typing import Dict, Any, List, Optional, Tuple
import time
import difflib
import unicodedata
import logging
from PIL import Image, ImageOps
import pytesseract

from .ocr_preprocessor import OCRPreprocessor

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
    Coordinates multi-pass OCR extraction and intelligent line-level reconciliation.
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
        normalizes coordinates back to unpadded source space, and returns structured lines.
        """
        start_t = time.perf_counter()

        # Add white border padding to prevent edge text clipping
        if pad > 0:
            padded_img = ImageOps.expand(image, border=pad, fill="white")
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

            left = data["left"][i] - pad
            top = data["top"][i] - pad
            width = data["width"][i]
            height = data["height"][i]
            right = left + width
            bottom = top + height
            conf = float(data["conf"][i]) if str(data["conf"][i]).replace("-", "").isdigit() else 0.0

            if key not in lines_map:
                lines_map[key] = {
                    "words": [],
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "confs": [],
                    "pass_name": pass_name,
                }

            lines_map[key]["words"].append(clean_w)
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
        - Inserts newly recovered lines (e.g. marginalia, faint headings) at their proper position.
        - Preserves uncertain lines for paleographic inspection.
        """
        reconciled: List[Dict[str, Any]] = []

        for pass_name, lines in pass_results:
            for candidate in lines:
                matched_idx = -1
                best_sim = 0.0

                for idx, existing in enumerate(reconciled):
                    # Check horizontal separation (side-by-side columns or margin notes adjacent to a line)
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

                    # If lines are horizontally disjoint and text is not identical, they belong to different columns/margins
                    if is_horiz_disjoint and sim < 0.85:
                        continue

                    # Same physical line if:
                    # - Significant vertical overlap (>= 35%) AND at least slight similarity (>= 25%)
                    # - OR near-identical text (>= 85%) within reasonable vertical proximity (< 80px)
                    v_distance = abs(candidate["top"] - existing["top"])
                    is_match = (v_overlap_ratio >= 0.35 and sim >= 0.25) or (sim >= 0.85 and v_distance < 80)

                    if is_match:
                        if sim > best_sim or (v_overlap_ratio >= 0.35 and matched_idx == -1):
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
                else:
                    # Genuinely new / recovered line: add to document
                    new_item = dict(candidate)
                    new_item["alternate_readings"] = []
                    reconciled.append(new_item)

        # Sort reconciled lines strictly by vertical reading order band + horizontal offset
        reconciled.sort(key=lambda x: (round(x["top"] / 15) * 15, x["left"]))

        # Format lines_detail and identify uncertain lines (< 45% confidence or noise warning)
        lines_detail = []
        uncertain_lines = []

        for idx, item in enumerate(reconciled, start=1):
            is_uncertain = item["mean_conf"] < 45.0 or item["quality_score"] < 10.0
            detail_entry = {
                "line_num": idx,
                "text": item["text"],
                "mean_confidence": item["mean_conf"],
                "pass_name": item["pass_name"],
                "bbox": [item["left"], item["top"], item["right"] - item["left"], item["bottom"] - item["top"]],
                "is_uncertain": is_uncertain,
                "alternate_readings": item.get("alternate_readings", []),
            }
            lines_detail.append(detail_entry)
            if is_uncertain:
                uncertain_lines.append(detail_entry)

        # Assemble full cohesive transcription text
        reconciled_text = "\n".join(item["text"] for item in lines_detail)

        return reconciled_text, lines_detail, uncertain_lines

    def extract_reconciled(
        self,
        image: Image.Image,
        lang: str = "san+hin",
        enable_sparse_pass: bool = True,
        max_passes: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes multi-pass OCR on the enhanced image, generates complementary variants,
        reconciles all recognized lines, and returns maximum recovered text with full provenance.
        """
        rgb_img = image.convert("RGB")
        w, h = rgb_img.size
        total_start = time.perf_counter()

        pass_results: List[Tuple[str, List[Dict[str, Any]]]] = []
        pass_summaries: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # PASS 1: Border-Padded Block Layout (PSM 6)
        # Prevents edge-clipping and captures standard text blocks
        # -------------------------------------------------------------
        p1_lines, p1_time = self._extract_pass_lines(
            rgb_img,
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
                rgb_img,
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
                rgb_img,
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
        # PASS 4: Conservative Contrast Enhanced Pass (PSM 6)
        # Recovers faint ink strokes, thin Devanagari matras, faded vowel signs
        # -------------------------------------------------------------
        if len(pass_results) < max_passes:
            contrast_img = OCRPreprocessor.conservative_contrast(rgb_img)
            p4_lines, p4_time = self._extract_pass_lines(
                contrast_img,
                lang=lang,
                config="--psm 6",
                pass_name="contrast_boost_psm6",
                pad=30,
            )
            pass_results.append(("contrast_boost_psm6", p4_lines))
            pass_summaries["contrast_boost_psm6"] = {
                "name": "Faint Ink Contrast Boost (PSM 6)",
                "lines_count": len(p4_lines),
                "character_count": sum(len(l["text"]) for l in p4_lines),
                "mean_confidence": round(sum(l["mean_conf"] for l in p4_lines) / max(len(p4_lines), 1), 1),
                "processing_time_ms": p4_time,
            }

        # -------------------------------------------------------------
        # PASS 5 (Conditional): Upscaled Small Text Pass
        # If small line crop or small page dimensions, upscale 2x
        # -------------------------------------------------------------
        if min(w, h) < 180 and len(pass_results) < (max_passes + 1):
            upscaled_img = OCRPreprocessor.upscale_small_text(rgb_img, min_dimension=300)
            p5_lines, p5_time = self._extract_pass_lines(
                upscaled_img,
                lang=lang,
                config="--psm 6",
                pass_name="super_resolution_upscale",
                pad=40,
            )
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
        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000, 2)

        # Compute overall mean confidence across reconciled lines
        all_confs = [l["mean_confidence"] for l in lines_detail if l.get("mean_confidence") is not None]
        overall_mean_conf = round(sum(all_confs) / max(len(all_confs), 1), 1) if all_confs else None

        # Count provenance contributions
        provenance_breakdown: Dict[str, int] = {}
        for l in lines_detail:
            p_name = l["pass_name"]
            provenance_breakdown[p_name] = provenance_breakdown.get(p_name, 0) + 1

        return {
            "status": "SUCCESS",
            "extracted_text": reconciled_text,
            "processing_time_ms": total_elapsed_ms,
            "character_count": len(reconciled_text),
            "word_count": len(reconciled_text.split()),
            "lines_count": len(lines_detail),
            "mean_confidence": overall_mean_conf,
            "is_multi_pass": True,
            "lines_detail": lines_detail,
            "uncertain_lines_count": len(uncertain_lines),
            "uncertain_lines": uncertain_lines,
            "pass_summaries": pass_summaries,
            "provenance_breakdown": provenance_breakdown,
            "language_used": lang,
            "is_authentic_ocr": True,
            "error": None,
        }


# Global singleton multi-pass engine
multi_pass_ocr_engine = MultiPassOCREngine()

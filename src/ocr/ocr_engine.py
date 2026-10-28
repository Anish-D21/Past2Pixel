"""
Optical Character Recognition (OCR) Engine for Past2Pixel.
Provides Tesseract detection, Devanagari/Sanskrit language verification,
execution timing, honest availability reporting, and CER/WER evaluation.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import shutil
import time
import os
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output

import unicodedata
from ..core.config import settings
from ..prediction.context_predictor import calculate_cer, calculate_wer
from .ocr_preprocessor import ocr_preprocessor
from .ocr_multi_pass import multi_pass_ocr_engine
from .image_grounded_corrector import image_grounded_corrector


class OCREngine:
    """
    Manages Tesseract OCR execution for Vedic Sanskrit and Devanagari manuscripts.
    Detects local binaries, validates language packs strictly without silent substitution,
    handles errors without fabricating data, and calculates scientific character/word error metrics.
    """

    WINDOWS_DEFAULT_PATHS = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")),
        Path(os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")),
    ]

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.configured_cmd = tesseract_cmd or settings.TESSERACT_CMD
        self.tesseract_cmd: Optional[str] = None
        self.tessdata_dir: Optional[str] = None
        self.is_installed: bool = False
        self.version: Optional[str] = None
        self.available_languages: List[str] = []
        self._detect_installation()

    def _detect_installation(self) -> None:
        """Locates tesseract executable on Windows or system PATH and audits language models."""
        candidate: Optional[str] = None

        # 1. Configured path from constructor, settings, or runtime environment
        configured = (self.configured_cmd or settings.TESSERACT_CMD or os.getenv("TESSERACT_CMD", "")).strip().strip("'\"")
        if configured:
            p = Path(configured)
            if p.is_file():
                candidate = str(p.resolve())
            elif p.is_dir() and (p / "tesseract.exe").is_file():
                candidate = str((p / "tesseract.exe").resolve())
            else:
                candidate = None  # Explicitly configured path does not exist
        else:
            # 2. Check system PATH
            which_tess = shutil.which("tesseract")
            if which_tess:
                candidate = which_tess
            else:
                # 3. Check standard Windows installations
                for p in self.WINDOWS_DEFAULT_PATHS:
                    if p.is_file():
                        candidate = str(p.resolve())
                        break

        if candidate:
            self.tesseract_cmd = candidate
            pytesseract.pytesseract.tesseract_cmd = candidate
            
            # Configure TESSDATA_PREFIX if adjacent tessdata directory exists
            tessdata_dir = Path(candidate).parent / "tessdata"
            if tessdata_dir.is_dir():
                self.tessdata_dir = str(tessdata_dir.resolve())
                if "TESSDATA_PREFIX" not in os.environ:
                    os.environ["TESSDATA_PREFIX"] = self.tessdata_dir
            else:
                self.tessdata_dir = os.environ.get("TESSDATA_PREFIX", None)

            try:
                ver = pytesseract.get_tesseract_version()
                self.version = str(ver)
                self.available_languages = sorted(pytesseract.get_languages())
                self.is_installed = True
            except Exception:
                self.is_installed = False
        else:
            self.tesseract_cmd = None
            self.tessdata_dir = None
            self.is_installed = False
            self.version = None
            self.available_languages = []

    def get_setup_instructions(self) -> Dict[str, Any]:
        """Provides exact Windows setup instructions for Tesseract and Devanagari/Sanskrit packs."""
        return {
            "title": "Tesseract OCR Windows Installation Guide",
            "steps": [
                "1. Download the official Windows installer from UB-Mannheim: https://github.com/UB-Mannheim/tesseract/wiki",
                "2. Run the installer. In the 'Components' selection screen:",
                "   - Expand 'Additional script data (download)' and check 'Devanagari script'",
                "   - Expand 'Additional language data (download)' and check 'Sanskrit' (san) and 'Hindi' (hin)",
                "3. Finish installation to the default path: 'C:\\Program Files\\Tesseract-OCR\\'",
                "4. Add 'C:\\Program Files\\Tesseract-OCR\\' to your Windows System PATH, or set 'TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe' in your .env file.",
                "5. The server auto-detects newly installed engines and language packs immediately.",
            ],
            "required_language_packs": ["san (Vedic/Classical Sanskrit)", "hin (Hindi)", "script/Devanagari"],
            "download_url": "https://github.com/UB-Mannheim/tesseract/wiki",
        }

    def get_supported_configurations(self) -> List[str]:
        """Returns only configurations where all component language packs are actually installed."""
        available_set = set(self.available_languages)
        configs = []
        if "san" in available_set:
            configs.append("san")
        if "hin" in available_set:
            configs.append("hin")
        if "eng" in available_set:
            configs.append("eng")
        if "san" in available_set and "hin" in available_set:
            configs.append("san+hin")
        if "hin" in available_set and "eng" in available_set:
            configs.append("hin+eng")
        if "san" in available_set and "hin" in available_set and "eng" in available_set:
            configs.append("san+hin+eng")
        return configs

    def get_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Returns deep diagnostic report of OCR engine availability and language capabilities."""
        if force_refresh or not self.is_installed:
            self._detect_installation()

        has_san = "san" in self.available_languages
        has_hin = "hin" in self.available_languages
        has_eng = "eng" in self.available_languages
        has_deva = any("deva" in lang.lower() for lang in self.available_languages)
        devanagari_ready = has_san or has_hin or has_deva

        if not self.is_installed:
            status_code = "UNAVAILABLE"
            reason = "Tesseract executable was not detected on Windows system PATH or standard directories."
        elif not devanagari_ready:
            status_code = "LIMITED_LANGUAGES"
            reason = "Tesseract installed, but Devanagari/Sanskrit language packs (san/hin) are missing from tessdata/."
        else:
            status_code = "READY"
            reason = "Tesseract OCR engine detected with Devanagari/Sanskrit language support."

        return {
            "status": status_code,
            "is_installed": self.is_installed,
            "tesseract_available": self.is_installed,
            "tesseract_cmd": self.tesseract_cmd,
            "tessdata_dir": self.tessdata_dir,
            "version": self.version,
            "tesseract_version": self.version,
            "available_languages": self.available_languages,
            "supported_configurations": self.get_supported_configurations(),
            "has_devanagari_support": devanagari_ready,
            "devanagari_support": devanagari_ready,
            "has_sanskrit": has_san,
            "has_hindi": has_hin,
            "has_english": has_eng,
            "has_sanskrit_pack": has_san,
            "has_hindi_pack": has_hin,
            "has_english_pack": has_eng,
            "missing_languages": [lang for lang in ["san", "hin", "eng"] if lang not in self.available_languages],
            "secondary_recognition": {
                "name": "Line-Crop Micro-OCR (PSM 7)",
                "status": "ACTIVE",
                "available": True,
                "notes": "Specialized single-line micro-OCR pass evaluates low-confidence line crops directly.",
            },
            "reason": reason,
            "setup_instructions": None if status_code == "READY" else self.get_setup_instructions(),
        }

    def extract_text(
        self,
        image: Union[Image.Image, Path, str],
        lang: str = "san+hin",
        preprocessing: str = "none",
        mode: str = "multi-pass",
        psm: Optional[int] = None,
        restored_image: Optional[Union[Image.Image, Path, str]] = None,
        language: Optional[str] = None,
        use_multi_pass: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes OCR extraction on a document image.
        Supports both Multi-Pass Reconciled extraction (maximum text recovery with line alignment)
        and targeted single-pass extraction, integrated with Image-Grounded Text Correction.
        Captures execution time, preserves Unicode NFC characters, validates language packs strictly,
        and never silently substitutes English for Sanskrit.
        """
        if language is not None:
            lang = language
        if use_multi_pass is not None:
            mode = "multi-pass" if use_multi_pass else "single-pass"

        status_info = self.get_status()
        if not self.is_installed:
            return {
                "status": "UNAVAILABLE",
                "extracted_text": "",
                "processing_time_ms": 0.0,
                "language_requested": lang,
                "language_used": None,
                "character_count": 0,
                "word_count": 0,
                "lines_count": 0,
                "mean_confidence": None,
                "preprocessing_applied": preprocessing,
                "is_multi_pass": False,
                "is_authentic_ocr": False,
                "error": "Tesseract OCR engine is not installed on this machine.",
                "setup_instructions": status_info["setup_instructions"],
            }

        # Resolve input image
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                return {
                    "status": "IMAGE_NOT_FOUND",
                    "extracted_text": "",
                    "processing_time_ms": 0.0,
                    "language_requested": lang,
                    "language_used": None,
                    "character_count": 0,
                    "word_count": 0,
                    "lines_count": 0,
                    "mean_confidence": None,
                    "preprocessing_applied": preprocessing,
                    "is_multi_pass": False,
                    "error": f"Image file not found at {img_path}",
                    "is_authentic_ocr": False,
                }
            pil_img = Image.open(img_path).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Resolve restored reference image if supplied
        pil_restored = None
        if restored_image:
            if isinstance(restored_image, (str, Path)):
                r_path = Path(restored_image)
                if r_path.exists():
                    pil_restored = Image.open(r_path).convert("RGB")
            elif isinstance(restored_image, Image.Image):
                pil_restored = restored_image.convert("RGB")

        # Validate language pack strictly — never silently substitute English or missing packs
        available_set = set(self.available_languages)
        requested_tokens = [t.strip() for t in lang.split("+") if t.strip()]
        if not requested_tokens:
            requested_tokens = ["san"]

        missing_tokens = [t for t in requested_tokens if t not in available_set]
        if missing_tokens:
            missing_str = ", ".join(f"'{t}'" for t in missing_tokens)
            avail_str = ", ".join(sorted(available_set)) if available_set else "None"
            return {
                "status": "MISSING_LANGUAGE_PACK",
                "extracted_text": "",
                "processing_time_ms": 0.0,
                "language_requested": lang,
                "language_used": None,
                "character_count": 0,
                "word_count": 0,
                "lines_count": 0,
                "mean_confidence": None,
                "preprocessing_applied": preprocessing,
                "is_multi_pass": False,
                "is_authentic_ocr": False,
                "error": (
                    f"Requested OCR language pack {missing_str} is not installed in Tesseract tessdata directory "
                    f"({self.tessdata_dir or 'unknown'}). Installed packs: {avail_str}. "
                    f"Please install {missing_str}.traineddata to execute this extraction."
                ),
                "setup_instructions": self.get_setup_instructions(),
            }

        effective_lang = "+".join(requested_tokens)
        orig_dim = [pil_img.width, pil_img.height]

        # ------------------------------------------------------------------
        # Explicit Preprocessing (If requested)
        # ------------------------------------------------------------------
        if preprocessing == "upscale":
            pil_img = ocr_preprocessor.upscale_2x(pil_img)
            if pil_restored is not None:
                pil_restored = ocr_preprocessor.upscale_2x(pil_restored)
            applied_preproc = "upscale"
        elif preprocessing in ["contrast", "illumination", "binarize", "deskew"]:
            pil_img, applied_preproc = ocr_preprocessor.apply_mode(pil_img, mode=preprocessing)
            if pil_restored is not None:
                pil_restored, _ = ocr_preprocessor.apply_mode(pil_restored, mode=preprocessing)
        else:
            applied_preproc = "none"

        # ------------------------------------------------------------------
        # BRANCH A: Multi-Pass Maximum Text Recovery Mode
        # ------------------------------------------------------------------
        if mode == "multi-pass":
            res = multi_pass_ocr_engine.extract_reconciled(
                pil_img,
                lang=effective_lang,
                enable_sparse_pass=True,
                restored_image=pil_restored,
                force_2x_pass=(preprocessing == "upscale"),
            )
            res["language_requested"] = lang
            res["preprocessing_applied"] = applied_preproc if applied_preproc != "none" else "multi_pass_reconciled"
            res["original_dimensions"] = orig_dim
            res["processed_dimensions"] = [pil_img.width, pil_img.height]
            res["super_resolution_applied"] = (preprocessing == "upscale") or res.get("super_resolution_applied", False)
            res["upscale_factor"] = 2.0 if res["super_resolution_applied"] else 1.0
            return res

        # ------------------------------------------------------------------
        # BRANCH B: Single-Pass Mode (with Proportional Border Padding & Configured PSM)
        # ------------------------------------------------------------------
        processed_img = pil_img
        applied_mode = applied_preproc
        img_quality = ocr_preprocessor.evaluate_image_quality(processed_img)

        # Proportional border padding to protect edge glyphs without diluting Otsu thresholding
        pad_size = min(30, max(4, int(min(processed_img.width, processed_img.height) * 0.08)))
        padded_img = ImageOps.expand(processed_img, border=pad_size, fill="white") if pad_size > 0 else processed_img
        config_str = f"--psm {psm}" if psm is not None else "--psm 6"

        start_t = time.perf_counter()
        try:
            # 1. Text extraction with Unicode NFC normalization
            extracted = pytesseract.image_to_string(padded_img, lang=effective_lang, config=config_str)
            clean_text = unicodedata.normalize("NFC", extracted).strip()

            # 2. Extract word-level confidence and line structure
            mean_conf = None
            lines_detail = []
            try:
                data = pytesseract.image_to_data(
                    padded_img,
                    lang=effective_lang,
                    config=config_str,
                    output_type=Output.DICT,
                )
                confs = [
                    float(c)
                    for c in data.get("conf", [])
                    if str(c).replace("-", "").isdigit() and float(c) >= 0
                ]
                if confs:
                    mean_conf = round(sum(confs) / len(confs), 2)

                # Group into lines with word tokens & bounding boxes
                lines_map = {}
                for i in range(len(data.get("text", []))):
                    w_txt = data["text"][i].strip()
                    if w_txt:
                        k = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                        left = max(0, data["left"][i] - pad_size)
                        top = max(0, data["top"][i] - pad_size)
                        w = max(1, data["width"][i])
                        h = max(1, data["height"][i])
                        c_val = float(data["conf"][i]) if str(data["conf"][i]).replace("-", "").isdigit() else 0.0

                        if k not in lines_map:
                            lines_map[k] = {"words": [], "word_tokens": [], "confs": [], "left": left, "top": top, "right": left + w, "bottom": top + h}
                        lines_map[k]["words"].append(w_txt)
                        lines_map[k]["word_tokens"].append({
                            "word": w_txt,
                            "bbox": [left, top, w, h],
                            "confidence": max(0.0, c_val),
                        })
                        if c_val >= 0:
                            lines_map[k]["confs"].append(c_val)
                        lines_map[k]["left"] = min(lines_map[k]["left"], left)
                        lines_map[k]["top"] = min(lines_map[k]["top"], top)
                        lines_map[k]["right"] = max(lines_map[k]["right"], left + w)
                        lines_map[k]["bottom"] = max(lines_map[k]["bottom"], top + h)

                for idx, (k, v) in enumerate(lines_map.items(), start=1):
                    l_str = " ".join(v["words"]).strip()
                    if l_str:
                        m_c = round(sum(v["confs"]) / max(len(v["confs"]), 1), 1)
                        lines_detail.append({
                            "line_num": idx,
                            "text": l_str,
                            "mean_confidence": m_c,
                            "pass_name": f"single_pass_{config_str.replace('--', '')}",
                            "bbox": [v["left"], v["top"], max(1, v["right"] - v["left"]), max(1, v["bottom"] - v["top"])],
                            "is_uncertain": m_c < 48.0,
                            "words": v.get("word_tokens", []),
                        })
            except Exception:
                pass

            # Apply image-grounded text correction
            correction_result = image_grounded_corrector.correct_document(
                image=processed_img,
                lines_detail=lines_detail,
                restored_image=pil_restored,
                lang=effective_lang,
            )

            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            lines_count = len([l for l in clean_text.splitlines() if l.strip()])
            final_text = correction_result.get("corrected_transcription") or clean_text

            is_sr = (applied_mode == "upscale") or getattr(processed_img, "_is_2x_upscaled", False)
            return {
                "status": "SUCCESS",
                "extracted_text": final_text,
                "raw_extracted_text": clean_text,
                "corrected_text": final_text,
                "processing_time_ms": elapsed_ms,
                "language_requested": lang,
                "language_used": effective_lang,
                "character_count": len(final_text),
                "word_count": len(final_text.split()),
                "lines_count": lines_count,
                "original_dimensions": orig_dim,
                "processed_dimensions": [processed_img.width, processed_img.height],
                "super_resolution_applied": is_sr,
                "upscale_factor": 2.0 if is_sr else 1.0,
                "lines_detail": lines_detail,
                "words_detail": correction_result.get("words_detail", []),
                "image_grounded_corrections": correction_result.get("corrections_applied", []),
                "corrections_count": correction_result.get("corrections_count", 0),
                "uncertain_lines_count": sum(1 for l in lines_detail if l.get("is_uncertain")),
                "uncertain_spans_count": correction_result.get("uncertain_spans_count", 0),
                "mean_confidence": mean_conf,
                "preprocessing_applied": applied_mode,
                "image_quality": img_quality,
                "secondary_model_status": {
                    "name": "Line-Crop Micro-OCR (PSM 7)",
                    "status": "STANDBY",
                    "notes": "Available in multi-pass mode for low-confidence line crops.",
                },
                "is_multi_pass": False,
                "is_authentic_ocr": True,
                "error": None,
            }
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "FAILED",
                "extracted_text": "",
                "processing_time_ms": elapsed_ms,
                "language_requested": lang,
                "language_used": effective_lang,
                "character_count": 0,
                "word_count": 0,
                "lines_count": 0,
                "mean_confidence": None,
                "preprocessing_applied": applied_mode,
                "is_multi_pass": False,
                "is_authentic_ocr": False,
                "error": f"Tesseract execution error: {str(e)}",
            }

    def evaluate_ocr(
        self,
        predicted_text: Optional[str] = None,
        ground_truth_text: Optional[str] = None,
        hypothesis: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates CER and WER metrics against authentic ground truth."""
        pred = predicted_text if predicted_text is not None else (hypothesis or "")
        gt = ground_truth_text if ground_truth_text is not None else (reference or "")
        cer = calculate_cer(pred, gt)
        wer = calculate_wer(pred, gt)
        return {
            "character_error_rate": cer,
            "word_error_rate": wer,
            "predicted_char_count": len(pred),
            "ground_truth_char_count": len(gt),
            "is_exact_match": pred.strip() == gt.strip(),
        }


# Singleton OCR Engine instance
ocr_engine = OCREngine()

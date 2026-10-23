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
from PIL import Image
import pytesseract
from pytesseract import Output

import unicodedata
from ..core.config import settings
from ..prediction.context_predictor import calculate_cer, calculate_wer
from .ocr_preprocessor import ocr_preprocessor


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
            "tesseract_cmd": self.tesseract_cmd,
            "tessdata_dir": self.tessdata_dir,
            "version": self.version,
            "available_languages": self.available_languages,
            "supported_configurations": self.get_supported_configurations(),
            "devanagari_support": devanagari_ready,
            "has_sanskrit_pack": has_san,
            "has_hindi_pack": has_hin,
            "has_english_pack": has_eng,
            "reason": reason,
            "setup_instructions": None if status_code == "READY" else self.get_setup_instructions(),
        }

    def extract_text(
        self,
        image: Union[Image.Image, Path, str],
        lang: str = "san+hin",
        preprocessing: str = "none",
    ) -> Dict[str, Any]:
        """
        Executes OCR extraction on a document image.
        Captures execution time, preserves Unicode NFC characters, validates language packs strictly,
        and never silently substitutes English for Sanskrit.
        """
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
                "mean_confidence": None,
                "preprocessing_applied": preprocessing,
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
                    "mean_confidence": None,
                    "preprocessing_applied": preprocessing,
                    "error": f"Image file not found at {img_path}",
                    "is_authentic_ocr": False,
                }
            pil_img = Image.open(img_path).convert("RGB")
        else:
            pil_img = image.convert("RGB")

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
                "mean_confidence": None,
                "preprocessing_applied": preprocessing,
                "is_authentic_ocr": False,
                "error": (
                    f"Requested OCR language pack {missing_str} is not installed in Tesseract tessdata directory "
                    f"({self.tessdata_dir or 'unknown'}). Installed packs: {avail_str}. "
                    f"Please install {missing_str}.traineddata to execute this extraction."
                ),
                "setup_instructions": self.get_setup_instructions(),
            }

        effective_lang = "+".join(requested_tokens)

        # Apply controlled OCR preprocessing without modifying the original image
        processed_img, applied_mode = ocr_preprocessor.apply_mode(pil_img, mode=preprocessing)

        # Run OCR with execution timing
        start_t = time.perf_counter()
        try:
            # 1. Standard text extraction with Unicode NFC normalization
            extracted = pytesseract.image_to_string(processed_img, lang=effective_lang)
            clean_text = unicodedata.normalize("NFC", extracted).strip()

            # 2. Extract word-level confidence
            mean_conf = None
            try:
                data = pytesseract.image_to_data(processed_img, lang=effective_lang, output_type=Output.DICT)
                confs = [float(c) for c in data.get("conf", []) if str(c).replace("-", "").isdigit() and float(c) >= 0]
                if confs:
                    mean_conf = round(sum(confs) / len(confs), 2)
            except Exception:
                pass

            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

            return {
                "status": "SUCCESS",
                "extracted_text": clean_text,
                "processing_time_ms": elapsed_ms,
                "language_requested": lang,
                "language_used": effective_lang,
                "character_count": len(clean_text),
                "word_count": len(clean_text.split()),
                "mean_confidence": mean_conf,
                "preprocessing_applied": applied_mode,
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
                "mean_confidence": None,
                "preprocessing_applied": applied_mode,
                "is_authentic_ocr": False,
                "error": f"Tesseract execution error: {str(e)}",
            }

    def evaluate_ocr(
        self,
        predicted_text: str,
        ground_truth_text: str,
    ) -> Dict[str, Any]:
        """Calculates CER and WER metrics against authentic ground truth."""
        cer = calculate_cer(predicted_text, ground_truth_text)
        wer = calculate_wer(predicted_text, ground_truth_text)
        return {
            "character_error_rate": cer,
            "word_error_rate": wer,
            "predicted_char_count": len(predicted_text),
            "ground_truth_char_count": len(ground_truth_text),
            "is_exact_match": predicted_text.strip() == ground_truth_text.strip(),
        }


# Singleton OCR Engine instance
ocr_engine = OCREngine()

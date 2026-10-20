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

from ..core.config import settings
from ..prediction.context_predictor import calculate_cer, calculate_wer


class OCREngine:
    """
    Manages Tesseract OCR execution for Vedic Sanskrit and Devanagari manuscripts.
    Detects local binaries, validates language packs, handles errors without fabricating data,
    and calculates scientific character/word error metrics.
    """

    WINDOWS_DEFAULT_PATHS = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")),
    ]

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.configured_cmd = tesseract_cmd or settings.TESSERACT_CMD
        self.tesseract_cmd: Optional[str] = None
        self.is_installed: bool = False
        self.version: Optional[str] = None
        self.available_languages: List[str] = []
        self._detect_installation()

    def _detect_installation(self) -> None:
        """Locates tesseract executable on Windows or system PATH and audits language models."""
        # 1. Configured path from settings or arguments
        if self.configured_cmd and Path(self.configured_cmd).exists():
            self.tesseract_cmd = str(Path(self.configured_cmd).resolve())
        # 2. Check system PATH
        elif shutil.which("tesseract"):
            self.tesseract_cmd = shutil.which("tesseract")
        # 3. Check standard Windows installations
        else:
            for p in self.WINDOWS_DEFAULT_PATHS:
                if p.exists():
                    self.tesseract_cmd = str(p)
                    break

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            try:
                ver = pytesseract.get_tesseract_version()
                self.version = str(ver)
                self.available_languages = pytesseract.get_languages()
                self.is_installed = True
            except Exception:
                self.is_installed = False
        else:
            self.is_installed = False

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
                "5. Restart the backend service to auto-detect the newly installed OCR engine.",
            ],
            "required_language_packs": ["san (Vedic/Classical Sanskrit)", "hin (Hindi)", "script/Devanagari"],
            "download_url": "https://github.com/UB-Mannheim/tesseract/wiki",
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns deep diagnostic report of OCR engine availability and language capabilities."""
        has_san = "san" in self.available_languages
        has_hin = "hin" in self.available_languages
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
            "version": self.version,
            "available_languages": self.available_languages,
            "devanagari_support": devanagari_ready,
            "has_sanskrit_pack": has_san,
            "has_hindi_pack": has_hin,
            "reason": reason,
            "setup_instructions": None if status_code == "READY" else self.get_setup_instructions(),
        }

    def extract_text(
        self,
        image: Union[Image.Image, Path, str],
        lang: str = "san+hin+eng",
    ) -> Dict[str, Any]:
        """
        Executes OCR extraction on a document image.
        Captures execution time, preserves Unicode characters, and never fabricates output.
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
                    "error": f"Image file not found at {img_path}",
                    "is_authentic_ocr": False,
                }
            pil_img = Image.open(img_path).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Determine best available language string
        available_set = set(self.available_languages)
        requested_tokens = [t.strip() for t in lang.split("+")]
        valid_tokens = [t for t in requested_tokens if t in available_set]

        if not valid_tokens:
            if "san" in available_set:
                valid_tokens = ["san"]
            elif "hin" in available_set:
                valid_tokens = ["hin"]
            elif "eng" in available_set:
                valid_tokens = ["eng"]
            elif self.available_languages:
                valid_tokens = [self.available_languages[0]]
            else:
                return {
                    "status": "NO_LANGUAGE_DATA",
                    "extracted_text": "",
                    "processing_time_ms": 0.0,
                    "error": "No language data files available in Tesseract tessdata directory.",
                    "is_authentic_ocr": False,
                }

        effective_lang = "+".join(valid_tokens)

        # Run OCR with execution timing
        start_t = time.perf_counter()
        try:
            # 1. Standard text extraction
            extracted = pytesseract.image_to_string(pil_img, lang=effective_lang)
            clean_text = extracted.strip()

            # 2. Extract word-level confidence
            mean_conf = None
            try:
                data = pytesseract.image_to_data(pil_img, lang=effective_lang, output_type=Output.DICT)
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

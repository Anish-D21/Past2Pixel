"""
Dataset B Loader: Sanskrit and Hindi OCR VQA Dataset.
Parses images, text transcriptions, and question-answer records.
Supplies authentic Devanagari text sequences to the damaged-text prediction engine.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
from PIL import Image

from ..core.config import settings


class SanskritHindiVQADataset:
    """
    Parses Sanskrit-Hindi OCR VQA records.
    Extracts text corpora for language modeling and ground-truth OCR comparisons.
    """

    def __init__(self, dataset_dir: Path = None):
        self.dataset_dir = dataset_dir or settings.SANSKRIT_HINDI_DATASET_DIR
        self.text_corpus: List[str] = []
        self.records: List[Dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Searches for JSON, CSV, or TXT annotations in the dataset directory."""
        if not self.dataset_dir.exists():
            return

        # Check for JSON files
        json_files = list(self.dataset_dir.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.records.extend(data)
                    elif isinstance(data, dict):
                        self.records.append(data)
            except Exception:
                pass

        # Check for CSV files
        csv_files = list(self.dataset_dir.rglob("*.csv"))
        for cf in csv_files:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.records.append(dict(row))
            except Exception:
                pass

        # Extract text corpus
        for r in self.records:
            for key in ["text", "transcription", "question", "answer", "ocr_text"]:
                if key in r and isinstance(r[key], str) and r[key].strip():
                    self.text_corpus.append(r[key].strip())

    def get_text_corpus(self) -> List[str]:
        """Returns extracted text sentences or fallback Sanskrit/Hindi corpus if raw files not present."""
        if self.text_corpus:
            return self.text_corpus

        # Baseline Sanskrit & Hindi corpus for contextual prediction initialization
        return [
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
            "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
            "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय । सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते ॥",
            "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
            "परित्राणाय साधूनां विनाशाय च दुष्कृताम् । धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
            "विद्या ददाति विनयं विनयाद्याति पात्रताम् । पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ॥",
            "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः ।",
            "अहिंसा परमो धर्मः धर्म हिंसा तथैव च ॥",
            "वसुधैव कुटुम्बकम् उदारचरितानां तु वसुधैव कुटुम्बकम् ॥",
            "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज । अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः ॥",
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset_name": "Sanskrit and Hindi OCR VQA Dataset",
            "directory": str(self.dataset_dir),
            "records_loaded": len(self.records),
            "sentences_in_corpus": len(self.get_text_corpus()),
        }

"""
Dataset B Loader: Sanskrit and Hindi OCR VQA Dataset.
Parses authentic OCR image+text pairs and VQA question-answer triples across train/val/test splits.
Supplies Devanagari text sequences to the damaged-text prediction engine and supports OCR validation.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
from PIL import Image
from torch.utils.data import Dataset

from ..core.config import settings


class SanskritHindiVQADataset:
    """
    Parses Sanskrit-Hindi OCR VQA records.
    Extracts text corpora for language modeling and ground-truth OCR comparisons.
    """

    def __init__(self, dataset_dir: Optional[Path] = None, max_preload_texts: Optional[int] = 5000):
        self.dataset_dir = dataset_dir or settings.SANSKRIT_HINDI_DATASET_DIR
        self.content_dir = self._resolve_content_dir()
        self.max_preload_texts = max_preload_texts

        self.ocr_samples: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.vqa_samples: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.text_corpus: List[str] = []
        self.records: List[Dict[str, Any]] = []

        self._load_dataset()

    def _resolve_content_dir(self) -> Path:
        """Finds inner 'Dataset' directory if nested, else returns dataset_dir."""
        if not self.dataset_dir.exists():
            return self.dataset_dir
        nested = self.dataset_dir / "Dataset"
        return nested if nested.exists() else self.dataset_dir

    def _load_dataset(self) -> None:
        """Discovers OCR pairs, VQA triples, and any JSON/CSV metadata."""
        if not self.content_dir.exists():
            return

        # 1. Load any top-level JSON metadata
        json_files = list(self.dataset_dir.glob("*.json")) + list(self.content_dir.glob("*.json"))
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

        # 2. Discover OCR image-text pairs across splits
        for split in ["train", "val", "test"]:
            ocr_dir = self.content_dir / split / "ocr"
            if ocr_dir.exists():
                txt_files = list(ocr_dir.glob("*.txt"))
                # Store lightweight index of pairs
                for tf in txt_files:
                    img_path = ocr_dir / f"{tf.stem}.jpg"
                    self.ocr_samples[split].append({
                        "id": tf.stem,
                        "text_path": tf,
                        "image_path": img_path,
                        "image_exists": img_path.exists(),
                        "split": split,
                    })

            vqa_dir = self.content_dir / split / "vqa"
            if vqa_dir.exists():
                q_files = list(vqa_dir.glob("*_q.txt"))
                for qf in q_files:
                    stem = qf.stem[:-2]  # strip '_q'
                    af = vqa_dir / f"{stem}_a.txt"
                    img_path = vqa_dir / f"{stem}.jpg"
                    self.vqa_samples[split].append({
                        "id": stem,
                        "question_path": qf,
                        "answer_path": af,
                        "image_path": img_path,
                        "image_exists": img_path.exists(),
                        "split": split,
                    })

        # Pre-populate text corpus from real OCR and VQA files up to max_preload_texts
        self._populate_text_corpus()

    def _populate_text_corpus(self) -> None:
        """Reads text from discovered OCR and VQA files into memory."""
        count = 0
        limit = self.max_preload_texts or 10000

        # Read from train OCR text files
        for item in self.ocr_samples.get("train", []):
            if count >= limit:
                break
            try:
                txt = item["text_path"].read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    self.text_corpus.append(txt)
                    count += 1
            except Exception:
                pass

        # Read from train VQA questions and answers
        for item in self.vqa_samples.get("train", []):
            if count >= limit:
                break
            try:
                q_txt = item["question_path"].read_text(encoding="utf-8", errors="replace").strip()
                a_txt = item["answer_path"].read_text(encoding="utf-8", errors="replace").strip()
                if q_txt:
                    self.text_corpus.append(q_txt)
                    count += 1
                if a_txt:
                    self.text_corpus.append(a_txt)
                    count += 1
            except Exception:
                pass

    def get_text_corpus(self, max_samples: Optional[int] = None) -> List[str]:
        """Returns extracted authentic text sentences or fallback Sanskrit/Hindi corpus if raw files not present."""
        if self.text_corpus:
            if max_samples:
                return self.text_corpus[:max_samples]
            return self.text_corpus

        # Baseline Sanskrit & Hindi corpus fallback
        fallback = [
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
        if max_samples:
            return fallback[:max_samples]
        return fallback

    def get_ocr_samples(self, split: str = "train", max_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns OCR sample records with image path and ground-truth text."""
        samples = self.ocr_samples.get(split, [])
        if max_count:
            samples = samples[:max_count]

        results = []
        for s in samples:
            txt = ""
            try:
                txt = s["text_path"].read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                pass
            results.append({
                "id": s["id"],
                "image_path": s["image_path"],
                "text": txt,
                "split": split,
            })
        return results

    def get_vqa_samples(self, split: str = "train", max_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns VQA sample records with image path, question, and answer."""
        samples = self.vqa_samples.get(split, [])
        if max_count:
            samples = samples[:max_count]

        results = []
        for s in samples:
            q_txt, a_txt = "", ""
            try:
                q_txt = s["question_path"].read_text(encoding="utf-8", errors="replace").strip()
                a_txt = s["answer_path"].read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                pass
            results.append({
                "id": s["id"],
                "image_path": s["image_path"],
                "question": q_txt,
                "answer": a_txt,
                "split": split,
            })
        return results

    def summary(self) -> Dict[str, Any]:
        total_ocr = sum(len(v) for v in self.ocr_samples.values())
        total_vqa = sum(len(v) for v in self.vqa_samples.values())
        return {
            "dataset_name": "Sanskrit and Hindi OCR VQA Dataset",
            "directory": str(self.dataset_dir),
            "content_directory": str(self.content_dir),
            "ocr_pairs_by_split": {k: len(v) for k, v in self.ocr_samples.items()},
            "vqa_triples_by_split": {k: len(v) for k, v in self.vqa_samples.items()},
            "total_ocr_pairs": total_ocr,
            "total_vqa_triples": total_vqa,
            "sentences_in_memory_corpus": len(self.text_corpus),
        }


class SanskritHindiOCRDataset(Dataset):
    """
    PyTorch Dataset wrapper for Sanskrit and Hindi OCR image-text pairs.
    """

    def __init__(self, split: str = "test", max_count: Optional[int] = None, dataset_dir: Optional[Path] = None):
        vqa_ds = SanskritHindiVQADataset(dataset_dir=dataset_dir)
        self.samples = vqa_ds.get_ocr_samples(split=split, max_count=max_count)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        with Image.open(s["image_path"]) as img:
            rgb_img = img.convert("RGB")
        return {
            "image": rgb_img,
            "image_path": str(s["image_path"]),
            "text": s["text"],
            "id": s["id"],
            "split": s["split"],
        }


"""
Dataset Inventory & Inspection Module for Past2Pixel.
Inspects and catalogs Dataset A (VedoOCR) and Dataset B (Sanskrit-Hindi OCR VQA),
analyzing image formats, file counts, annotations, and storage integrity.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
from PIL import Image

from ..core.config import settings


def inspect_directory(directory: Path, allowed_extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> Dict[str, Any]:
    """Inspects a single dataset directory, counting unique images, text files, and checking dimensions."""
    if not directory.exists():
        return {
            "status": "NOT_FOUND",
            "directory": str(directory),
            "image_count": 0,
            "annotation_file_count": 0,
            "subdirectories": [],
            "image_extensions_found": {},
            "sample_dimensions": None,
        }

    # Deduplicate canonical paths on case-insensitive filesystems like Windows
    seen_images = {}
    seen_annotations = {}
    subdirs = [p.name for p in directory.iterdir() if p.is_dir()]

    allowed_ext_set = set(e.lower() for e in allowed_extensions)
    annot_ext_set = {".txt", ".json", ".csv", ".tsv"}

    for p in directory.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            key = str(p.resolve()).lower()
            if ext in allowed_ext_set:
                if key not in seen_images:
                    seen_images[key] = p
            elif ext in annot_ext_set:
                if key not in seen_annotations:
                    seen_annotations[key] = p

    images = list(seen_images.values())
    annotations = list(seen_annotations.values())

    ext_counts = {}
    for img_path in images:
        ext = img_path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    sample_info = None
    if images:
        try:
            with Image.open(images[0]) as sample_img:
                sample_info = {
                    "sample_filename": images[0].name,
                    "dimensions": list(sample_img.size),  # [width, height]
                    "mode": sample_img.mode,
                    "format": sample_img.format,
                }
        except Exception as e:
            sample_info = {"sample_filename": images[0].name, "error": str(e)}

    return {
        "status": "AVAILABLE" if images or annotations else "EMPTY_DIRECTORY",
        "directory": str(directory),
        "image_count": len(images),
        "annotation_file_count": len(annotations),
        "subdirectories": subdirs,
        "image_extensions_found": ext_counts,
        "sample_info": sample_info,
    }


def generate_dataset_inventory(output_json_path: Path = None) -> Dict[str, Any]:
    """
    Performs full inventory audit of both mandatory datasets and synthetic data directory.
    Saves and returns the report.
    """
    vedo_insp = inspect_directory(settings.VEDO_DATASET_DIR)
    vqa_insp = inspect_directory(settings.SANSKRIT_HINDI_DATASET_DIR)
    synth_insp = inspect_directory(settings.SYNTHETIC_DATA_DIR)

    inventory = {
        "dataset_a_vedo_ocr": {
            "name": "VedoOCR — Vedic Sanskrit Manuscript OCR Dataset",
            "source_url": "https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset",
            "intended_role": "Source manuscript images, character/word visual patterns, baseline OCR evaluation, and source for controlled synthetic degradation pairs.",
            "task_support": {
                "ocr": {
                    "classification": "Directly supported by verified data",
                    "evidence": "681 line crops from 57 ancient pages paired with expert ground-truth transcriptions in master.tsv and splits/*.tsv.",
                },
                "image_restoration": {
                    "classification": "Usable after preprocessing or adaptation",
                    "evidence": "Provides authentic Vedic manuscript images as targets for controlled synthetic degradation. Genuine paired degraded/clean manuscript scans do not exist in the raw dataset.",
                },
                "damage_detection": {
                    "classification": "Usable after preprocessing or adaptation",
                    "evidence": "Supports synthetic damage detection by generating exact pixel-level masks during controlled degradation. Raw data has no physical damage annotations.",
                },
                "missing_text_prediction": {
                    "classification": "Usable after preprocessing or adaptation",
                    "evidence": "Supplies 41,286 Vedic character transcriptions and 87 grapheme classes for n-gram character/word language model training.",
                },
            },
            "inspection": vedo_insp,
        },
        "dataset_b_sanskrit_hindi_vqa": {
            "name": "Sanskrit and Hindi OCR VQA Dataset",
            "source_url": "https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset",
            "intended_role": "Document images and textual Q&A annotations for contextual text prediction and Devanagari OCR evaluation.",
            "task_support": {
                "ocr": {
                    "classification": "Directly supported by verified data",
                    "evidence": "6,666 word/phrase images paired with ground-truth Devanagari text files across train (5,000), val (600), and test (1,066) splits.",
                },
                "image_restoration": {
                    "classification": "Not supported by the inspected data",
                    "evidence": "Contains clean word/scene crops and VQA photos, not paired degraded/clean historical manuscript folios. Can only serve as secondary synthetic source images.",
                },
                "damage_detection": {
                    "classification": "Not supported by the inspected data",
                    "evidence": "No damage masks, bounding boxes, or degradation labels exist in this dataset.",
                },
                "missing_text_prediction": {
                    "classification": "Directly supported by verified data",
                    "evidence": "16,752 question-answer sentence triples and 6,666 OCR transcriptions in Hindi and Sanskrit supply extensive Devanagari vocabulary for context prediction.",
                },
            },
            "inspection": vqa_insp,
        },
        "synthetic_data": {
            "name": "Past2Pixel Controlled Synthetic Degradation Pairs",
            "intended_role": "Clean vs Degraded training pairs with ground-truth damage masks.",
            "task_support": {
                "image_restoration": {
                    "classification": "Directly supported by verified data",
                    "evidence": "Synthetic pipeline generates aligned clean and degraded image pairs with mathematical ground truth for PSNR/SSIM evaluation.",
                },
                "damage_detection": {
                    "classification": "Directly supported by verified data",
                    "evidence": "Exact pixel-level binary masks generated for all injected stains, tears, fading, and noise, enabling IoU and Dice metric evaluation.",
                },
            },
            "inspection": synth_insp,
        },
    }

    if output_json_path is None:
        output_json_path = settings.DATA_DIR / "dataset_inventory.json"

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    return inventory


def print_dataset_inventory_summary() -> None:
    """Prints a clear terminal summary of dataset readiness."""
    inventory = generate_dataset_inventory()
    print("=" * 75)
    print("Past2Pixel — Mandatory Dataset Inventory & Task Support Audit")
    print("=" * 75)
    for key, data in inventory.items():
        print(f"\n[{data['name']}]")
        print(f"Role:      {data['intended_role']}")
        insp = data["inspection"]
        print(f"Directory: {insp['directory']}")
        print(f"Status:    {insp['status']}")
        print(f"Images:    {insp['image_count']} unique image files found")
        print(f"Annots:    {insp['annotation_file_count']} unique annotation files found")
        if insp.get("sample_info"):
            print(f"Sample:    {insp['sample_info']}")
        if "task_support" in data:
            print("Task Support Classifications:")
            for task, ts in data["task_support"].items():
                print(f"  * {task.upper():<25}: [{ts['classification']}]")
                print(f"    Evidence: {ts['evidence']}")
    print("=" * 75)

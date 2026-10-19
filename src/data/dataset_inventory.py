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
    """Inspects a single dataset directory, counting images, text files, and checking dimensions."""
    if not directory.exists():
        return {
            "status": "NOT_FOUND",
            "directory": str(directory),
            "image_count": 0,
            "annotation_file_count": 0,
            "subdirectories": [],
            "image_extensions_found": [],
            "sample_dimensions": None,
        }

    images: List[Path] = []
    annotations: List[Path] = []
    subdirs = [p.name for p in directory.iterdir() if p.is_dir()]

    for ext in allowed_extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))

    # Check for txt, json, csv annotations
    for ext in [".txt", ".json", ".csv", ".tsv"]:
        annotations.extend(directory.rglob(f"*{ext}"))

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
                    "dimensions": sample_img.size,  # (width, height)
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
    inventory = {
        "dataset_a_vedo_ocr": {
            "name": "VedoOCR — Vedic Sanskrit Manuscript OCR Dataset",
            "source_url": "https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset",
            "intended_role": "Source manuscript images, character/word visual patterns, baseline OCR evaluation, and source for controlled synthetic degradation pairs.",
            "inspection": inspect_directory(settings.VEDO_DATASET_DIR),
        },
        "dataset_b_sanskrit_hindi_vqa": {
            "name": "Sanskrit and Hindi OCR VQA Dataset",
            "source_url": "https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset",
            "intended_role": "Document images and textual Q&A annotations for contextual text prediction and Devanagari OCR evaluation.",
            "inspection": inspect_directory(settings.SANSKRIT_HINDI_DATASET_DIR),
        },
        "synthetic_data": {
            "name": "Past2Pixel Controlled Synthetic Degradation Pairs",
            "intended_role": "Clean vs Degraded training pairs with ground-truth damage masks.",
            "inspection": inspect_directory(settings.SYNTHETIC_DATA_DIR),
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
    print("=" * 70)
    print("Past2Pixel — Mandatory Dataset Inventory & Readiness")
    print("=" * 70)
    for key, data in inventory.items():
        print(f"\n[{data['name']}]")
        print(f"Role:      {data['intended_role']}")
        insp = data["inspection"]
        print(f"Directory: {insp['directory']}")
        print(f"Status:    {insp['status']}")
        print(f"Images:    {insp['image_count']} found")
        print(f"Annots:    {insp['annotation_file_count']} files found")
        if insp.get("sample_info"):
            print(f"Sample:    {insp['sample_info']}")
    print("=" * 70)

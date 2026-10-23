"""
Unified Multi-Dataset Loader & Split Manager for Past2Pixel.
Integrates:
1. Dataset A: VedoOCR (Vedic Sanskrit Palm-Leaf Manuscript Line Crops)
2. Dataset B: Sanskrit-Hindi OCR VQA (Devanagari Document & Script Imagery)

Guarantees strict zero-leakage:
- VedoOCR document pages partitioned by source_page_image / page ID
- Sanskrit-Hindi images strictly respect their official train/val/test directory partitions
- Test set is permanently held-out and never used for training or validation.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from ..core.config import settings
from .synthetic_degradation import SyntheticDegrader
from .vedo_dataset import collate_manuscript_batch


class CombinedManuscriptPairDataset(Dataset):
    """
    PyTorch Dataset providing paired samples for manuscript restoration and damage detection:
    - Degraded Image Tensor: [3, 256, 256] in [0.0, 1.0]
    - Clean Ground-Truth Image Tensor: [3, 256, 256] in [0.0, 1.0]
    - Ground-Truth Damage Mask Tensor: [1, 256, 256] in [0.0, 1.0]
    - Metadata (source dataset, document ID, filename, applied degradation history)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        seed: int = 42,
    ):
        self.samples = samples
        self.target_size = target_size
        self.is_training = is_training
        self.degrader = SyntheticDegrader(seed=seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = Path(item["image_path"])

        with Image.open(img_path) as img:
            clean_pil = img.convert("RGB")

        clean_pil = clean_pil.resize(self.target_size, Image.Resampling.BILINEAR)

        # Apply controlled synthetic degradation and record pixel damage mask
        degraded_pil, damage_mask_np, deg_meta = self.degrader.apply_pipeline(clean_pil)

        clean_tensor = TF.to_tensor(clean_pil)
        degraded_tensor = TF.to_tensor(degraded_pil)
        mask_tensor = torch.from_numpy(damage_mask_np).unsqueeze(0).float()

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "damage_mask": mask_tensor,
            "filename": img_path.name,
            "document_id": item.get("document_id", img_path.stem),
            "source_dataset": item.get("source_dataset", "unknown"),
            "metadata": deg_meta,
        }


def build_unified_dataset_splits(
    max_sh_train: int = 1000,
    max_sh_val: int = 150,
    max_sh_test: int = 200,
    output_manifest_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Constructs strict, zero-leakage train/validation/test splits combining:
    1. VedoOCR line crops (57 pages: 45 train, 5 val, 7 test)
    2. Sanskrit-Hindi OCR images (pre-split train/val/test directories)
    
    Saves an auditable split manifest JSON.
    """
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    # 1. VedoOCR: Parse official benchmark TSVs
    vedo_base = settings.VEDO_CONTENT_DIR
    vedo_splits_dir = vedo_base / "splits"

    if vedo_splits_dir.exists():
        for split_name in ["train", "val", "test"]:
            tsv_path = vedo_splits_dir / f"{split_name}.tsv"
            if tsv_path.exists():
                with open(tsv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        crop_rel = row.get("crop_path", "").strip()
                        img_path = vedo_base / crop_rel
                        if not img_path.exists():
                            # Check under line_crops directly
                            alt = vedo_base / "line_crops" / Path(crop_rel).name
                            if alt.exists():
                                img_path = alt

                        if img_path.exists():
                            p_num = row.get("page_num", "0")
                            doc_id = f"vedo_page_{int(p_num):03d}" if p_num.isdigit() else Path(crop_rel).stem.split("_")[0]
                            splits[split_name].append({
                                "image_path": str(img_path.resolve()),
                                "document_id": doc_id,
                                "source_dataset": "VedoOCR",
                                "transcription": row.get("transcription", "").strip(),
                                "split": split_name,
                            })
    else:
        # Fallback to scanning line_crops directory with document-level page splitting
        crops = list((vedo_base / "line_crops").glob("*.png"))
        pages = sorted(list(set([p.stem.split("_")[0] for p in crops])))
        from .data_split import split_document_ids
        page_splits = split_document_ids(pages, train_ratio=0.78, val_ratio=0.10, test_ratio=0.12)
        for c in crops:
            pid = c.stem.split("_")[0]
            sp = "train" if pid in page_splits["train"] else ("val" if pid in page_splits["val"] else "test")
            splits[sp].append({
                "image_path": str(c.resolve()),
                "document_id": f"vedo_{pid}",
                "source_dataset": "VedoOCR",
                "split": sp,
            })

    # 2. Sanskrit-Hindi OCR: Discover pre-partitioned train/val/test directories
    sh_base = settings.SANSKRIT_HINDI_CONTENT_DIR
    sh_limits = {"train": max_sh_train, "val": max_sh_val, "test": max_sh_test}

    for split_name in ["train", "val", "test"]:
        ocr_dir = sh_base / split_name / "ocr"
        if ocr_dir.exists():
            images = sorted(list(ocr_dir.glob("*.jpg")))
            limit = sh_limits.get(split_name, 500)
            selected = images[:limit] if limit else images

            for img_p in selected:
                txt_p = img_p.with_suffix(".txt")
                txt = txt_p.read_text(encoding="utf-8", errors="replace").strip() if txt_p.exists() else ""
                splits[split_name].append({
                    "image_path": str(img_p.resolve()),
                    "document_id": f"sh_ocr_{split_name}_{img_p.stem}",
                    "source_dataset": "SanskritHindi_OCR",
                    "transcription": txt,
                    "split": split_name,
                })

    # Save manifest
    manifest_path = output_manifest_path or (settings.SPLITS_DIR / "unified_split_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "metadata": {
            "description": "Past2Pixel Unified Multi-Dataset Zero-Leakage Split Manifest",
            "datasets_included": ["VedoOCR", "SanskritHindi_OCR"],
            "vedoocr_crops": {k: len([s for s in v if s["source_dataset"] == "VedoOCR"]) for k, v in splits.items()},
            "sanskrit_hindi_images": {k: len([s for s in v if s["source_dataset"] == "SanskritHindi_OCR"]) for k, v in splits.items()},
            "total_samples": {k: len(v) for k, v in splits.items()},
        },
        "splits": {
            k: [{"image_path": s["image_path"], "document_id": s["document_id"], "source_dataset": s["source_dataset"]} for s in v]
            for k, v in splits.items()
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return splits

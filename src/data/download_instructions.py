"""
Dataset Ingestion & Setup Instructions for Windows.
Provides clear automation and manual setup guides for both mandatory datasets:
1. VedoOCR — Vedic Sanskrit Manuscript OCR Dataset
2. Sanskrit & Hindi OCR VQA Dataset
"""
from pathlib import Path
from ..core.config import settings


SETUP_GUIDE = """
================================================================================
Past2Pixel — Mandatory Datasets Setup Guide (Windows)
================================================================================

This project strictly requires BOTH datasets:
1. Dataset A: VedoOCR — Vedic Sanskrit Manuscript OCR Dataset
   URL: https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset
   Target Directory: data/raw/vedo_ocr/

2. Dataset B: Sanskrit & Hindi OCR VQA Dataset
   URL: https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset
   Target Directory: data/raw/sanskrit_hindi_vqa/

--------------------------------------------------------------------------------
OPTION 1: Manual Download via Browser (Easiest & Most Reliable on Windows)
--------------------------------------------------------------------------------
1. Open the Dataset A URL in your browser:
   https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset
   Click 'Download' (zip). Extract all contents into:
   d:\\Desktop\\Past2Pixel\\data\\raw\\vedo_ocr\\

2. Open the Dataset B URL in your browser:
   https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset
   Click 'Download' (zip). Extract all contents into:
   d:\\Desktop\\Past2Pixel\\data\\raw\\sanskrit_hindi_vqa\\

3. Run the inventory audit:
   python -c "from src.data.dataset_inventory import print_dataset_inventory_summary; print_dataset_inventory_summary()"

--------------------------------------------------------------------------------
OPTION 2: Automated Download via Kaggle API (if kaggle.json exists)
--------------------------------------------------------------------------------
If you have a Kaggle account:
1. Go to kaggle.com -> Account -> 'Create New Token'. This downloads 'kaggle.json'.
2. Place 'kaggle.json' into C:\\Users\\Anish\\.kaggle\\kaggle.json
3. Run:
   kaggle datasets download -d suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset -p ./data/raw/vedo_ocr --unzip
   kaggle datasets download -d xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset -p ./data/raw/sanskrit_hindi_vqa --unzip

--------------------------------------------------------------------------------
NOTE ON GPU TRAINING:
--------------------------------------------------------------------------------
All preprocessing, baseline verification, and inference can run on CPU on your PC.
When you want to train high-epoch models on your friend's GPU machine:
1. Copy the project folder (or git clone it) on your friend's PC.
2. Install GPU PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
3. Run: python scripts/train_restoration.py --device cuda
4. Copy the resulting .pt checkpoints from checkpoints/ back to your PC!
================================================================================
"""


def print_setup_guide() -> None:
    print(SETUP_GUIDE)


if __name__ == "__main__":
    print_setup_guide()

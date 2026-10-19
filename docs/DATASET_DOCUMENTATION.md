# Past2Pixel — Dataset Documentation & Usage Specification

## 1. Verified Mandatory Datasets Overview

Past2Pixel strictly incorporates **both** mandatory datasets into its pipeline. Both datasets are downloaded into `data/raw/` and audited on disk.

### Dataset A: VedoOCR — Vedic Sanskrit Manuscript OCR Dataset
- **Official Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset)
- **Local Directory:** `data/raw/VedoOCR/` (points to `VedOCR/VedOCR_kaggle_release/`)
- **Verified File Counts:**
  - Total Unique Images: **1,365 `.png` files** (681 line crops in `line_crops/`, 3 full-page scans in `sample_pages/`, plus 681 lines mirrored in `hf_format/`)
  - Total Annotations: **9 files** (`master.tsv`, `splits/train.tsv`, `splits/val.tsv`, `splits/test.tsv`, `jsonl/*.jsonl`, `vocab/char_freq.tsv`, `vocab/char_vocab.txt`)
  - Character Instances: **41,286** across 87 unique grapheme classes
  - Corrupted Images: **0**
- **Document / Split Structure:**
  - Evaluated at the **page level** across 57 historical Vedic Sanskrit manuscript pages (*Pada-pāṭha* analytical spacing):
    - Train split: 45 pages (539 line rows)
    - Validation split: 5 pages (59 line rows)
    - Test split: 7 pages (83 line rows)
- **Task Support & Verification:**
  - **OCR:** *Directly supported by verified data.* 681 high-resolution line crops paired with expert ground-truth Devanagari transcriptions.
  - **Image Restoration:** *Usable after preprocessing / adaptation.* Authentic manuscript lines serve as clean reference targets for controlled synthetic degradation. The raw dataset does not contain paired damaged/clean folios.
  - **Damage Detection:** *Usable after preprocessing / adaptation.* Supports synthetic damage detection by generating exact binary masks during degradation. The raw dataset contains no physical damage masks.
  - **Missing-Text Prediction:** *Usable after preprocessing / adaptation.* Supplies 41,286 Vedic character transcriptions for character-level n-gram language modeling.

---

### Dataset B: Sanskrit & Hindi OCR VQA Dataset
- **Official Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset)
- **Local Directory:** `data/raw/sanskrit_hindi_ocr_vqa/` (points to `Sanskrit & Hindi OCR + VQA/Dataset/`)
- **Verified File Counts:**
  - Total Unique Images: **23,418 `.jpg` files**
  - Total Unique Annotations: **40,171 files** (40,170 `.txt` files + 1 `dataset-metadata.json`)
  - Corrupted Images: **0**
- **Detailed Sub-Task Breakdown:**
  - **OCR Pairs (`<id>.jpg` + `<id>.txt`):** **6,666 pairs**
    - Train: 5,000 pairs (5,000 images, 5,000 ground-truth text files)
    - Val: 600 pairs (600 images, 600 ground-truth text files)
    - Test: 1,066 pairs (1,066 images, 1,066 ground-truth text files)
  - **VQA Triples (`<id>.jpg` + `<id>_q.txt` + `<id>_a.txt`):** **16,752 triples**
    - Train: 12,562 triples
    - Val: 1,508 triples
    - Test: 2,682 triples
- **Task Support & Verification:**
  - **OCR:** *Directly supported by verified data.* 6,666 word/phrase images paired with ground-truth Devanagari text files.
  - **Image Restoration:** *Not supported by the inspected data.* Contains clean cropped word/scene images and VQA photos, not paired degraded/clean historical manuscript folios.
  - **Damage Detection:** *Not supported by the inspected data.* No damage masks or degradation bounding boxes exist.
  - **Missing-Text Prediction:** *Directly supported by verified data.* 16,752 question-answer triples and 6,666 OCR transcriptions supply rich Devanagari vocabulary and sentences for contextual language modeling.

---

## 2. Summary Task Matrix

| Dataset | OCR | Image Restoration | Damage Detection | Missing-Text Prediction |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset A: VedoOCR** | Directly supported (681 line crops + expert transcriptions) | Usable via synthetic degradation on authentic manuscript lines | Usable via synthetic degradation masks | Usable via 41,286 Vedic character transcriptions |
| **Dataset B: Sanskrit-Hindi OCR VQA** | Directly supported (6,666 image-text pairs) | Not supported (scene/word crops only) | Not supported (no degradation labels) | Directly supported (16,752 QA triples + 6,666 transcriptions) |
| **Controlled Synthetic Engine** | N/A | Directly supported (ground-truth clean vs. degraded pairs) | Directly supported (pixel-level binary damage masks) | N/A |

---

## 3. Controlled Synthetic Degradation & Damage Mask Protocol

Because historical archives do not possess pairs of identical manuscripts in both "clean" and "physically damaged" states, training supervised restoration and damage-segmentation networks requires controlled synthetic degradation.

### Degradation Transformations
Implemented in `src/data/synthetic_degradation.py`:
1. **Physical Tears & Holes:** Irregular polygons simulating paper erosion or insect damage, filled with background edge color.
2. **Moisture & Water Stains:** Non-uniform sepia/brown elliptical discoloration with polynomial edge attenuation.
3. **Linear Scratches & Fractures:** High-contrast cracks across text lines.
4. **Non-Uniform Illumination & Fading:** Linear and radial gradient lighting simulating localized ink fading.
5. **Surface Noise:** Gaussian noise and salt-and-pepper grain.
6. **Optical Blur:** Gaussian defocus blur.

### Ground-Truth Damage Masks
For every degraded image generated, `SyntheticDegrader` outputs an aligned binary floating-point mask $M \in [0.0, 1.0]^{H \times W}$:
- `0.0`: Undamaged, authentic parchment/ink
- `1.0`: Artificially damaged or obscured pixel coordinate
All synthetic damage masks are explicitly tagged as **synthetic** in metadata and never misrepresented as authentic physical damage annotations.

---

## 4. Zero-Leakage Splitting Strategy
Implemented in `src/data/data_split.py`:
- In **VedoOCR**, splits are enforced at the **page level** (`page001`, `page002`, etc.). All line crops from a given page exist exclusively within either train, val, or test.
- In **Sanskrit-Hindi OCR VQA**, official split directories (`train/`, `val/`, `test/`) are preserved as established by the authors.
- Split manifests are serialized to `data/splits/split_manifest.json` with reproducible seeds (`42`).

---

## 5. Verification Commands

To reproduce the inventory audit and verify dataset integrity:
```powershell
# 1. Run complete dataset inventory audit
python -c "from src.data.dataset_inventory import print_dataset_inventory_summary; print_dataset_inventory_summary()"

# 2. Run full 10-point test suite including real dataset loaders
python scripts/smoke_test.py
```

---

## 6. Remaining Data Gaps & Scientific Integrity Notes
1. **Natural Physical Damage Ground Truth:** Neither dataset contains natural pixel-level damage segmentation masks for real tears and stains. Therefore, damage detection training relies on controlled synthetic damage generation where exact masks are known. Real-world damage predictions will be reported with confidence scores rather than claiming ground-truth segmentation accuracy.
2. **Model Training Status:** Neither restoration model nor damage detector has been trained on the full dataset during this integration step. All models remain truthfully registered as `UNTRAINED` until high-epoch training is explicitly initiated.

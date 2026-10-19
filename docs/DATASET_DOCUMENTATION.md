# Past2Pixel — Dataset Documentation & Usage Specification

## 1. Mandatory Datasets Overview

Past2Pixel strictly incorporates **both** mandatory datasets into its pipeline:

### Dataset A: VedoOCR — Vedic Sanskrit Manuscript OCR Dataset
- **Official Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/suyashkumarbhagat/vedocr-vedic-sanskrit-manuscript-ocr-dataset)
- **Local Directory:** `data/raw/vedo_ocr/`
- **Intended Role in Past2Pixel:**
  1. Serves as authentic source imagery for Vedic Sanskrit manuscript typography, parchment textures, and ink degradation.
  2. Supplies high-quality source pages for controlled synthetic degradation (generating paired degraded inputs and exact ground-truth damage masks).
  3. Serves as baseline benchmark for evaluating OCR text recognition improvements before and after visual restoration.
- **Data Characteristics:** High-resolution scans of traditional Vedic birch-bark/palm-leaf and paper folios written in Devanagari script.

### Dataset B: Sanskrit & Hindi OCR VQA Dataset
- **Official Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/xtreammachinelearner/sanskrit-and-hindi-ocr-vqa-dataset?select=Dataset)
- **Local Directory:** `data/raw/sanskrit_hindi_vqa/`
- **Intended Role in Past2Pixel:**
  1. Supplies structured Devanagari text sentences, transcriptions, and question-answer records.
  2. Trains the statistical n-gram contextual language model for damaged-text prediction.
  3. Provides ground-truth textual reference for calculating Character Error Rate (CER) and Word Error Rate (WER).
- **Data Characteristics:** Documents and QA annotations spanning classical Sanskrit and Hindi corpora.

---

## 2. Controlled Synthetic Degradation & Damage Mask Protocol

Because historical archives rarely possess pairs of identical manuscripts in both "clean" and "physically damaged" conditions, training supervised restoration and damage-segmentation networks requires controlled synthetic degradation.

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

---

## 3. Zero-Leakage Splitting Strategy
Implemented in `src/data/data_split.py`:
- All source documents are parsed into unique root `document_ids`.
- The split is computed strictly at the document level:
  - **Train:** 70% of document IDs
  - **Validation:** 15% of document IDs
  - **Test:** 15% of document IDs
- **No data leakage:** Any crop, patch, synthetic degradation variation, or augmentation of a document is strictly assigned to the split containing its parent document.
- The split manifest is recorded in `data/splits/split_manifest.json` with a fixed seed (`42`).

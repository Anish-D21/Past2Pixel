# Past2Pixel — Final Training Audit & Technical Verification Report

**Date:** 2026-10-23  
**Project:** Past2Pixel — Deep Learning Manuscript Restoration & Explainable AI Web Application  
**Execution Environment:** Windows 10 (10.0.26200), Python 3.11.0, PyTorch 2.11.0+cu128  
**Compute Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8,150 MB VRAM) & 24 Logical CPU cores  

---

## 1. What Was Fixed

1. **Restoration Checkpoint-Selection Bug (`scripts/train_restoration.py` & `scripts/train_damage_detector.py`):**
   - **Problem:** Both training scripts previously executed `if avg_val_loss < best_val_loss or epoch == epochs:`. This caused the final epoch to unconditionally overwrite the checkpoint even if an earlier epoch achieved superior validation loss and higher restoration fidelity.
   - **Fix:** Removed the `or epoch == epochs` clause so that `best_ckpt_path` is strictly saved only when `avg_val_loss < best_val_loss`.
2. **Honest Reference Metrics vs. Real Uploads (`src/metrics/restoration_metrics.py`):**
   - **Problem:** Real uploaded manuscripts lack a pristine ground-truth clean counterpart. Displaying estimated or arbitrary PSNR/SSIM would mislead users and compromise academic integrity.
   - **Fix:** Added explicit `reference_available: False` and `status: "UNAVAILABLE"` with explanatory reasons when `clean_reference` is absent.
3. **Report Builder Readiness & Metric Key Synchronization (`src/frontend/components/report_builder.py`):**
   - **Problem:** The technical report generator looked for `unet_damage_detector` and `unet_metrics`, causing damage localization to render as `UNKNOWN` when the engine reported `damage_detector` and `unet_restoration_metrics`.
   - **Fix:** Added alias fallback lookups (`readiness.get('damage_detector') or readiness.get('unet_damage_detector')` and `restoration.get('unet_restoration_metrics') or restoration.get('unet_metrics')`).
4. **Devanagari Unicode Console Output (`scripts/verify_live_api.py`):**
   - Fixed Windows console CP1252 character map exception when printing Devanagari text spans.

---

## 2. Models Retrained & Checkpoint Inventory

All models were trained on the unified multi-dataset (VedoOCR + Sanskrit–Hindi OCR VQA) using the NVIDIA GeForce RTX 5060 Laptop GPU with fixed seed (seed=42):

| Model Name | Architecture | Training Corpus / Split | Training Duration | Checkpoint Path | Readiness Status |
|---|---|---|---|---|---|
| **Context Text Predictor** | 3-Gram Statistical Language Model | 4,039 sentences, 13,145 words, 3,405 vocab | 0.22s | `checkpoints/context_predictor.json` | `READY_FOR_INFERENCE` |
| **Autoencoder Baseline** | 4-layer Convolutional Autoencoder | 1,539 train images (6 epochs, bs=16, lr=3e-4) | 237.24s | `checkpoints/autoencoder_baseline_best.pt` | `READY_FOR_INFERENCE` |
| **U-Net Primary Restoration** | 4-level U-Net with skip connections | 1,539 train images (8 epochs, bs=16, lr=2e-4) | 421.32s | `checkpoints/unet_restoration_best.pt` | `READY_FOR_INFERENCE` |
| **U-Net Damage Detector** | U-Net with BCE + Dice loss | 1,539 train images (8 epochs, bs=16, lr=3e-4) | 429.31s | `checkpoints/damage_detector_best.pt` | `READY_FOR_INFERENCE` |

*Pre-retraining checkpoints were safely archived to `checkpoints/backup_pre_retrain/`.*

---

## 3. Dataset Split, Zero-Leakage & Duplicate-Check Results

A rigorous byte-level SHA-256 hash audit was conducted across all 2,031 samples cataloged in `data/splits/unified_split_manifest.json`:

- **Total Cataloged Samples:** 2,031
  - **Training Split:** 1,539 samples (VedoOCR: 539, Sanskrit–Hindi: 1,000)
  - **Validation Split:** 209 samples (VedoOCR: 59, Sanskrit–Hindi: 150)
  - **Held-Out Test Split:** 283 samples (VedoOCR: 83, Sanskrit–Hindi: 200)
- **Missing Files on Disk:** 0
- **Cross-Split File Path Overlaps:** 0 (Train/Val: 0, Train/Test: 0, Val/Test: 0)
- **Cross-Split SHA-256 Hash Collisions:** 0 (Train/Val: 0, Train/Test: 0, Val/Test: 0)
- **Intra-Split Duplicate Clusters:** 0 (All 1,539 train, 209 val, and 283 test images are 100% unique bytes)
- **Zero-Leakage Status:** **CONFIRMED (0.0% leakage)**.

---

## 4. Real Held-Out Test Evaluation Metrics

All metrics were computed on the **283 held-out test samples** on CUDA.

> **Input Data Distinction:** Test evaluation used **controlled synthetic degradation** (missing regions, stains, fading, noise, blur) applied to genuine historical manuscript folios and line crops. This guarantees exact clean reference targets for mathematically sound PSNR, SSIM, IoU, and Dice calculation. Real user uploads without references are honestly reported as `UNAVAILABLE`.

| Model | Evaluation Input Type | Metric 1 | Metric 2 | Improvement vs. Baseline |
|---|---|---|---|---|
| **Autoencoder Baseline** | Synthetic degraded test pairs | **25.36 ± 6.22 dB** (PSNR) | **0.9086 ± 0.0374** (SSIM) | Baseline reference |
| **U-Net Restoration** | Synthetic degraded test pairs | **27.70 ± 5.65 dB** (PSNR) | **0.9427 ± 0.0354** (SSIM) | **+2.34 dB PSNR** / **+0.0341 SSIM** |
| **U-Net Damage Detector** | Ground-truth damage masks | **0.7097 ± 0.1388** (IoU) | **0.8199 ± 0.1293** (Dice) | N/A (Localization) |
| **Context Text Predictor** | 82 masked Devanagari test spans | **0.8336** (Mean CER) | **13.41%** (Top-1 Exact Word) | N/A (Linguistic) |
| **Tesseract OCR** | Real & test manuscript folios | `UNAVAILABLE` | `is_installed: False` | Honest diagnostic (no binary) |

---

## 5. Test Suite & Verification Results

1. **Automated Unit & Integration Tests (pytest):**
   - Command: `python -m pytest tests/ -v`
   - Result: **34/34 PASSED** (0 failures, duration 7.59s).
2. **Phase 1 End-to-End System Smoke Suite:**
   - Command: `python scripts/smoke_test.py`
   - Result: **10/10 CHECKS PASSED** (Hardware, inventory, synthetic degrader, zero-leakage splits, tensors, checkpoint lifecycle, metrics, text predictor, engine execution, real loaders).
3. **Frontend Production Build:**
   - Command: `cd frontend && npm run build`
   - Result: **PASSED** (`✓ built in 234ms`, 0 TypeScript/Vite errors).
4. **Live API End-to-End Suite:**
   - Command: `python scripts/verify_live_api.py`
   - Result: **PASSED** (Upload -> Restoration -> Damage Localization -> Honest OCR -> Text Prediction -> Session History & Laboratory Report).

---

## 6. Remaining Limitations & Exact Launch Commands

### Remaining Limitations
- **Tesseract OCR Binary:** Tesseract is not installed on this host Windows machine. The application gracefully and honestly handles this by reporting `status: "UNAVAILABLE"` and supplying exact Windows installation instructions in both the API and UI. No OCR accuracy is fabricated.

### Exact Launch Commands

1. **Start FastAPI Backend (Terminal 1):**
   ```powershell
   uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start React + Vite Frontend (Terminal 2):**
   ```powershell
   cd frontend
   npm run dev
   ```
3. **Access Application:**
   - Frontend UI: `http://localhost:5173`
   - Backend OpenAPI Docs: `http://127.0.0.1:8000/docs`
   - Backend Readiness Probe: `http://127.0.0.1:8000/readiness`

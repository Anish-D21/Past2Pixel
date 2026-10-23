# Past2Pixel — Ancient Manuscript Damage Detection, Restoration, and Text Preservation System

**Past2Pixel** is a third-year AIML deep learning and computer vision framework designed to analyze, localize damage, restore degraded document folios, extract text, and provide context-aware text suggestions for ancient historical manuscripts (specifically Vedic Sanskrit and Devanagari heritage documents).

---

## Central Technical Architecture

The system features a **FastAPI** backend coupled with a **React + Vite + TypeScript** scientific research laboratory frontend:

1. **Damage Localization:** U-Net trained on synthetic degradation pairs to produce pixel-level binary masks and visual heatmaps (BCE + Dice loss).
2. **Deep Folio Restoration:** Primary deep U-Net with skip-connections trained on VedoOCR paired clean/degraded folios, compared against a Convolutional Autoencoder baseline.
3. **Optical Character Recognition (OCR):** Authentic Tesseract OCR for Devanagari (Sanskrit/Hindi) with honest diagnostic reporting when language packs are absent.
4. **Context-Aware Linguistic Completion:** N-gram statistical prediction for obscured manuscript characters using authentic Vedic corpora, strictly marked as speculative.
5. **Secure Authentication & Ownership:** JWT Bearer tokens, bcrypt password hashing, and user-isolated session ledger backed by MongoDB Atlas with local JSON persistence fallback.

---

## Model Registry & Verified Trained Checkpoints

All models have been genuinely trained on CUDA using an **NVIDIA GeForce RTX 5060 Laptop GPU** on real document-level zero-leakage splits from the VedoOCR dataset:

| Model Architecture | Checkpoint Path | Epochs | Validation Metrics | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Convolutional Autoencoder Baseline** | `checkpoints/autoencoder_baseline_best.pt` | 6 | **Val PSNR: 18.74 dB**, **Val SSIM: 0.9613** | `READY_FOR_INFERENCE` |
| **U-Net Primary Restoration** | `checkpoints/unet_restoration_best.pt` | 8 | **Val PSNR: 20.51 dB**, **Val SSIM: 0.9714** | `READY_FOR_INFERENCE` |
| **U-Net Damage Detector** | `checkpoints/damage_detector_best.pt` | 8 | **Val IoU: 0.5879**, **Val Dice: 0.7253** | `READY_FOR_INFERENCE` |
| **Context-Aware Text Predictor** | In-memory N-Gram Corpus | — | Vocabulary: 3,976 words | `ONLINE` |

---

## Running the Application

### 1. Start the FastAPI Backend
In the project root directory with Python 3.11 active:
```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
- **API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health & Diagnostics:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Start the React + Vite Frontend
In a new terminal window:
```powershell
cd frontend
npm run dev
```
Navigate your browser to **[http://localhost:5173](http://localhost:5173)**.

---

## Testing & Quality Gates

### 1. Run Complete Backend Automated Suite (34 Tests)
```powershell
python -m pytest tests/ -v
```

### 2. Run Phase 1 Deep Learning Verification Suite (10 Tests)
```powershell
python scripts/smoke_test.py
```

### 3. Verify MongoDB Atlas Status
```powershell
python -c "from src.db.mongodb import db_manager; print(db_manager.get_status())"
```

### 4. Build Production Frontend Bundle
```powershell
cd frontend
npm run build
```

---

## Retraining Models
To execute the complete sequential training pipeline on CUDA:
```powershell
python -u scripts/train_all_models.py
```
Checkpoints will automatically save to `checkpoints/` and refresh the model registry.

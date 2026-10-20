# Past2Pixel — Ancient Manuscript Damage Detection, Restoration, and Text Preservation System

**Past2Pixel** is a deep learning and computer vision framework designed to analyze, localize damage, restore degraded document folios, extract text, and provide context-aware text suggestions for ancient historical manuscripts (specifically Vedic Sanskrit and Devanagari heritage documents).

---

## Central Technical Distinctions
The system strictly distinguishes between four separate tasks:
1. **Damage Detection:** Identifying and localizing tears, stains, fading, and scratches into pixel-level binary masks and visual heatmaps.
2. **Image Restoration:** Enhancing contrast, removing noise, and restoring legibility using deep learning (U-Net vs. Autoencoder baseline).
3. **Optical Character Recognition (OCR):** Recognizing machine-readable text from visible characters.
4. **Damaged-Text Prediction:** Contextual linguistic suggestions for obscured/missing characters based on surrounding readable text and corpus statistics. **Predictions are explicitly marked as speculative and never presented as verified historical truth.**

---

## Four-Phase Roadmap
- **[x] PHASE 1 — AI Core, Datasets, Damage Detection, and Restoration Engine (Completed)**
- **[x] PHASE 2 — Backend, OCR, Database, Authentication, and Text Prediction (Completed)**
- **[ ] PHASE 3 — Frontend and Complete User Workflow**
- **[ ] PHASE 4 — Session-Aware Reporting, Integration, Testing, and Final Polish**

---

## Backend & API Usage (FastAPI)

### 1. Requirements Installation
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
*(Optional: add your MongoDB Atlas URI into `MONGODB_URI`. If left blank, the backend automatically uses durable local file persistence).*

### 3. Launch the Backend Server
```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Once started, explore the interactive API documentation:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Testing & Verification

### Run Phase 2 Backend Automated Suite (18 Tests)
```powershell
python -m pytest tests/ -v
```

### Run Phase 1 Deep Learning Verification Suite (10 Tests)
```powershell
python scripts/smoke_test.py
```

### Run Dataset Inventory Audit
```powershell
python -c "from src.data.dataset_inventory import print_dataset_inventory_summary; print_dataset_inventory_summary()"
```

---

## GPU Training (For High-Epoch Models)
If you want to train high-epoch models on a CUDA-enabled GPU:
1. Refer to [`docs/GPU_TRAINING_GUIDE.md`](file:///c:/Users/Jitesh/Desktop/DL%20Project/Past2Pixel/docs/GPU_TRAINING_GUIDE.md).
2. Run `run_gpu_training.bat` or `python scripts/train_restoration.py --device cuda`.
3. Checkpoint files will automatically save to `checkpoints/`.


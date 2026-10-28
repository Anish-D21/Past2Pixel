# Past2Pixel — Ancient Manuscript Restoration, Explainable AI & Text Preservation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://react.dev/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-12.8%20RTX%205060-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Past2Pixel** is a deep learning and computer vision framework for the digital preservation of historical manuscripts (specifically Vedic Sanskrit and Devanagari heritage folios). The system localizes physical document damage, restores background discolorations while locking authentic calligraphic ink strokes in place, extracts text using multi-pass OCR, and recovers damaged words through contextual language modeling.

---

## System Architecture & Pipeline

```text
[Input Degraded Folio] ──► [Device Detection & Image Normalization]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
[Stage 1: Damage U-Net]                            [Stage 2: Restoration U-Net]
 (Semantic Segmentation, BCE+Dice)                  (Skip Connections, Hybrid Loss)
            │                                                   │
            ▼                                                   ▼
  {Damage Mask & Heatmap}                               {Restored Canvas}
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                      [Stage 3: Stroke Preservation Engine]
                       - Adaptive Laplacian Edge Gating & Sauvola Filtering
                       - Authentic Ink Density Fusion (Zero Hallucination)
                                      │
                                      ▼
                        {Restored Clean Manuscript}
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 [Stage 4: Multi-Pass OCR]                             [Stage 6: Explainable AI]
  - 5-Pass Adaptive Tesseract                           - Grad-CAM Heatmaps
  - Geometric Line Reconciliation                       - Integrated Gradients
  - Languages: Sanskrit, Hindi, English                 - Damage Boundary Overlays
            │                                                   │
            ▼                                                   ▼
  {Extracted Unicode Text}                             [Interactive Web Dashboard]
            │
            ▼
[Stage 5: Linguistic Context N-Gram]
 - 3-Gram Devanagari Language Model
 - Perplexity-Ranked Token Completion
            │
            ▼
 {Final Digital Transcription & Downloadable PDF Laboratory Report}
```

---

## Verified Model Checkpoints & Performance

All models are trained with zero data leakage on the unified 2,031-sample corpus using an **NVIDIA GeForce RTX 5060 Laptop GPU** (CUDA 12.8):

| Model Name | Architecture | Trainable Parameters | Checkpoint Location | Held-Out Test Evaluation ($N = 283$) | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **U-Net Primary Restoration** | 4-Level Skip U-Net | **7,764,547** | `checkpoints/unet_restoration_best.pt` | **PSNR: 27.70 ± 5.65 dB**, **SSIM: 0.9427 ± 0.0354** | `READY_FOR_INFERENCE` |
| **Convolutional Autoencoder** | 3-Level Hourglass (Baseline) | **1,208,419** | `checkpoints/autoencoder_baseline_best.pt` | **PSNR: 25.36 ± 6.22 dB**, **SSIM: 0.9086 ± 0.0374** | `READY_FOR_INFERENCE` |
| **Damage Segmentation U-Net** | 4-Level Semantic U-Net | **7,761,985** | `checkpoints/damage_detector_best.pt` | **Mean IoU: 0.7097 ± 0.1388**, **Dice: 0.8199 ± 0.1293** | `READY_FOR_INFERENCE` |
| **Context Text Predictor** | 3-Gram Statistical LM | 3,405 Vocab | `checkpoints/context_predictor.json` | 4,039 sentences, 13,145 words | `ONLINE` |

*Note: On paired ground-truth lines (`page001_line001.png`), restoration achieves a **50.0% relative Character Error Rate (CER) reduction** ($37.21\% \to 18.60\%$). On severely degraded folios (`00000003.png`), the multi-pass OCR engine recovers **1,471 characters across 128 lines** where single-pass OCR produces 0 characters.*

---

## Quickstart: Running the Application

Running Past2Pixel requires two separate terminal windows (one for the backend API and one for the frontend client).

### Step 1: Start the FastAPI Backend

Open PowerShell in the project root directory:

```powershell
# 1. Navigate to project root
cd "C:\Users\Jitesh\Desktop\DL Project\Past2Pixel"

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Launch backend via Python module (prevents Windows Application Control blocks)
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Backend Health & Diagnostics:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI JSON Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### Step 2: Start the React + Vite Frontend

Open a second terminal window:

```powershell
# 1. Navigate to the frontend directory
cd "C:\Users\Jitesh\Desktop\DL Project\Past2Pixel\frontend"

# 2. Start the development server
npm run dev
```

Navigate your web browser to: **[http://localhost:5173](http://localhost:5173)**.

---

## Configuration & Environment Variables (`.env`)

Configuration is managed via `.env` in the project root:

| Key | Default Value | Description |
| :--- | :--- | :--- |
| `DEVICE` | `auto` | Execution hardware: `auto`, `cuda`, or `cpu`. |
| `TESSERACT_CMD` | `C:\Users\Jitesh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe` | Absolute path to Tesseract OCR binary. |
| `TESSDATA_PREFIX` | `C:\Users\Jitesh\AppData\Local\Programs\Tesseract-OCR\tessdata` | Path to language pack files (`san`, `hin`, `eng`). |
| `MONGODB_URI` | `mongodb+srv://...` | MongoDB Atlas URI (falls back to local JSON if empty). |
| `MONGODB_DB_NAME` | `past2pixel` | Target database collection name. |
| `JWT_SECRET_KEY` | *(Configured)* | Cryptographic key for user authentication tokens. |
| `MAX_UPLOAD_SIZE_MB` | `15` | Maximum manuscript upload file size. |

---

## Testing & Quality Assurance Suite

Past2Pixel includes automated testing gates:

```powershell
# 1. Run Complete Backend Unit & Integration Tests (34 Tests)
python -m pytest tests/ -v

# 2. Run End-to-End Deep Learning Smoke Test Suite (10 Gates)
python scripts/smoke_test.py

# 3. Verify Live REST API Endpoint Connectivity
python scripts/verify_live_api.py

# 4. Verify MongoDB Atlas Persistence & Diagnostics
python -c "from src.db.mongodb import db_manager; print(db_manager.get_status())"

# 5. Build Production Frontend Bundle (TypeScript + Vite)
cd frontend
npm run build
```

---

## Retraining Models & Benchmarking

To re-run training or re-evaluate checkpoints:

```powershell
# Run full sequential training on GPU (Autoencoder -> U-Net -> Damage Detector -> LM)
python -u scripts/train_all_models.py

# Run standalone held-out test evaluation on CUDA
python scripts/evaluate_models.py

# Run multi-pass OCR benchmarking
python scripts/evaluate_ocr_pipeline.py
```

---

## Project Team & Academic Credit

- **Institution:** St. Francis Institute of Technology (SFIT), University of Mumbai
- **Department:** Department of Computer Engineering (AIML)
- **Project Team (Batch 2):** Anish Desai, Rohan Satkar, Jitesh Zope, Bhavisha Bhanushali
- **Project Supervisor:** Dr. Joanne Gomes

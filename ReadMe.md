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
- **[ ] PHASE 2 — Backend, OCR, Database, Authentication, and Text Prediction**
- **[ ] PHASE 3 — Frontend and Complete User Workflow**
- **[ ] PHASE 4 — Session-Aware Reporting, Integration, Testing, and Final Polish**

---

## Phase 1 Architecture & Implemented Modules
```
src/
├── core/
│   ├── config.py                 # Centralized configuration & settings
│   └── device.py                 # Compute hardware detection & timer
├── data/
│   ├── synthetic_degradation.py  # Degradation pipeline & damage mask generator
│   ├── data_split.py             # Zero-leakage document ID splitter
│   ├── dataset_inventory.py      # Dataset A & B auditor
│   ├── vedo_dataset.py           # Dataset A loader & synthetic sample generator
│   └── sanskrit_hindi_dataset.py # Dataset B loader & vocabulary extractor
├── models/
│   ├── autoencoder.py            # Convolutional Autoencoder baseline
│   ├── unet_restoration.py       # Primary U-Net restoration network
│   ├── unet_damage_detector.py   # U-Net damage segmentation & overlay generator
│   └── model_registry.py         # Model lifecycle & checkpoint manager
├── metrics/
│   ├── restoration_metrics.py    # PSNR & SSIM on valid reference pairs
│   └── damage_metrics.py         # IoU, Dice, Precision, and Recall
├── prediction/
│   └── context_predictor.py      # N-gram LM, CER, and WER metrics
└── pipeline/
    └── restoration_engine.py     # Integrated Phase 1 execution engine
```

---

## Quickstart & Verification (Windows CPU)

### 1. Requirements Installation
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run Complete CPU Verification Suite
```powershell
python scripts/smoke_test.py
```

### 3. Check Dataset Inventory
```powershell
python -c "from src.data.dataset_inventory import print_dataset_inventory_summary; print_dataset_inventory_summary()"
```

### 4. GPU Training (For Friend's Machine)
If you want to train high-epoch models on your friend's GPU machine:
1. Refer to [`docs/GPU_TRAINING_GUIDE.md`](file:///d:/Desktop/Past2Pixel/docs/GPU_TRAINING_GUIDE.md).
2. Run `run_gpu_training.bat` or `python scripts/train_restoration.py --device cuda`.
3. Copy the resulting `.pt` checkpoints back to `checkpoints/` on your PC.

# Past2Pixel — System Architecture & Design Specification

## 1. Architectural Overview
**Past2Pixel** is an AI-powered digital preservation framework designed for ancient and historical manuscripts (specifically Sanskrit and Devanagari documents). The system decomposes document preservation into four mathematically distinct stages:

```
                                  [ Input Manuscript Image ]
                                              │
                                              ▼
                             [ Validation & Device Detection ]
                                              │
                                              ▼
                             [ Preprocessing & Standardization ]
                                  (Resize to 256x256, Normalize)
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
            [ Damage Localization Model ]             [ Primary Restoration Model ]
             (UNet Semantic Segmentation)                 (UNet with Skip Conns)
                         │                                         │
                         ▼                                         ▼
             Pixel-Level Damage Mask                      Restored Clean Image
             & Visual Crimson Overlay                 (vs ConvAutoencoder Baseline)
                         │                                         │
                         └────────────────────┬────────────────────┘
                                              ▼
                                 [ Optical Character Recognition ]
                                     (Tesseract Devanagari)
                                              │
                                              ▼
                                [ Damaged-Text Prediction Engine ]
                                  (Context-Aware N-Gram Model)
                                              │
                                              ▼
                                  [ Scientific Evaluation ]
                             - PSNR / SSIM (Reference Pairs)
                             - IoU / Dice (Damage Masks)
                             - CER / WER (Text Ground Truth)
                                              │
                                              ▼
                                [ Image-Specific Session Report ]
```

---

## 2. Directory Layout & Modular Structure
```
Past2Pixel/
├── checkpoints/              # Checkpoint storage (.pt weights)
├── data/
│   ├── raw/
│   │   ├── vedo_ocr/         # Dataset A: VedoOCR
│   │   └── sanskrit_hindi_vqa/# Dataset B: Sanskrit-Hindi OCR VQA
│   ├── synthetic/            # Synthetic degraded/clean pairs & masks
│   ├── splits/               # Split manifests (train/val/test)
│   └── samples/              # Synthetic manuscript samples for testing
├── docs/                     # Comprehensive documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET_DOCUMENTATION.md
│   ├── MODEL_DOCUMENTATION.md
│   ├── GPU_TRAINING_GUIDE.md
│   └── TESTING.md
├── scripts/                  # Standalone CLI tools & test runners
│   ├── train_restoration.py  # U-Net & Autoencoder restoration trainer
│   ├── train_damage_detector.py # U-Net damage segmentation trainer
│   └── smoke_test.py         # End-to-end CPU verification suite
├── src/
│   ├── core/
│   │   ├── config.py         # Centralized configuration & environment vars
│   │   └── device.py         # Hardware detection, timing, and memory logging
│   ├── data/
│   │   ├── synthetic_degradation.py # Controlled degradation & exact mask engine
│   │   ├── data_split.py     # Leakage-free document ID splitter
│   │   ├── dataset_inventory.py     # Dataset cataloging & auditing
│   │   ├── vedo_dataset.py   # Dataset A loader & sample generator
│   │   └── sanskrit_hindi_dataset.py # Dataset B loader & vocabulary extractor
│   ├── models/
│   │   ├── autoencoder.py    # Baseline Convolutional Autoencoder
│   │   ├── unet_restoration.py # Primary U-Net restoration network
│   │   ├── unet_damage_detector.py # U-Net damage segmentation network
│   │   └── model_registry.py # Model lifecycle tracking & checkpoint loader
│   ├── metrics/
│   │   ├── restoration_metrics.py # PSNR and SSIM computation
│   │   └── damage_metrics.py      # IoU, Dice, Precision, and Recall
│   ├── prediction/
│   │   └── context_predictor.py   # N-gram LM, CER, and WER metrics
│   └── pipeline/
│       └── restoration_engine.py  # Integrated execution engine & report generator
├── requirements.txt
├── run_gpu_training.bat      # One-click GPU training script
└── ReadMe.md
```

---

## 3. Mathematical Formulations & Losses

### 3.1 Primary U-Net Restoration Loss
Combines $L_1$ loss (preserving text sharpness and high-frequency stroke boundaries) with Mean Squared Error (MSE) loss (smoothing parchment background):
$$\mathcal{L}_{\text{restoration}} = \alpha \cdot \mathcal{L}_1(\hat{Y}, Y) + (1 - \alpha) \cdot \mathcal{L}_{\text{MSE}}(\hat{Y}, Y)$$
where $\alpha = 0.8$, $\hat{Y} \in [0, 1]^{3 \times H \times W}$ is the restored tensor, and $Y$ is the clean ground-truth target.

### 3.2 Damage Segmentation Loss
Combines Binary Cross-Entropy with Dice Loss to counter severe class imbalance between damaged pixels and undamaged manuscript parchment:
$$\mathcal{L}_{\text{damage}} = \beta \cdot \mathcal{L}_{\text{BCE}}(P, M) + (1 - \beta) \cdot \left(1 - \frac{2 \sum P_i M_i + \epsilon}{\sum P_i + \sum M_i + \epsilon}\right)$$
where $\beta = 0.5$, $P$ is the predicted probability map, and $M \in \{0, 1\}^{H \times W}$ is the ground-truth damage mask.

### 3.3 Objective Evaluation Metrics
- **Peak Signal-to-Noise Ratio (PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$
- **Structural Similarity Index (SSIM):**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- **Intersection over Union (IoU):**
  $$\text{IoU} = \frac{|P \cap M|}{|P \cup M|}$$
- **Character Error Rate (CER):**
  $$\text{CER} = \frac{S + D + I}{N}$$

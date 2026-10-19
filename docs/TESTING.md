# Past2Pixel — Testing & Verification Report (Phase 1)

This document records the automated testing suite, commands, and actual outcomes observed during Phase 1 development.

## 1. Running the Verification Suite
To execute the complete end-to-end CPU verification suite on Windows:
```powershell
python scripts/smoke_test.py
```

---

## 2. Test Execution Results (Observed)

```
======================================================================
TEST COMPONENT                                | STATUS | DETAILS
----------------------------------------------------------------------
1. Compute Device & Hardware Detection        | PASS   | CPU: 12 logical cores
2. Dataset Inventory & Path Verification      | PASS   | Cataloged Dataset A & B
3. Synthetic Degradation & Damage Mask        | PASS   | Applied: ['missing_regions', 'stains', 'fading_uneven_light', 'surface_noise', 'optical_blur']
4. Zero-Leakage Dataset Splitting             | PASS   | Strict document ID isolation verified
5. Model Architectures & Forward Tensors      | PASS   | AE, UNet-Restore, and UNet-Detector tensors verified
6. Model Checkpoint Save/Load Lifecycle       | PASS   | Checkpoint serialization and reload verified
7. Scientific Evaluation Metrics              | PASS   | PSNR: 37.0dB, SSIM: 0.999, IoU: 1.0
8. Context-Aware Text Prediction & CER/WER    | PASS   | Vocabulary size: 78 words
9. End-to-End RestorationEngine Execution     | PASS   | Runtime: 1128.51ms
======================================================================
>>> ALL PHASE 1 SYSTEM TESTS PASSED SUCCESSFULLY! <<<
```

---

## 3. Individual Component Tests Covered

1. **Hardware Detection:** Queries Windows platform, logical CPU core count (12 cores detected), RAM, and PyTorch CUDA availability.
2. **Dataset Auditing:** Verifies presence of Dataset A (`data/raw/vedo_ocr`) and Dataset B (`data/raw/sanskrit_hindi_vqa`), image extensions, and JSON annotation counts.
3. **Synthetic Degradation:** Confirms generation of degraded manuscript image alongside an aligned binary damage mask ($M \in [0, 1]$).
4. **Leak-Free Splitting:** Validates 70/15/15 train/val/test split across unique document IDs, ensuring disjoint sets.
5. **Model Forward Passes:** Verifies shapes of output tensors from `ConvAutoencoder` `(2, 3, 256, 256)`, `UNetRestoration` `(2, 3, 256, 256)`, and `UNetDamageDetector` `(2, 1, 256, 256)`.
6. **Checkpoint Lifecycle:** Verifies state dict serialization, metadata preservation, and deserialization using PyTorch 2.6+ standards.
7. **Scientific Metrics:** Validates PSNR and SSIM on identical images ($SSIM = 1.0$, $PSNR = \infty$) and degraded targets; verifies IoU/Dice calculation and honest handling of `UNAVAILABLE` when clean targets are missing.
8. **Text Prediction & Edit Distance:** Fits n-gram language model on Devanagari corpus; tests candidate completion and Levenshtein CER/WER calculation.
9. **Integrated Pipeline:** Verifies end-to-end `RestorationEngine.process_image()` execution, session ID generation, timing metrics, and artifact saving.

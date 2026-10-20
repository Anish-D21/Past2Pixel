# Past2Pixel — Automated Testing & Verification Suite

This document records the automated testing suite, execution commands, and verified test results for Past2Pixel.

---

## 1. Running the Test Suites

### Suite A: Phase 2 Backend & Integration Suite (Pytest)
Executes all 18 automated unit and integration tests covering authentication, upload security, OCR, database isolation, pipeline execution, and text prediction:
```powershell
python -m pytest tests/ -v
```

### Suite B: Phase 1 Deep Learning Core Verification Suite
Executes the complete 10-point end-to-end CPU smoke test covering compute, datasets, synthetic degradation, splitting, forward passes, serialization, metrics, and real dataset decoders:
```powershell
python scripts/smoke_test.py
```

---

## 2. Phase 2 Pytest Execution Results (18 Passed)

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Jitesh\Desktop\DL Project\Past2Pixel
collected 18 items

tests/test_auth.py::test_password_hashing PASSED                         [  5%]
tests/test_auth.py::test_jwt_token_encode_decode PASSED                  [ 11%]
tests/test_auth.py::test_user_registration_and_login PASSED              [ 16%]
tests/test_ocr.py::test_ocr_status_structure PASSED                      [ 22%]
tests/test_ocr.py::test_ocr_session_not_found PASSED                     [ 27%]
tests/test_ocr.py::test_ocr_cer_wer_evaluation PASSED                    [ 33%]
tests/test_pipeline_and_models.py::test_model_readiness_untrained_status PASSED [ 38%]
tests/test_pipeline_and_models.py::test_restoration_execution_and_artifacts PASSED [ 44%]
tests/test_pipeline_and_models.py::test_restoration_missing_session PASSED [ 50%]
tests/test_prediction.py::test_prediction_output_and_labeling PASSED     [ 55%]
tests/test_prediction.py::test_prediction_empty_context_rejected PASSED  [ 61%]
tests/test_sessions_and_db.py::test_database_status PASSED               [ 66%]
tests/test_sessions_and_db.py::test_session_ownership_and_isolation PASSED [ 72%]
tests/test_upload_validation.py::test_valid_image_upload PASSED          [ 77%]
tests/test_upload_validation.py::test_unsupported_file_extension_rejected PASSED [ 83%]
tests/test_upload_validation.py::test_empty_file_upload_rejected PASSED  [ 88%]
tests/test_upload_validation.py::test_corrupt_image_bytes_rejected PASSED [ 94%]
tests/test_upload_validation.py::test_oversized_file_rejected PASSED     [100%]

======================= 18 passed, 2 warnings in 7.65s ========================
```

---

## 3. Phase 1 Verification Suite Results (10 Passed)

```
======================================================================
Past2Pixel — Phase 1 End-to-End CPU Verification Suite
======================================================================

======================================================================
TEST COMPONENT                                | STATUS | DETAILS
----------------------------------------------------------------------
1. Compute Device & Hardware Detection        | PASS   | CPU: 24 logical cores
2. Dataset Inventory & Path Verification      | PASS   | Cataloged Dataset A & B
3. Synthetic Degradation & Damage Mask        | PASS   | Applied: ['missing_regions', 'stains', 'fading_uneven_light', 'surface_noise', 'optical_blur']
4. Zero-Leakage Dataset Splitting             | PASS   | Strict document ID isolation verified
5. Model Architectures & Forward Tensors      | PASS   | AE, UNet-Restore, and UNet-Detector tensors verified
6. Model Checkpoint Save/Load Lifecycle       | PASS   | Checkpoint serialization and reload verified
7. Scientific Evaluation Metrics              | PASS   | PSNR: 37.0dB, SSIM: 0.999, IoU: 1.0
8. Context-Aware Text Prediction & CER/WER    | PASS   | Vocabulary size: 3976 words
9. End-to-End RestorationEngine Execution     | PASS   | Runtime: 136.10ms
10. Real Dataset Loaders & Image Decoding     | PASS   | VedoOCR: 83 test lines, VQA-OCR: 5 verified samples
======================================================================

>>> ALL PHASE 1 SYSTEM TESTS PASSED SUCCESSFULLY! <<<
```

---

## 4. Test Coverage Matrix

1. **Authentication & Token Lifecycle:** Validates bcrypt salt hashing, password verification, signed JWT token issue/decode, duplicate registration rejection, bad password 401 handling, and `/api/auth/me` protected endpoint access.
2. **Upload Security & Validation:** Verifies allowed extensions (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`), rejects unsupported extensions (`.txt`, `.exe`) with 400, rejects 0-byte uploads with 400, detects corrupted headers with 400, and rejects files exceeding `MAX_UPLOAD_SIZE_MB` with 413.
3. **OCR Engine & Availability:** Tests diagnostic inspection, honest `UNAVAILABLE` reporting when Tesseract binary is missing with step-by-step installation instructions, and Levenshtein CER/WER calculation against ground truth.
4. **Pipeline & Model Integrity:** Ensures untrained models are explicitly reported as `UNTRAINED_RANDOM_WEIGHTS` (never falsely claimed as trained), verifies image restoration and crimson overlay generation, and checks 404 handling on missing sessions.
5. **Context Text Prediction:** Tests n-gram language model prediction, enforces `is_speculative=True`, verifies the presence of the mandatory uncertainty label and historical disclaimer, and asserts 400 on empty context spans.
6. **Persistence & Session Ownership:** Tests MongoDB Atlas status reporting, transparent local file-backed persistence fallback, and strict user isolation (User B cannot view or delete User A's session -> 403 Forbidden).

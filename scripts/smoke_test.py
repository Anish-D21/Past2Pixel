"""
End-to-End CPU Smoke Test and Validation Suite for Past2Pixel Phase 1.
Verifies all models, data pipelines, metrics, and inference pipelines on local hardware.
"""
import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.device import get_device, get_device_info, log_system_info
from src.data.dataset_inventory import generate_dataset_inventory
from src.data.synthetic_degradation import SyntheticDegrader
from src.data.vedo_dataset import generate_synthetic_manuscript_sample, ManuscriptPairDataset
from src.data.sanskrit_hindi_dataset import SanskritHindiVQADataset
from src.data.data_split import split_document_ids
from src.models.autoencoder import ConvAutoencoder
from src.models.unet_restoration import UNetRestoration
from src.models.unet_damage_detector import UNetDamageDetector, generate_damage_overlay
from src.models.model_registry import model_registry, save_checkpoint, load_model_for_inference
from src.metrics.restoration_metrics import calculate_psnr, calculate_ssim, evaluate_restoration_pair
from src.metrics.damage_metrics import calculate_segmentation_metrics
from src.prediction.context_predictor import ContextTextPredictor, calculate_cer, calculate_wer
from src.pipeline.restoration_engine import RestorationEngine


def run_smoke_test():
    print("=" * 70)
    print("Past2Pixel — Phase 1 End-to-End CPU Verification Suite")
    print("=" * 70)
    test_results = []
    settings.ensure_directories()
    device = get_device("cpu")  # Force CPU for smoke testing

    # Test 1: Hardware & Device Detection
    try:
        dev_info = get_device_info(device)
        assert dev_info["device_type"] == "cpu"
        assert dev_info["python_version"].startswith("3.11")
        test_results.append(("1. Compute Device & Hardware Detection", "PASS", f"CPU: {dev_info.get('cpu_count_logical')} logical cores"))
    except Exception as e:
        test_results.append(("1. Compute Device & Hardware Detection", "FAIL", str(e)))

    # Test 2: Dataset Inventory & Paths
    try:
        inv = generate_dataset_inventory()
        assert "dataset_a_vedo_ocr" in inv
        assert "dataset_b_sanskrit_hindi_vqa" in inv
        test_results.append(("2. Dataset Inventory & Path Verification", "PASS", "Cataloged Dataset A & B"))
    except Exception as e:
        test_results.append(("2. Dataset Inventory & Path Verification", "FAIL", str(e)))

    # Test 3: Synthetic Degradation & Damage Mask Engine
    try:
        degrader = SyntheticDegrader(seed=42)
        test_img = Image.new("RGB", (256, 256), color=(240, 225, 200))
        deg_img, mask, meta = degrader.apply_pipeline(test_img)
        assert deg_img.size == (256, 256)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() <= mask.max() <= 1.0
        test_results.append(("3. Synthetic Degradation & Damage Mask", "PASS", f"Applied: {meta['applied_degradations']}"))
    except Exception as e:
        test_results.append(("3. Synthetic Degradation & Damage Mask", "FAIL", str(e)))

    # Test 4: Leak-Free Data Splitting
    try:
        sample_doc_ids = [f"doc_{i:03d}" for i in range(20)]
        splits = split_document_ids(sample_doc_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        assert len(splits["train"]) == 14
        assert len(splits["val"]) == 3
        assert len(splits["test"]) == 3
        # Strict zero-leakage check
        assert set(splits["train"]).isdisjoint(set(splits["test"]))
        assert set(splits["train"]).isdisjoint(set(splits["val"]))
        test_results.append(("4. Zero-Leakage Dataset Splitting", "PASS", "Strict document ID isolation verified"))
    except Exception as e:
        test_results.append(("4. Zero-Leakage Dataset Splitting", "FAIL", str(e)))

    # Test 5: Model Architectures & Forward Passes
    try:
        dummy_input = torch.randn(2, 3, 256, 256)

        # Baseline Autoencoder
        ae = ConvAutoencoder(in_channels=3)
        ae_out = ae(dummy_input)
        assert ae_out.shape == (2, 3, 256, 256)
        assert 0.0 <= ae_out.min() <= ae_out.max() <= 1.0

        # U-Net Restoration
        unet = UNetRestoration(in_channels=3, out_channels=3)
        unet_out = unet(dummy_input)
        assert unet_out.shape == (2, 3, 256, 256)
        assert 0.0 <= unet_out.min() <= unet_out.max() <= 1.0

        # U-Net Damage Detector
        detector = UNetDamageDetector(in_channels=3)
        det_out = detector(dummy_input)
        assert det_out.shape == (2, 1, 256, 256)
        assert 0.0 <= det_out.min() <= det_out.max() <= 1.0

        test_results.append(("5. Model Architectures & Forward Tensors", "PASS", "AE, UNet-Restore, and UNet-Detector tensors verified"))
    except Exception as e:
        test_results.append(("5. Model Architectures & Forward Tensors", "FAIL", str(e)))

    # Test 6: Checkpoint Save & Reload Lifecycle
    try:
        test_ckpt_path = settings.CHECKPOINT_DIR / "test_smoke_ckpt.pt"
        save_checkpoint(
            model=ae,
            optimizer=None,
            epoch=1,
            val_loss=0.045,
            metrics={"val_psnr": 28.5},
            device_info=dev_info,
            output_path=test_ckpt_path,
        )
        assert test_ckpt_path.exists()
        reloaded_ae, meta = load_model_for_inference(ConvAutoencoder(in_channels=3), test_ckpt_path, device)
        assert meta["epoch"] == 1
        assert meta["metrics"]["val_psnr"] == 28.5
        test_ckpt_path.unlink()  # Clean up test checkpoint
        test_results.append(("6. Model Checkpoint Save/Load Lifecycle", "PASS", "Checkpoint serialization and reload verified"))
    except Exception as e:
        test_results.append(("6. Model Checkpoint Save/Load Lifecycle", "FAIL", str(e)))

    # Test 7: Metrics Engine (PSNR, SSIM, IoU, Dice)
    try:
        sample_img_path = settings.SAMPLES_DIR / "metric_test_manuscript.png"
        generate_synthetic_manuscript_sample(sample_img_path, width=256, height=256, seed=123)
        with Image.open(sample_img_path) as im:
            clean_arr = np.array(im, dtype=np.float32) / 255.0
        
        # Test identical images
        assert abs(calculate_ssim(clean_arr, clean_arr) - 1.0) < 1e-4
        assert calculate_psnr(clean_arr, clean_arr) == float("inf")

        # Test slightly degraded image
        noisy_arr = np.clip(clean_arr * 0.96 + 0.02, 0.0, 1.0).astype(np.float32)
        psnr_val = calculate_psnr(noisy_arr, clean_arr)
        ssim_val = calculate_ssim(noisy_arr, clean_arr)
        assert psnr_val > 25.0
        assert ssim_val > 0.85

        # Unavailable handling
        unavail_res = evaluate_restoration_pair(noisy_arr, None)
        assert unavail_res["status"] == "UNAVAILABLE"

        # Damage metrics
        true_m = np.zeros((256, 256), dtype=np.float32)
        true_m[50:100, 50:100] = 1.0
        pred_m = true_m.copy()
        seg_res = calculate_segmentation_metrics(pred_m, true_m)
        assert seg_res["iou"] == 1.0
        assert seg_res["dice"] == 1.0

        test_results.append(("7. Scientific Evaluation Metrics", "PASS", f"PSNR: {psnr_val:.1f}dB, SSIM: {ssim_val:.3f}, IoU: 1.0"))
    except Exception as e:
        test_results.append(("7. Scientific Evaluation Metrics", "FAIL", repr(e)))

    # Test 8: Context-Aware Text Predictor, CER & WER
    try:
        vqa_dataset = SanskritHindiVQADataset()
        corpus = vqa_dataset.get_text_corpus()
        predictor = ContextTextPredictor(n=2)
        train_res = predictor.train_on_corpus(corpus)
        assert train_res["total_words_processed"] > 20

        # Run contextual prediction
        pred_res = predictor.predict_missing_span("धर्मक्षेत्रे", "समवेता")
        assert pred_res["status"] in ["PREDICTED", "LOW_CONFIDENCE", "INSUFFICIENT_CONTEXT"]
        assert pred_res["is_speculative"] is True

        # CER & WER
        cer = calculate_cer("कुरुक्षेत्रे", "कुरुक्षेत्रे")
        wer = calculate_wer("धर्मक्षेत्रे कुरुक्षेत्रे", "धर्मक्षेत्रे कुरुक्षेत्रे")
        assert cer == 0.0
        assert wer == 0.0

        test_results.append(("8. Context-Aware Text Prediction & CER/WER", "PASS", f"Vocabulary size: {train_res['unique_vocabulary_size']} words"))
    except Exception as e:
        test_results.append(("8. Context-Aware Text Prediction & CER/WER", "FAIL", str(e)))

    # Test 9: Integrated Restoration Engine Execution
    try:
        sample_img_path = settings.SAMPLES_DIR / "smoke_test_manuscript.png"
        generate_synthetic_manuscript_sample(sample_img_path, width=256, height=256, seed=777)

        engine = RestorationEngine(device_preference="cpu")
        report = engine.process_image(
            input_image=sample_img_path,
            reference_clean_image=sample_img_path,
        )

        assert report["session_id"].startswith("p2p_")
        assert "timings_ms" in report
        assert "damage_detection_result" in report
        assert "restoration_results" in report
        assert report["images"]["restored"].size == (256, 256)

        test_results.append(("9. End-to-End RestorationEngine Execution", "PASS", f"Runtime: {report['timings_ms']['total_pipeline']}ms"))
    except Exception as e:
        test_results.append(("9. End-to-End RestorationEngine Execution", "FAIL", str(e)))

    # Test 10: Real Dataset Loaders & Image Decoding
    try:
        from src.data.vedo_dataset import load_vedo_records, VedoOCRDataset
        from src.data.sanskrit_hindi_dataset import SanskritHindiOCRDataset

        # Verify VedoOCR real records
        vedo_records = load_vedo_records("test")
        assert len(vedo_records) > 0, "No VedoOCR test records found"
        vedo_ds = VedoOCRDataset("test")
        v_sample = vedo_ds[0]
        assert v_sample["image"].size[0] > 0 and v_sample["image"].size[1] > 0
        assert len(v_sample["transcription"]) > 0

        # Verify real Vedo image with ManuscriptPairDataset (synthetic degradation on real manuscript)
        real_crops = [r["image_path"] for r in vedo_records[:3]]
        pair_ds = ManuscriptPairDataset(real_crops, target_size=(256, 256), seed=42)
        pair_item = pair_ds[0]
        assert pair_item["degraded"].shape == (3, 256, 256)
        assert pair_item["clean"].shape == (3, 256, 256)
        assert pair_item["damage_mask"].shape == (1, 256, 256)

        # Verify Sanskrit-Hindi OCR real dataset
        vqa_ocr_ds = SanskritHindiOCRDataset("test", max_count=5)
        assert len(vqa_ocr_ds) > 0, "No Sanskrit-Hindi OCR test samples found"
        s_sample = vqa_ocr_ds[0]
        assert s_sample["image"].size[0] > 0
        assert len(s_sample["text"]) > 0

        test_results.append((
            "10. Real Dataset Loaders & Image Decoding",
            "PASS",
            f"VedoOCR: {len(vedo_records)} test lines, VQA-OCR: {len(vqa_ocr_ds.samples)} verified samples",
        ))
    except Exception as e:
        test_results.append(("10. Real Dataset Loaders & Image Decoding", "FAIL", str(e)))

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'TEST COMPONENT':<45} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 70)
    all_passed = True
    for name, status, details in test_results:
        print(f"{name:<45} | {status:<6} | {details}")
        if status != "PASS":
            all_passed = False
    print("=" * 70)

    if all_passed:
        print("\n>>> ALL PHASE 1 SYSTEM TESTS PASSED SUCCESSFULLY! <<<\n")
        return True
    else:
        print("\n>>> SOME TESTS FAILED. CHECK DETAILS ABOVE. <<<\n")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

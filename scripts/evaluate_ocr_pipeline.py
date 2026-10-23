"""
scripts/evaluate_ocr_pipeline.py
Academic evaluation of Tesseract OCR with Sanskrit, Hindi, and English configurations.
Evaluates CER, WER, and context-aware completion on authentic manuscript samples.
"""
import os
import sys
import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.ocr.ocr_engine import ocr_engine
from src.ocr.ocr_preprocessor import OCRPreprocessor
from src.prediction.context_predictor import ContextTextPredictor
from src.api.routes.predict import get_trained_predictor


def main():
    print("=" * 70)
    print("Past2Pixel — Authentic OCR & Text Completion Evaluation")
    print("=" * 70)

    # 1. Inspect Engine Status
    status = ocr_engine.get_status(force_refresh=True)
    print(f"Tesseract Version: {status.get('version')}")
    print(f"Tesseract Binary:  {status.get('tesseract_cmd')}")
    print(f"Tessdata Path:     {status.get('tessdata_dir')}")
    print(f"Installed Packs:   {status.get('available_languages')}")
    print(f"Supported Configs: {status.get('supported_configurations')}")
    print("-" * 70)

    # 2. Sanskrit Evaluation on Authentic VedoOCR Line Crops
    print("\n--- 1. Authentic Vedic Sanskrit OCR Evaluation (VedoOCR) ---")
    kaggle_dir = Path("data/raw/VedOCR/VedOCR_kaggle_release")
    tsv_path = kaggle_dir / "master.tsv"

    san_eval_results = []
    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        tested = 0
        for r in rows:
            crop_rel = r.get("crop_path") or ""
            img_path = kaggle_dir / crop_rel
            gt = r.get("transcription", "").strip()

            if not img_path.exists() or not gt:
                continue

            try:
                img = Image.open(img_path).convert("RGB")

                # Test A: Raw OCR
                ocr_raw = ocr_engine.extract_text(img, lang="san", preprocessing="none")
                eval_raw = ocr_engine.evaluate_ocr(ocr_raw["extracted_text"], gt)

                # Test B: Preprocessed OCR (auto/contrast/illumination)
                ocr_prep = ocr_engine.extract_text(img, lang="san", preprocessing="auto")
                eval_prep = ocr_engine.evaluate_ocr(ocr_prep["extracted_text"], gt)

                san_eval_results.append({
                    "file": img_path.name,
                    "gt": gt,
                    "raw_text": ocr_raw["extracted_text"].strip(),
                    "raw_cer": eval_raw["character_error_rate"],
                    "raw_wer": eval_raw["word_error_rate"],
                    "prep_text": ocr_prep["extracted_text"].strip(),
                    "prep_cer": eval_prep["character_error_rate"],
                    "prep_wer": eval_prep["word_error_rate"],
                    "prep_applied": ocr_prep.get("preprocessing_applied"),
                })
                tested += 1
                if tested >= 5:
                    break
            except Exception as e:
                print(f"Error testing {img_path.name}: {e}")

        for res in san_eval_results:
            print(f"\nSample: {res['file']}")
            print(f"  Ground Truth:    {res['gt']}")
            print(f"  Raw OCR:         {res['raw_text'] or '(None)'} | CER: {res['raw_cer']:.3f}, WER: {res['raw_wer']:.3f}")
            print(f"  Prep ({res['prep_applied']}): {res['prep_text'] or '(None)'} | CER: {res['prep_cer']:.3f}, WER: {res['prep_wer']:.3f}")
    else:
        print("VedoOCR master.tsv not found.")

    # 3. Hindi Evaluation on Authentic VQA OCR Images
    print("\n--- 2. Hindi Devanagari OCR Evaluation (Authentic Dataset) ---")
    vqa_test_ocr = Path("data/raw/Sanskrit & Hindi OCR + VQA/Dataset/test/ocr")
    hin_eval_results = []
    if vqa_test_ocr.exists():
        tested_hin = 0
        for jpg_file in sorted(vqa_test_ocr.glob("*.jpg")):
            txt_file = jpg_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
            with open(txt_file, "r", encoding="utf-8") as tf:
                gt_hin = tf.read().strip()
            if not gt_hin:
                continue

            try:
                img_hin = Image.open(jpg_file).convert("RGB")
                ocr_hin = ocr_engine.extract_text(img_hin, lang="hin", preprocessing="auto")
                eval_hin = ocr_engine.evaluate_ocr(ocr_hin["extracted_text"], gt_hin)
                hin_eval_results.append({
                    "file": jpg_file.name,
                    "gt": gt_hin,
                    "pred": ocr_hin["extracted_text"].strip(),
                    "cer": eval_hin["character_error_rate"],
                    "wer": eval_hin["word_error_rate"],
                    "prep": ocr_hin.get("preprocessing_applied"),
                })
                tested_hin += 1
                if tested_hin >= 5:
                    break
            except Exception as e:
                print(f"Error testing Hindi {jpg_file.name}: {e}")

        for res in hin_eval_results:
            print(f"\nHindi Sample: {res['file']}")
            print(f"  Ground Truth: {res['gt']}")
            print(f"  OCR Output:   {res['pred'] or '(None)'} | Prep: {res['prep']} | CER: {res['cer']:.3f}, WER: {res['wer']:.3f}")
    else:
        print("VQA OCR test directory not found.")

    # 4. English Evaluation
    print("\n--- 3. English Transliteration / Metadata OCR Evaluation ---")
    eng_samples = [
        "Past2Pixel: Deep Learning Manuscript Restoration and Explainable AI",
        "Vedic Sanskrit Manuscript Folio Rigveda Hymn",
    ]
    for idx, text in enumerate(eng_samples):
        img = Image.new("RGB", (750, 70), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), text, fill=(0, 0, 0))

        ocr_res = ocr_engine.extract_text(img, lang="eng", preprocessing="auto")
        eval_metrics = ocr_engine.evaluate_ocr(ocr_res["extracted_text"], text)
        print(f"English Sample {idx+1}:")
        print(f"  GT:   {text}")
        print(f"  Pred: {ocr_res['extracted_text'].strip()} (CER: {eval_metrics['character_error_rate']:.3f}, WER: {eval_metrics['word_error_rate']:.3f})")

    # 5. Context-Aware Damaged Text Completion Trace
    print("\n--- 4. Traced Context-Aware Damaged Text Completion ---")
    predictor = get_trained_predictor()
    print(f"Trained Predictor Vocab Size: {len(predictor.vocabulary)} words")
    print(f"N-Gram LM Order: {predictor.n}")

    # Trace realistic OCR text with an uncertain / damaged span
    working_ocr_text = "धर्मक्षेत्रे कुरु[?]ेत्रे समवेता युयुत्सवः मामकाः"
    uncertain_token = "कुरु[?]ेत्रे"
    print(f"Active Extracted Text: '{working_ocr_text}'")
    print(f"Selected Uncertain Span: '{uncertain_token}'")

    # Context extraction
    left_ctx, target, right_ctx = ContextTextPredictor.extract_context_from_text(
        full_text=working_ocr_text,
        target_span=uncertain_token,
    )
    print(f"Extracted Left Context:  '{left_ctx}'")
    print(f"Extracted Right Context: '{right_ctx}'")

    # Predict candidates
    pred_res = predictor.predict_missing_span(
        left_context=left_ctx,
        right_context=right_ctx,
        uncertain_span=uncertain_token,
        max_candidates=4,
    )
    print(f"Prediction Status: {pred_res['status']}")
    print("Top Candidates:")
    for c in pred_res.get("candidates", []):
        print(f"  - {c.get('text')}: score={c.get('context_score'):.4f} ({c.get('model_used')})")

    # Trace User Acceptance
    top_cand = pred_res.get("prediction")
    if top_cand:
        accepted_text = working_ocr_text.replace(uncertain_token, top_cand)
        print(f"\nUser Accepts Suggestion '{top_cand}':")
        print(f"  Working Text Updated To: '{accepted_text}'")
        print(f"  Raw OCR Remains Intact:  '{working_ocr_text}'")

    print("\n" + "=" * 70)
    print("OCR & Context-Aware Completion Evaluation Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

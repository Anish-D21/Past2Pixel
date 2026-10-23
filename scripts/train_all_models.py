"""
Sequential Master GPU Training Script for Past2Pixel.
Executes training in strict dependency order on CUDA:
Stage B: Convolutional Autoencoder Baseline
Stage C: U-Net Primary Restoration Network
Stage D: U-Net Damage Localization & Segmentation Network
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.device import log_system_info
from scripts.train_restoration import train_restoration
from scripts.train_damage_detector import train_damage_detector
from src.models.model_registry import model_registry


from src.core.config import settings
from src.prediction.context_predictor import ContextTextPredictor
from src.data.combined_dataset import build_unified_dataset_splits


def train_context_predictor_stage(splits):
    print("=" * 70, flush=True)
    print(">>> STAGE A: Training Context-Aware Damaged Text Predictor on Devanagari Corpus", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    sentences = []
    # 1. From VedoOCR train transcriptions
    for s in splits.get("train", []):
        t = s.get("transcription", "").strip()
        if t:
            sentences.append(t)

    # 2. From Sanskrit-Hindi train/ocr/*.txt
    sh_train_ocr = settings.SANSKRIT_HINDI_CONTENT_DIR / "train" / "ocr"
    if sh_train_ocr.exists():
        for txt_file in list(sh_train_ocr.glob("*.txt"))[:2500]:
            content = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sentences.append(content)

    predictor = ContextTextPredictor(n=3)
    meta = predictor.train_on_corpus(sentences)
    out_path = settings.CHECKPOINT_DIR / "context_predictor.json"
    predictor.save(out_path)
    print(f"[Stage A Completed in {time.time() - t0:.2f}s] Sentences: {meta['sentences_trained']}, Words: {meta['total_words_processed']}, Vocab: {meta['unique_vocabulary_size']}, Contexts: {meta['unique_contexts']}")
    print(f"Checkpoint saved: {out_path}\n", flush=True)
    return out_path, meta


def run_full_training():
    print("=" * 70, flush=True)
    print("Past2Pixel — Master Sequential Deep Learning Model Training Pipeline", flush=True)
    print("Unified Dataset: VedoOCR + Sanskrit-Hindi OCR VQA", flush=True)
    print("=" * 70, flush=True)
    log_system_info()

    total_start = time.time()

    # Build or verify unified dataset splits once
    splits = build_unified_dataset_splits()
    print(f"[Dataset Split Verified] Train: {len(splits['train'])} | Val: {len(splits['val'])} | Held-out Test: {len(splits['test'])}\n", flush=True)

    # 1. Stage A: Context-Aware Text Predictor
    train_context_predictor_stage(splits)

    # 2. Stage B: Convolutional Autoencoder Baseline
    print("=" * 70, flush=True)
    print(">>> STAGE B: Training Convolutional Autoencoder Baseline (6 Epochs)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    ae_ckpt = train_restoration(
        model_type="autoencoder",
        device_name="cuda",
        epochs=6,
        batch_size=16,
        lr=0.0003,
    )
    print(f"[Stage B Completed in {time.time() - t0:.2f}s] Checkpoint: {ae_ckpt}\n", flush=True)

    # 3. Stage C: U-Net Primary Restoration Model
    print("=" * 70, flush=True)
    print(">>> STAGE C: Training U-Net Primary Restoration Model (8 Epochs)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    unet_ckpt = train_restoration(
        model_type="unet",
        device_name="cuda",
        epochs=8,
        batch_size=16,
        lr=0.0002,
    )
    print(f"[Stage C Completed in {time.time() - t0:.2f}s] Checkpoint: {unet_ckpt}\n", flush=True)

    # 4. Stage D: U-Net Damage Detector
    print("=" * 70, flush=True)
    print(">>> STAGE D: Training U-Net Damage Detector (8 Epochs)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    det_ckpt = train_damage_detector(
        device_name="cuda",
        epochs=8,
        batch_size=16,
        lr=0.0003,
    )
    print(f"[Stage D Completed in {time.time() - t0:.2f}s] Checkpoint: {det_ckpt}\n", flush=True)

    total_elapsed = time.time() - total_start
    print("=" * 70, flush=True)
    print(f"ALL TRAINING COMPLETED IN {total_elapsed:.2f}s (~{total_elapsed / 60:.1f} mins)!", flush=True)
    print("=" * 70, flush=True)

    # Verify all models in registry
    model_registry.refresh_checkpoint_status()
    for name, info in model_registry.get_all_statuses().items():
        print(f"Model: {name} | Status: {info['status']} | Checkpoint: {info.get('checkpoint_path')}", flush=True)
        if info.get("training_metadata"):
            print(f"       Metadata: {info['training_metadata']}", flush=True)


if __name__ == "__main__":
    run_full_training()

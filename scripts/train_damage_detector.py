"""
Training Script for U-Net Damage Localization & Segmentation Network.
Trained on paired manuscript degradation masks using combined Binary Cross-Entropy (BCE) and Dice Loss.
"""
import argparse
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.device import get_device, get_device_info, log_system_info
from src.models.unet_damage_detector import UNetDamageDetector
from src.models.model_registry import save_checkpoint
from src.data.vedo_dataset import ManuscriptPairDataset, generate_synthetic_manuscript_sample, collate_manuscript_batch
from src.data.data_split import split_document_ids, save_split_manifest
from src.data.combined_dataset import build_unified_dataset_splits, CombinedManuscriptPairDataset
from src.metrics.damage_metrics import calculate_segmentation_metrics


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for handling severe class imbalance between
    damaged regions (tears/stains/scratches) and clean document parchment.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        )
        return (self.bce_weight * bce_loss) + ((1.0 - self.bce_weight) * dice_loss)


def train_damage_detector(
    device_name: str = "auto",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.0003,
    num_samples: int = 16,
    seed: int = 42,
) -> Path:
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    device = get_device(device_name)
    device_info = get_device_info(device)
    log_system_info()

    settings.ensure_directories()
    sample_dir = settings.SAMPLES_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover or create source manuscript samples via unified multi-dataset loader
    splits = build_unified_dataset_splits()
    train_samples = splits.get("train", [])
    val_samples = splits.get("val", [])

    if not train_samples:
        print(f"[Notice] Generating {num_samples} synthetic manuscript samples for damage detector training...")
        for i in range(num_samples):
            p = sample_dir / f"doc_{i:03d}_page_01.png"
            if not p.exists():
                generate_synthetic_manuscript_sample(p, seed=100 + i)
        image_paths = list(sample_dir.glob("*.png"))
        doc_ids = list(set([p.stem.split("_")[0] for p in image_paths]))
        local_splits = split_document_ids(doc_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        save_split_manifest(local_splits)
        train_paths = [p for p in image_paths if p.stem.split("_")[0] in local_splits["train"]]
        val_paths = [p for p in image_paths if p.stem.split("_")[0] in local_splits["val"]]
        train_dataset = ManuscriptPairDataset(train_paths, is_training=True)
        val_dataset = ManuscriptPairDataset(val_paths, is_training=False)
    else:
        v_train = len([s for s in train_samples if s.get("source_dataset") == "VedoOCR"])
        sh_train = len([s for s in train_samples if s.get("source_dataset") == "SanskritHindi_OCR"])
        v_val = len([s for s in val_samples if s.get("source_dataset") == "VedoOCR"])
        sh_val = len([s for s in val_samples if s.get("source_dataset") == "SanskritHindi_OCR"])
        print(f"[Unified Dataset Loaded] Train: {len(train_samples)} (Vedo: {v_train}, SH: {sh_train}) | Val: {len(val_samples)} (Vedo: {v_val}, SH: {sh_val})")
        train_dataset = CombinedManuscriptPairDataset(train_samples, is_training=True, seed=seed)
        val_dataset = CombinedManuscriptPairDataset(val_samples, is_training=False, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_manuscript_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_manuscript_batch, num_workers=0)

    model = UNetDamageDetector(in_channels=3).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_ckpt_path = settings.CHECKPOINT_DIR / settings.DAMAGE_DETECTOR_CHECKPOINT

    print(f"\n[Starting Damage Detector Training on {device.type.upper()}]")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            degraded = batch["degraded"].to(device)
            target_mask = batch["damage_mask"].to(device)

            optimizer.zero_grad()
            pred_mask = model(degraded)
            loss = criterion(pred_mask, target_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)

        epoch_loss = running_loss / max(1, len(train_dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        iou_list = []
        dice_list = []
        with torch.no_grad():
            for batch in val_loader:
                degraded = batch["degraded"].to(device)
                target_mask = batch["damage_mask"].to(device)
                pred_mask = model(degraded)
                loss = criterion(pred_mask, target_mask)
                val_loss += loss.item() * degraded.size(0)

                for p_t, t_t in zip(pred_mask, target_mask):
                    metrics = calculate_segmentation_metrics(p_t, t_t, threshold=0.4)
                    if metrics["iou"] is not None:
                        iou_list.append(metrics["iou"])
                        dice_list.append(metrics["dice"])

        avg_val_loss = val_loss / max(1, len(val_dataset))
        avg_iou = float(torch.tensor(iou_list).mean().item()) if iou_list else 0.0
        avg_dice = float(torch.tensor(dice_list).mean().item()) if dice_list else 0.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {avg_iou:.4f} | "
            f"Val Dice: {avg_dice:.4f}"
        )

        if avg_val_loss < best_val_loss or epoch == epochs:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=avg_val_loss,
                metrics={"val_iou": round(avg_iou, 4), "val_dice": round(avg_dice, 4)},
                device_info=device_info,
                output_path=best_ckpt_path,
            )

    elapsed = time.time() - start_time
    print(f"\n[Damage Detector Training Finished] Elapsed: {elapsed:.2f}s. Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Past2Pixel Damage Detector Training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()

    train_damage_detector(
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
    )

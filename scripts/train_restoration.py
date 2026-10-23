"""
Training Script for Manuscript Image Restoration Models (U-Net & Autoencoder Baseline).
Supports both local CPU smoke-testing and high-performance CUDA execution on GPU machines.
"""
import argparse
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.device import get_device, get_device_info, log_system_info
from src.models.unet_restoration import UNetRestoration
from src.models.autoencoder import ConvAutoencoder
from src.models.model_registry import save_checkpoint
from src.data.vedo_dataset import ManuscriptPairDataset, generate_synthetic_manuscript_sample, collate_manuscript_batch
from src.data.data_split import split_document_ids, save_split_manifest
from src.data.combined_dataset import build_unified_dataset_splits, CombinedManuscriptPairDataset
from src.metrics.restoration_metrics import calculate_psnr, calculate_ssim


class RestorationLoss(nn.Module):
    """
    Composite Loss for Document Image Restoration:
    L1 loss (for edge sharpness and text contrast) + MSE loss (for smooth background).
    """

    def __init__(self, alpha_l1: float = 0.8, alpha_mse: float = 0.2):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_mse = alpha_mse
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.alpha_l1 * self.l1(pred, target)) + (self.alpha_mse * self.mse(pred, target))


def train_restoration(
    model_type: str = "unet",
    device_name: str = "auto",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.0002,
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
        print(f"[Notice] Raw datasets not found. Generating {num_samples} synthetic Devanagari manuscript samples in {sample_dir}...")
        for i in range(num_samples):
            p = sample_dir / f"doc_{i:03d}_page_01.png"
            generate_synthetic_manuscript_sample(p, seed=seed + i)
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

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_manuscript_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_manuscript_batch, num_workers=0)

    # Instantiate Model
    if model_type.lower() == "autoencoder":
        model = ConvAutoencoder(in_channels=3).to(device)
        ckpt_filename = settings.AUTOENCODER_CHECKPOINT
    else:
        model = UNetRestoration(in_channels=3, out_channels=3).to(device)
        ckpt_filename = settings.UNET_RESTORATION_CHECKPOINT

    criterion = RestorationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_ckpt_path = settings.CHECKPOINT_DIR / ckpt_filename

    print(f"\n[Starting Training: {model.__class__.__name__} on {device.type.upper()}]")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            output = model(degraded)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)

        epoch_loss = running_loss / max(1, len(train_dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for batch in val_loader:
                degraded = batch["degraded"].to(device)
                clean = batch["clean"].to(device)
                output = model(degraded)
                loss = criterion(output, clean)
                val_loss += loss.item() * degraded.size(0)

                for pred_t, clean_t in zip(output, clean):
                    p_np = pred_t.permute(1, 2, 0).cpu().numpy()
                    c_np = clean_t.permute(1, 2, 0).cpu().numpy()
                    psnr_list.append(calculate_psnr(p_np, c_np))
                    ssim_list.append(calculate_ssim(p_np, c_np))

        avg_val_loss = val_loss / max(1, len(val_dataset))
        avg_psnr = float(torch.tensor(psnr_list).mean().item()) if psnr_list else 0.0
        avg_ssim = float(torch.tensor(ssim_list).mean().item()) if ssim_list else 0.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PSNR: {avg_psnr:.2f} dB | "
            f"Val SSIM: {avg_ssim:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=avg_val_loss,
                metrics={"val_psnr": round(avg_psnr, 2), "val_ssim": round(avg_ssim, 4)},
                device_info=device_info,
                output_path=best_ckpt_path,
            )

    elapsed = time.time() - start_time
    print(f"\n[Training Finished] Elapsed: {elapsed:.2f}s. Checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Past2Pixel Restoration Model Training")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "autoencoder"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()

    train_restoration(
        model_type=args.model,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
    )

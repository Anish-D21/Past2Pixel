# Past2Pixel — GPU Training Guide (For Training on Friend's GPU Machine)

This guide provides exact, step-by-step instructions for running high-epoch model training on your friend's GPU machine (NVIDIA CUDA) and copying the trained weights back to your local development machine.

---

### Step 1: Copy the Project Folder to Your Friend's Machine
You can transfer the `Past2Pixel` project folder using a USB flash drive, Google Drive, or Git:
```bash
git clone <your-repo-url>
cd Past2Pixel
```

---

### Step 2: Install CUDA-Enabled PyTorch on Your Friend's PC
Ensure Python 3.11 is installed, then install GPU-accelerated PyTorch:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pydantic-settings opencv-python-headless
```

Verify GPU availability:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Step 3: Run the Training Scripts

#### A. Train U-Net Primary Restoration Model (GPU)
```powershell
python scripts/train_restoration.py --model unet --device cuda --epochs 25 --batch-size 8 --lr 0.0002
```
*Output checkpoint:* `checkpoints/unet_restoration_best.pt`

#### B. Train Autoencoder Baseline Model (GPU)
```powershell
python scripts/train_restoration.py --model autoencoder --device cuda --epochs 20 --batch-size 8 --lr 0.0003
```
*Output checkpoint:* `checkpoints/autoencoder_baseline_best.pt`

#### C. Train U-Net Damage Detector / Localization Model (GPU)
```powershell
python scripts/train_damage_detector.py --device cuda --epochs 25 --batch-size 8 --lr 0.0003
```
*Output checkpoint:* `checkpoints/damage_detector_best.pt`

---

### Step 4: Copy Checkpoints Back to Your Laptop
Once training completes on your friend's PC, copy the files inside the `checkpoints/` folder:
- `checkpoints/unet_restoration_best.pt`
- `checkpoints/autoencoder_baseline_best.pt`
- `checkpoints/damage_detector_best.pt`

Paste them directly into your local `checkpoints/` directory at `d:\Desktop\Past2Pixel\checkpoints\`.

---

### Step 5: Verify on Your PC (CPU Mode)
Run the verification smoke test on your laptop:
```powershell
python scripts/smoke_test.py
```
Your local system will automatically detect the checkpoints, switch model readiness to `READY_FOR_INFERENCE`, and run inference seamlessly on your CPU!

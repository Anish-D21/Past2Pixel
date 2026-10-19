@echo off
REM Past2Pixel — One-Click GPU Training Script for Friend's Machine
echo ======================================================================
echo Past2Pixel: Starting GPU Training for Restoration and Damage Models
echo ======================================================================

echo [1/3] Training U-Net Restoration Model on CUDA...
python scripts/train_restoration.py --model unet --device cuda --epochs 25 --batch-size 8
if %ERRORLEVEL% NEQ 0 (
    echo Error during U-Net restoration training.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/3] Training Convolutional Autoencoder Baseline on CUDA...
python scripts/train_restoration.py --model autoencoder --device cuda --epochs 20 --batch-size 8
if %ERRORLEVEL% NEQ 0 (
    echo Error during Autoencoder training.
    pause
    exit /b %ERRORLEVEL%
)

echo [3/3] Training U-Net Damage Detector on CUDA...
python scripts/train_damage_detector.py --device cuda --epochs 25 --batch-size 8
if %ERRORLEVEL% NEQ 0 (
    echo Error during Damage Detector training.
    pause
    exit /b %ERRORLEVEL%
)

echo ======================================================================
echo Training Complete! 
echo All checkpoints saved in the 'checkpoints/' folder.
echo Copy the .pt files back to your laptop.
echo ======================================================================
pause

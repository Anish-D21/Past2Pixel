# Past2Pixel — Deep Learning Model Documentation

## 1. Primary U-Net Restoration Architecture (`UNetRestoration`)

### Architecture Specifications
- **Input Dimensions:** `(B, 3, 256, 256)` (RGB degraded manuscript patch/folio)
- **Output Dimensions:** `(B, 3, 256, 256)` (RGB restored clean output in $[0.0, 1.0]$)
- **Encoder (Contracting Path):**
  - Block 1: $3 \to 32 \to 32$ filters, $2\times 2$ MaxPool ($256 \to 128$)
  - Block 2: $32 \to 64 \to 64$ filters, $2\times 2$ MaxPool ($128 \to 64$)
  - Block 3: $64 \to 128 \to 128$ filters, $2\times 2$ MaxPool ($64 \to 32$)
  - Block 4: $128 \to 256 \to 256$ filters, $2\times 2$ MaxPool ($32 \to 16$)
- **Bottleneck:** $256 \to 512 \to 256$ with Dropout($0.2$)
- **Decoder (Expanding Path):**
  - UpBlock 1: Bilinear upsampling ($16 \to 32$) + skip concatenation ($256 + 256 \to 512$) $\to 128$ filters
  - UpBlock 2: Bilinear upsampling ($32 \to 64$) + skip concatenation ($128 + 128 \to 256$) $\to 64$ filters
  - UpBlock 3: Bilinear upsampling ($64 \to 128$) + skip concatenation ($64 + 64 \to 128$) $\to 32$ filters
  - UpBlock 4: Bilinear upsampling ($128 \to 256$) + skip concatenation ($32 + 32 \to 64$) $\to 32$ filters
- **Head:** $1\times 1$ Conv $32 \to 3$ with Sigmoid activation.
- **Total Parameters:** ~7,700,000 parameters.
- **Why U-Net?** The skip connections allow fine spatial features (thin ink strokes, diacritics, and ligatures) from early contracting layers to bypass the bottleneck directly to decoding layers, preventing the over-smoothing common in standard Autoencoders.

---

## 2. Convolutional Autoencoder Baseline (`ConvAutoencoder`)

### Architecture Specifications
- **Input Dimensions:** `(B, 3, 256, 256)`
- **Output Dimensions:** `(B, 3, 256, 256)`
- **Encoder:** 3 downsampling blocks ($3 \to 32 \to 64 \to 128$) with MaxPool.
- **Bottleneck:** $128 \to 256 \to 128$ filters.
- **Decoder:** 3 transpose convolutional upsampling blocks ($128 \to 64 \to 32 \to 3$) with Sigmoid output.
- **Total Parameters:** ~1,200,000 parameters.
- **Role & Limitations:** Serves as the control baseline for empirical comparison against U-Net. Because it lacks skip connections, all spatial edge information must compress through the bottleneck, making it blurrier on intricate Sanskrit ligatures.

---

## 3. U-Net Damage Detector / Localization (`UNetDamageDetector`)

### Architecture Specifications
- **Input:** `(B, 3, 256, 256)`
- **Output:** `(B, 1, 256, 256)` (Single-channel per-pixel probability map)
- **Activation:** Sigmoid
- **Loss:** Binary Cross-Entropy + Dice Loss (50/50 weighted):
  $$\mathcal{L} = 0.5 \cdot \text{BCE} + 0.5 \cdot (1 - \text{Dice})$$
- **Visual Overlay:** Generated using alpha-blended crimson highlighting ($R=230, G=50, B=40$) in areas where damage probability exceeds the user threshold (default $0.35$).

---

## 4. Context-Aware Text Predictor (`ContextTextPredictor`)

### Architecture & Math
- Statistical $N$-gram Language Model ($N=2, 3$) trained on authentic Sanskrit/Hindi Devanagari text corpora.
- Backoff strategy: Trigram $\to$ Bigram $\to$ Unigram.
- Outputs speculative candidate tokens with contextual probability and strict uncertainty disclaimers.
- **Evaluation:** Standard Character Error Rate (CER) and Word Error Rate (WER) via Levenshtein dynamic programming.

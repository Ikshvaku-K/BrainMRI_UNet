# 🧠 NeuroScan-UNet — Brain Tumor Classification via Modified U-Net

> A deep learning pipeline that repurposes the U-Net encoder for 4-class brain tumor classification from MRI scans, augmented with Grad-CAM explainability to produce visual attention heatmaps over detected regions.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Architecture Design](#architecture-design)
   - [Why U-Net?](#why-u-net)
   - [The Modification: Encoder-Only Classification](#the-modification-encoder-only-classification)
   - [Architecture Diagram](#architecture-diagram)
5. [Training Pipeline](#training-pipeline)
   - [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
   - [Loss Function & Optimizer](#loss-function--optimizer)
   - [Training Loop](#training-loop)
6. [Grad-CAM Explainability](#grad-cam-explainability)
7. [Results](#results)
8. [Project Structure](#project-structure)
9. [Getting Started](#getting-started)
10. [Dependencies](#dependencies)
11. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## Project Overview

**NeuroScan-UNet** is a medical image classification system that classifies brain MRI scans into four tumor categories:

| Class | Description |
|---|---|
| `glioma` | A tumor that starts in the glial cells of the brain or spine |
| `meningioma` | A tumor arising from the meninges surrounding the brain |
| `pituitary` | A tumor growing in the pituitary gland at the base of the brain |
| `notumor` | Healthy brain tissue with no observable tumor |

The core contribution is an **architectural adaptation of U-Net**: instead of performing pixel-wise segmentation (which requires spatial ground-truth masks), the decoder is removed and replaced with a lightweight classification head attached directly to the bottleneck. This preserves all the representational power of the U-Net encoder while drastically reducing parameters and memory usage.

---

## Problem Statement

Standard U-Net is designed for segmentation — it outputs a spatial mask the same size as the input. This requires **pixel-level annotations**, which are expensive and time-consuming to produce for medical datasets.

The Kaggle Brain MRI dataset provides only **image-level labels** (i.e., which tumor type is present), with no segmentation masks. A naive approach would be to fine-tune a standard CNN classifier (e.g., ResNet, VGG), but these architectures were not designed with the fine-grained texture sensitivity needed for MRI analysis.

**The insight:** U-Net's encoder is an exceptionally powerful feature extractor for biomedical images due to its double-convolution blocks and progressive spatial downsampling. We can exploit this without needing segmentation masks — by simply replacing the decoder with a global pooling + dense classification head.

---

## Dataset

The dataset follows a structured directory layout used by PyTorch's `ImageFolder`:

```
BrainMRI/
├── Training/
│   ├── glioma/          # ~826 MRI images
│   ├── meningioma/      # ~822 MRI images
│   ├── notumor/         # ~395 MRI images
│   └── pituitary/       # ~827 MRI images
└── Testing/
    ├── glioma/          # ~100 MRI images
    ├── meningioma/      # ~115 MRI images
    ├── notumor/         # ~105 MRI images
    └── pituitary/       # ~74 MRI images
```

> **Note:** The dataset is not included in this repository. It is sourced from the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Download it and place the `Training/` and `Testing/` folders at the root of your local `BrainMRI/` directory.

All images are resized to **224×224** pixels for uniform input to the model.

---

## Architecture Design

### Why U-Net?

U-Net was introduced by Ronneberger et al. (2015) specifically for biomedical image segmentation. Its key strength is the **encoder path**, which uses repeated double-convolution blocks (`Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU`) interleaved with max-pooling to progressively extract hierarchical features:

- **Early layers (64–128 channels)**: Capture low-level features — edges, gradients, tissue boundaries.
- **Middle layers (256–512 channels)**: Capture mid-level structure — shape of ventricles, asymmetry patterns.
- **Bottleneck (1024 channels)**: Captures high-level semantic features — tumor morphology, intensity distribution.

This hierarchical sensitivity to texture and structure is precisely what distinguishes MRI tumor classification from natural image classification.

### The Modification: Encoder-Only Classification

Standard U-Net decodes the bottleneck back to the input resolution using transposed convolutions and skip connections. For **classification**, this is wasteful — we don't need spatial predictions.

Instead:

```
Input (3 × 224 × 224)
    │
    ▼
[DoubleConv]  →  64 ch  (224 × 224)
    │ MaxPool
    ▼
[DoubleConv]  →  128 ch (112 × 112)
    │ MaxPool
    ▼
[DoubleConv]  →  256 ch (56 × 56)
    │ MaxPool
    ▼
[DoubleConv]  →  512 ch (28 × 28)
    │ MaxPool
    ▼
[DoubleConv]  →  1024 ch (14 × 14)   ← Bottleneck
    │
    ▼ AdaptiveAvgPool2d (1×1)
    │
    ▼ Flatten → 1024-dim vector
    │
    ▼ Linear(1024 → 512) + ReLU + Dropout(0.5)
    │
    ▼ Linear(512 → 4)
    │
    ▼ Class Logits (Glioma / Meningioma / Pituitary / No Tumor)
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     U-Net ENCODER                           │
│                                                             │
│  Input (3,224,224)                                          │
│       │                                                     │
│  ┌────▼────┐                                                │
│  │ inc     │  DoubleConv  →  (64, 224, 224)                 │
│  └────┬────┘                                                │
│  MaxPool2d                                                  │
│  ┌────▼────┐                                                │
│  │ down1   │  DoubleConv  →  (128, 112, 112)                │
│  └────┬────┘                                                │
│  MaxPool2d                                                  │
│  ┌────▼────┐                                                │
│  │ down2   │  DoubleConv  →  (256, 56, 56)                  │
│  └────┬────┘                                                │
│  MaxPool2d                                                  │
│  ┌────▼────┐                                                │
│  │ down3   │  DoubleConv  →  (512, 28, 28)                  │
│  └────┬────┘                                                │
│  MaxPool2d                                                  │
│  ┌────▼────┐                                                │
│  │ down4   │  DoubleConv  →  (1024, 14, 14) [BOTTLENECK]    │
│  └────┬────┘                                                │
└───────┼─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                 CLASSIFICATION HEAD                          │
│                                                             │
│  AdaptiveAvgPool2d → (1024, 1, 1)                           │
│  Flatten           → 1024                                   │
│  Linear(1024→512) + ReLU + Dropout(0.5)                     │
│  Linear(512→4)     → Logits                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

### Data Preprocessing & Augmentation

Training and testing transforms are defined separately using `torchvision.transforms`:

**Training transforms:**
```python
transforms.Resize((224, 224))        # Uniform input size
transforms.RandomHorizontalFlip()    # Augmentation: spatial invariance
transforms.RandomRotation(15)        # Augmentation: orientation invariance
transforms.ToTensor()
transforms.Normalize(                # ImageNet mean/std normalization
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

**Testing transforms:** Resize + ToTensor + Normalize only (no augmentation).

The normalization uses ImageNet statistics. While MRI images are grayscale in origin, the dataset provides them as 3-channel RGB JPEGs. ImageNet normalization still provides a reasonable initialization point and stabilizes training.

**Augmentation rationale:**
- `RandomHorizontalFlip`: Brain tumors are not symmetric; flipping forces the model to rely on texture rather than position.
- `RandomRotation(15)`: MRI scans can be acquired at slightly varying head orientations. Small rotations improve robustness.

### Loss Function & Optimizer

| Component | Choice | Rationale |
|---|---|---|
| **Loss** | `CrossEntropyLoss` | Numerically stable; internally applies `LogSoftmax + NLLLoss` over 4 classes |
| **Optimizer** | `Adam (lr=1e-4)` | Adaptive learning rates; prevents classification head from overriding encoder features too aggressively |
| **Batch Size** | `32` | Balances GPU memory utilization and gradient noise |
| **Epochs** | `5` | Sufficient for convergence given the dataset size; revisit for fine-tuning |
| **Dropout** | `0.5` | Applied in the classification head to regularize against overfitting on the limited dataset |

### Training Loop

Each epoch performs:
1. **Forward pass** through the model
2. **Loss computation** via `CrossEntropyLoss`
3. **Backpropagation** + weight update via `Adam`
4. **Validation pass** (no gradients) on the test split — computes validation accuracy

Progress is tracked with `tqdm` progress bars, and per-epoch metrics (loss, train accuracy, val accuracy) are stored in `history` for post-training visualization.

After training, results are saved as:
- `unet_brain_mri.pth` — model weights (PyTorch state dict)
- `training_curves.png` — side-by-side plots of train loss and train/val accuracy curves

---

## Grad-CAM Explainability

Beyond classification, the `visualize_boundaries.py` script implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate visual explanations for the model's predictions.

### How Grad-CAM Works

Grad-CAM answers: *"Which spatial regions in the input did the model focus on when making its prediction?"*

1. **Forward pass**: Run the image through the model; record activations at the target layer (`down4` — the bottleneck).
2. **Backward pass**: Compute the gradient of the predicted class score with respect to those activations.
3. **Weight activations**: Pool the gradients spatially (`mean over H, W`) to get per-channel importance weights.
4. **Generate CAM**: Take a weighted sum of activation maps, apply ReLU (keep only positive contributions), and normalize.
5. **Overlay**: Resize the 14×14 CAM to 224×224 and blend it over the original MRI using `cv2.addWeighted`.

```python
# Target layer: bottleneck (most semantically rich)
grad_cam = GradCAM(model, model.down4)

# Produces a normalized heatmap showing where the model "looks"
cam, pred_idx = grad_cam(input_tensor)
```

The output heatmaps are saved as `heatmap_result_<i>_<predicted_class>.png`, showing:
- **Left panel**: Original MRI scan
- **Right panel**: JET colormap heatmap overlaid on the MRI (red = high attention, blue = low attention)

This is critical for medical AI — it provides radiologist-interpretable evidence for why the model made a particular decision, supporting clinical trust and validation.

---

## Results

| Metric | Value |
|---|---|
| **Final Train Accuracy** | ~97% |
| **Final Validation Accuracy** | ~86% |
| **Architecture** | U-Net Encoder + Classification Head |
| **Parameters** | ~14.8M |
| **Input Resolution** | 224 × 224 × 3 |

**Training & Validation Curves:**

![Training Curves](training_curves.png)

**Grad-CAM Heatmap Samples:**

| Glioma | Meningioma | Pituitary |
|---|---|---|
| ![Glioma](heatmap_result_1_glioma.png) | ![Meningioma](heatmap_result_2_meningioma.png) | ![Pituitary](heatmap_result_3_pituitary.png) |

---

## Project Structure

```
BrainMRI/
├── .gitignore                   # Excludes datasets, weights, and caches
├── UNet_Project/
│   ├── train_unet.py            # Model definition + full training pipeline
│   ├── visualize_boundaries.py  # Grad-CAM inference + heatmap generation
│   ├── README.md                # This file
│   ├── deep_dive.md             # Detailed architectural analysis
│   └── .gitignore               # Subdirectory-level ignores
```

> **Not tracked in git (excluded via `.gitignore`):**
> `Training/`, `Testing/`, `*.pth`, `*.png`, `venv/`, `__pycache__/`

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (CUDA-enabled recommended)
- The Brain Tumor MRI Dataset from Kaggle

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Ikshvaku-K/BrainMRI_UNet.git
cd BrainMRI_UNet

# 2. Download and place the dataset
# Place Training/ and Testing/ folders at:
# BrainMRI/Training/ and BrainMRI/Testing/

# 3. Activate the virtual environment
source UNet_Project/venv/bin/activate

# 4. Train the model
cd UNet_Project
python train_unet.py
# Outputs: unet_brain_mri.pth, training_curves.png

# 5. Run Grad-CAM visualization
python visualize_boundaries.py
# Outputs: heatmap_result_<i>_<class>.png
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `torchvision` | Dataset loading, transforms, ImageFolder |
| `opencv-python` | Heatmap generation and image blending |
| `matplotlib` | Training curve plots and visualization panels |
| `tqdm` | Progress bars for training loop |
| `numpy` | Array operations in Grad-CAM |
| `Pillow` | Image loading for Grad-CAM inference |

---

## Design Decisions & Tradeoffs

| Decision | Chosen Approach | Alternative | Why |
|---|---|---|---|
| **Base Architecture** | U-Net Encoder | ResNet-50, VGG-16 | U-Net is tuned for biomedical textures; better edge sensitivity |
| **Decoder** | Removed (replaced with GAP) | Keep decoder + add classification token | Saves ~50% parameters; no segmentation masks available |
| **Pooling** | `AdaptiveAvgPool2d` | Flatten directly | Spatial agnostic; handles variable bottleneck sizes |
| **Explainability** | Grad-CAM on `down4` | LIME, SHAP | Grad-CAM is fast, spatial, and radiologist-friendly |
| **Normalization** | ImageNet stats | MRI-specific stats | Dataset provided as RGB JPEGs; acceptable approximation |
| **Augmentation** | Flip + Rotation | Color jitter, CutMix | MRI intensity distributions should not be perturbed |

---

*Built with PyTorch · Grad-CAM · OpenCV*

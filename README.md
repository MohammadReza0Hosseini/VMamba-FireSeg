# VMamba-FireSeg

VMamba-based wildfire and fire segmentation using RGB images and binary segmentation masks.

## Project overview

This project uses a hierarchical VMamba encoder and a U-Net-style decoder for pixel-level fire segmentation.

- **Task:** Binary semantic segmentation (fire / background)
- **Input:** RGB images
- **Input resolution:** 256 × 256
- **Model:** MyUnet with VMamba encoder and U-Net-like decoder
- **Encoder dimensions:** (96, 192, 384, 768)
- **Encoder depths:** (2, 2, 8, 2)
- **Training:** 30 epochs
- **Training batch size:** 128
- **Output:** One-channel segmentation map

## Dataset layout

The training code expects the image and mask folders in this layout:

```text
project/
├── Image/
│   ├── Fire/
│   └── Not_Fire/
└── Segmentation_Mask/
    └── Fire/
```

Fire images are paired with masks by filename. Not-fire images are treated as background samples and use empty masks. The dataset itself is not included in this repository.

## Validation results

The reported validation run used 7,771 samples and a prediction threshold of 0.5.

| Metric | Result |
|---|---:|
| Pixel Accuracy | 0.996006 |
| Precision | 0.817799 |
| Recall | 0.885713 |
| F1 Score | 0.850403 |
| Specificity | 0.997438 |
| Global IoU | 0.739739 |
| Global Dice | 0.850403 |
| Mean IoU (per image) | 0.799273 |
| Mean Dice (per image) | 0.871411 |
| Fire detection rate | 0.991848 |
| No-fire false-positive image rate | 0.022657 |

These are results from the current experiment and should be interpreted with the validation split and evaluation procedure used by the project.

## Execution platform and environment

**The reported training and validation experiment was executed in a cloud-hosted Marimo notebook environment**, not on a local Windows PC or Kaggle.

Verified runtime details:

- Notebook/runtime: Marimo (cloud environment)
- Working directory: `/marimo`
- Python 3.13.11
- PyTorch 2.11.0+cu130
- CUDA 13.0
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0)
- `mamba-ssm` 2.3.2.post1
- `causal-conv1d` 1.7.0

The source code is a standalone Python script and can also be run outside Marimo, provided the compatible CUDA/PyTorch and package environment is configured.

Install the Python dependencies listed in [requirements.txt](requirements.txt). The PyTorch wheel must match the target CUDA platform; the experiment used the CUDA 13.0 build.

Environment reports and the kernel backup archive are stored separately in Google Drive.

## Model checkpoint

The trained checkpoint, `best_vmamba_fire.pth` (approximately 132.63 MB), is stored separately in Google Drive and is intentionally not committed to this repository.

## Repository contents

The source code, evaluation utilities, and documentation will be maintained here. Dataset files and large model artifacts are kept outside GitHub.

## Notes

- The validation split was created with an 80/20 split and `random_state=42`.
- Validation images were resized to 256 × 256 and normalized.
- The reported threshold was 0.5.
- Reproduce results using the matching source code, environment, and dataset version.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMamba Fire Segmentation — CUDA integration, validation, benchmark and training.

Target environment tested by the user:
  PyTorch 2.11.0+cu130, CUDA 13.0, RTX PRO 6000 Blackwell (sm_120)
  mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0

Architecture is kept at 256x256 with dims=(96,192,384,768),
depths=(2,2,8,2), d_state=1, ssm_ratio=1, mlp_ratio=4 and
four independent scan directions (row+, row-, column+, column-).

Examples:
  python VMamba_FireSeg_Complete.py --mode validate
  python VMamba_FireSeg_Complete.py --mode benchmark
  python VMamba_FireSeg_Complete.py --mode train --data-root /content --epochs 30

For Marimo, this is a normal Python module. Import it from a cell, then call
main([...]) or use the command line. No notebook cell magic is required.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# --------------------------- USER CONFIGURATION ---------------------------

IMAGE_SIZE = 256
BATCH_SIZE = 128
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

MODEL_DIMS = (96, 192, 384, 768)
MODEL_DEPTHS = (2, 2, 8, 2)
D_STATE = 1
SSM_RATIO = 1.0
MLP_RATIO = 4.0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 30
SEED = 42
CHECKPOINT_PATH = "best_vmamba_fire_cuda.pth"
HISTORY_PATH = "vmamba_fire_history.json"

# CUDA overlay created and tested in the previous setup conversation.
CUDA_OVERLAY = "/tmp/cuda13_overlay"


def configure_cuda_environment() -> None:
    """Set CUDA build/runtime paths without reinstalling any packages."""
    if Path(CUDA_OVERLAY).exists():
        os.environ.setdefault("CUDA_HOME", CUDA_OVERLAY)
        os.environ.setdefault("CUDA_PATH", CUDA_OVERLAY)
        os.environ.setdefault("PATH", "")
        os.environ["PATH"] = f"{CUDA_OVERLAY}/bin:" + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = (
            f"{CUDA_OVERLAY}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        os.environ["LIBRARY_PATH"] = (
            f"{CUDA_OVERLAY}/lib:" + os.environ.get("LIBRARY_PATH", "")
        )
        os.environ["CPATH"] = (
            f"{CUDA_OVERLAY}/include:" + os.environ.get("CPATH", "")
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "2")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict:
    import torch
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_overlay": CUDA_OVERLAY,
        "cuda_overlay_exists": Path(CUDA_OVERLAY).exists(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report.update({
            "gpu": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "vram_gb": round(p.total_memory / 1e9, 2),
        })
    print("\n========== ENVIRONMENT ==========")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# --------------------------- CUDA KERNEL IMPORT ---------------------------

configure_cuda_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F

SELECTIVE_SCAN_FN = None
FUSED_SCAN_AVAILABLE = False
FUSED_SCAN_IMPORT_ERROR = None
CAUSAL_CONV_AVAILABLE = False
CAUSAL_CONV_IMPORT_ERROR = None

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as SELECTIVE_SCAN_FN,
    )
    # Importing the compiled extension explicitly catches missing/broken .so files.
    import selective_scan_cuda  # noqa: F401
    FUSED_SCAN_AVAILABLE = True
except Exception as exc:
    FUSED_SCAN_IMPORT_ERROR = repr(exc)

try:
    import causal_conv1d_cuda  # noqa: F401
    CAUSAL_CONV_AVAILABLE = True
except Exception as exc:
    CAUSAL_CONV_IMPORT_ERROR = repr(exc)


# --------------------------- MODEL COMPONENTS ---------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SelectiveScan1D(nn.Module):
    """One S6 scan direction. Input/output layout: [B,L,C]."""

    def __init__(self, dim: int, d_state: int = D_STATE):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(dim / 16))
        self.x_proj = nn.Linear(dim, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)

        # Stable negative diagonal A initialization.
        base = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(base).repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, u):
        bsz, length, channels = u.shape
        params = self.x_proj(u)
        dt_low, b_state, c_state = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if FUSED_SCAN_AVAILABLE and u.is_cuda:
            # Mamba kernel layout:
            # u, delta: [B,C,L], A: [C,N], B/C: [B,N,L].
            u_scan = u.transpose(1, 2).contiguous()
            delta_scan = F.linear(
                dt_low, self.dt_proj.weight, bias=None
            ).transpose(1, 2).contiguous()
            B_scan = b_state.transpose(1, 2).contiguous()
            C_scan = c_state.transpose(1, 2).contiguous()
            A_scan = -torch.exp(self.A_log.float())
            D_scan = self.D.float()

            out = SELECTIVE_SCAN_FN(
                u_scan,
                delta_scan,
                A_scan,
                B_scan,
                C_scan,
                D_scan,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            return out.transpose(1, 2).contiguous()

        # Correct but deliberately slow fallback for CPU/debug only.
        delta = F.softplus(self.dt_proj(dt_low).float())
        u32 = u.float()
        b32, c32 = b_state.float(), c_state.float()
        A = -torch.exp(self.A_log.float())
        state = torch.zeros(
            bsz, channels, self.d_state, device=u.device, dtype=torch.float32
        )
        outputs = []
        for t in range(length):
            ut, dt = u32[:, t], delta[:, t]
            bt, ct = b32[:, t], c32[:, t]
            a_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            b_bar_u = dt.unsqueeze(-1) * bt.unsqueeze(1) * ut.unsqueeze(-1)
            state = a_bar * state + b_bar_u
            yt = (state * ct.unsqueeze(1)).sum(-1) + self.D.float() * ut
            outputs.append(yt)
        return torch.stack(outputs, dim=1).to(dtype=u.dtype)


class SS2D(nn.Module):
    """
    Four independent scan directions:
      0: row-major forward
      1: row-major reverse
      2: column-major forward
      3: column-major reverse
    Reverse outputs are restored to original order; column outputs are reshaped
    back to HxW before summation (cross-merge).
    """

    def __init__(self, dim: int, d_state: int = D_STATE,
                 ssm_ratio: float = SSM_RATIO, d_conv: int = 3):
        super().__init__()
        inner = int(dim * ssm_ratio)
        self.inner = inner
        self.in_proj = nn.Linear(dim, inner, bias=False)
        self.dwconv = nn.Conv2d(
            inner, inner, kernel_size=d_conv, padding=d_conv // 2,
            groups=inner, bias=True
        )
        self.act = nn.SiLU()
        self.scans = nn.ModuleList([
            SelectiveScan1D(inner, d_state=d_state) for _ in range(4)
        ])
        self.out_norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        b, h, w, _ = x.shape
        y = self.in_proj(x).permute(0, 3, 1, 2).contiguous()
        y = self.act(self.dwconv(y)).permute(0, 2, 3, 1).contiguous()

        row = y.reshape(b, h * w, self.inner)
        col = y.transpose(1, 2).contiguous().reshape(b, h * w, self.inner)

        y0 = self.scans[0](row)
        y1 = torch.flip(self.scans[1](torch.flip(row, dims=[1])), dims=[1])

        col_f = self.scans[2](col)
        y2 = col_f.reshape(b, w, h, self.inner).transpose(1, 2).contiguous()
        y2 = y2.reshape(b, h * w, self.inner)

        col_r = self.scans[3](torch.flip(col, dims=[1]))
        y3 = torch.flip(col_r, dims=[1])
        y3 = y3.reshape(b, w, h, self.inner).transpose(1, 2).contiguous()
        y3 = y3.reshape(b, h * w, self.inner)

        merged = (y0 + y1 + y2 + y3).reshape(b, h, w, self.inner)
        return self.out_proj(self.out_norm(merged))


class VSSBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = D_STATE,
                 ssm_ratio: float = SSM_RATIO, mlp_ratio: float = MLP_RATIO):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, ssm_ratio=ssm_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchPartition(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=4, stride=4, bias=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class VSSDownsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.reduction = nn.Conv2d(
            in_dim, out_dim, kernel_size=2, stride=2, bias=True
        )

    def forward(self, x):
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        return self.reduction(x).permute(0, 2, 3, 1).contiguous()


class VSSStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(dim) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VMambaEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        d1, d2, d3, d4 = MODEL_DIMS
        n1, n2, n3, n4 = MODEL_DEPTHS
        self.patch_partition = PatchPartition(in_channels, d1)
        self.stage1 = VSSStage(d1, n1)
        self.down1 = VSSDownsample(d1, d2)
        self.stage2 = VSSStage(d2, n2)
        self.down2 = VSSDownsample(d2, d3)
        self.stage3 = VSSStage(d3, n3)
        self.down3 = VSSDownsample(d3, d4)
        self.stage4 = VSSStage(d4, n4)

    def forward(self, x):
        x = self.patch_partition(x)
        e1 = self.stage1(x)
        e2 = self.stage2(self.down1(e1))
        e3 = self.stage3(self.down2(e2))
        e4 = self.stage4(self.down3(e3))
        return tuple(
            f.permute(0, 3, 1, 2).contiguous() for f in (e1, e2, e3, e4)
        )


class MyUnet(nn.Module):
    """VMamba hierarchical encoder + U-Net style decoder."""

    def __init__(self, in_channels: int = 3, classes: int = 1):
        super().__init__()
        self.encoder0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.backbone = VMambaEncoder(in_channels=in_channels)
        self.decoder4 = DecoderBlock(768, 384, 384)
        self.decoder3 = DecoderBlock(384, 192, 192)
        self.decoder2 = DecoderBlock(192, 96, 96)
        self.decoder1 = DecoderBlock(96, 64, 64)
        self.decoder0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 32),
        )
        self.segmentation_head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        e0 = self.encoder0(x)
        e1, e2, e3, e4 = self.backbone(x)
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        d0 = F.interpolate(d1, size=input_size, mode="bilinear",
                            align_corners=False)
        e0_full = F.interpolate(e0, size=input_size, mode="bilinear",
                                align_corners=False)
        d0 = self.decoder0(torch.cat([d0, e0_full], dim=1))
        return self.segmentation_head(d0)


# --------------------------- VALIDATION HELPERS ---------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_causal_conv1d() -> bool:
    print("\n========== CAUSAL-CONV1D TEST ==========")
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        return False
    if not CAUSAL_CONV_AVAILABLE:
        print("FAIL: causal_conv1d_cuda import failed:", CAUSAL_CONV_IMPORT_ERROR)
        return False
    try:
        from causal_conv1d import causal_conv1d_fn
        x = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        weight = torch.randn(16, 3, device="cuda", dtype=torch.float16,
                             requires_grad=True)
        bias = torch.randn(16, device="cuda", dtype=torch.float16,
                           requires_grad=True)
        y = causal_conv1d_fn(x, weight, bias, activation="silu")
        y.float().mean().backward()
        _sync()
        ok = (y.shape == x.shape and torch.isfinite(y).all().item()
              and x.grad is not None and weight.grad is not None
              and bias.grad is not None)
        print("output:", tuple(y.shape), y.dtype, y.device)
        print("input/weight/bias gradients:", x.grad is not None,
              weight.grad is not None, bias.grad is not None)
        print("RESULT:", "PASS" if ok else "FAIL")
        return bool(ok)
    except Exception:
        traceback.print_exc()
        return False


def test_selective_scan() -> bool:
    print("\n========== MAMBA SELECTIVE-SCAN TEST ==========")
    if not torch.cuda.is_available() or not FUSED_SCAN_AVAILABLE:
        print("SKIP/FAIL: fused selective scan unavailable:",
              FUSED_SCAN_IMPORT_ERROR)
        return False
    try:
        b, c, length, n = 2, 16, 128, 4
        u = torch.randn(b, c, length, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        delta = torch.randn(b, c, length, device="cuda", dtype=torch.float16,
                            requires_grad=True)
        A = -torch.rand(c, n, device="cuda", dtype=torch.float32)
        B = torch.randn(b, n, length, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        C = torch.randn(b, n, length, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        D = torch.ones(c, device="cuda", dtype=torch.float32)
        y = SELECTIVE_SCAN_FN(u, delta, A, B, C, D, delta_softplus=True)
        y.float().mean().backward()
        _sync()
        ok = (y.shape == u.shape and torch.isfinite(y).all().item()
              and all(t.grad is not None and torch.isfinite(t.grad).all().item()
                      for t in (u, delta, B, C)))
        print("output:", tuple(y.shape), y.dtype, y.device)
        print("forward finite:", bool(torch.isfinite(y).all().item()))
        print("backward gradients finite:", ok)
        print("RESULT:", "PASS" if ok else "FAIL")
        return bool(ok)
    except Exception:
        traceback.print_exc()
        return False


def test_four_direction_ss2d() -> bool:
    print("\n========== FOUR-DIRECTION SS2D TEST ==========")
    if not torch.cuda.is_available() or not FUSED_SCAN_AVAILABLE:
        print("SKIP: requires CUDA and fused selective_scan.")
        return False
    try:
        # Small dimensions isolate the 4-path logic without allocating the full model.
        module = SS2D(dim=32, d_state=D_STATE).cuda().half().train()
        x = torch.randn(2, 8, 12, 32, device="cuda",
                        dtype=torch.float16, requires_grad=True)
        y = module(x)
        y.float().square().mean().backward()
        _sync()
        direction_grads = [
            any(p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in scan.parameters())
            for scan in module.scans
        ]
        ok = (y.shape == x.shape and torch.isfinite(y).all().item()
              and x.grad is not None and all(direction_grads))
        print("input/output:", tuple(x.shape), tuple(y.shape))
        print("four direction parameter gradients:", direction_grads)
        print("RESULT:", "PASS" if ok else "FAIL")
        return bool(ok)
    except Exception:
        traceback.print_exc()
        return False
    finally:
        _clear_cuda()


def test_full_model(batch_size: int = 1, image_size: int = IMAGE_SIZE,
                    amp: bool = True) -> dict:
    print(f"\n========== FULL MODEL TEST: B={batch_size}, {image_size}x{image_size} ==========")
    result = {"batch_size": batch_size, "image_size": image_size,
              "success": False, "peak_vram_gb": None, "seconds": None,
              "error": None}
    if not torch.cuda.is_available():
        result["error"] = "CUDA unavailable"
        print(result)
        return result
    _clear_cuda()
    model = None
    try:
        model = MyUnet().cuda().train()
        x = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        target = torch.rand(batch_size, 1, image_size, image_size, device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        _sync()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        _sync()
        result["seconds"] = round(time.perf_counter() - start, 3)
        result["peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 3
        )
        result["output_shape"] = list(logits.shape)
        result["loss"] = float(loss.detach().item())
        result["success"] = (
            logits.shape == target.shape and torch.isfinite(logits).all().item()
            and math.isfinite(result["loss"])
        )
        print("output:", result["output_shape"])
        print("loss:", result["loss"])
        print("time:", result["seconds"], "sec")
        print("peak allocated VRAM:", result["peak_vram_gb"], "GB")
        print("RESULT:", "PASS" if result["success"] else "FAIL")
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = "CUDA_OUT_OF_MEMORY: " + str(exc).splitlines()[0]
        print("OOM:", result["error"])
    except Exception as exc:
        result["error"] = repr(exc)
        traceback.print_exc()
    finally:
        del model
        _clear_cuda()
    return result


def batch_size_capacity_test(max_batch: int = BATCH_SIZE,
                             image_size: int = IMAGE_SIZE) -> list[dict]:
    """Try increasing batch sizes, stopping at first OOM/failure."""
    print("\n========== BATCH-SIZE CAPACITY TEST ==========")
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]
    candidates = [b for b in candidates if b <= max_batch]
    results = []
    for batch in candidates:
        item = test_full_model(batch, image_size=image_size, amp=True)
        results.append(item)
        if not item["success"]:
            print("Stopped after first failed batch size:", batch)
            break
    passed = [r["batch_size"] for r in results if r["success"]]
    print("Largest tested successful batch:", max(passed) if passed else None)
    print("NOTE: This is a one-step forward/backward capacity test; real training")
    print("may require additional memory for dataloader, optimizer and validation.")
    return results


# --------------------------- DATASET AND TRAINING ---------------------------

def make_transforms():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def build_datasets(data_root: str):
    import cv2
    import numpy as np
    from glob import glob
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset

    root = Path(data_root)
    fire_dir = root / "Image" / "Fire"
    nofire_dir = root / "Image" / "Not_Fire"
    mask_dir = root / "Segmentation_Mask" / "Fire"

    for required in (fire_dir, nofire_dir, mask_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {required}\n"
                "Expected Image/Fire, Image/Not_Fire and Segmentation_Mask/Fire."
            )

    image_paths, mask_paths = [], []
    for image_path in sorted(glob(str(fire_dir / "*"))):
        mask_path = mask_dir / Path(image_path).name
        if mask_path.exists():
            image_paths.append(image_path)
            mask_paths.append(str(mask_path))
    for image_path in sorted(glob(str(nofire_dir / "*"))):
        image_paths.append(image_path)
        mask_paths.append(None)

    if not image_paths:
        raise RuntimeError("No images found. Check --data-root.")
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=SEED,
        stratify=None
    )
    train_tf, val_tf = make_transforms()

    class FireDataset(Dataset):
        def __init__(self, images, masks, transform=None):
            self.images, self.masks, self.transform = images, masks, transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {self.images[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.masks[idx] is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Could not read mask: {self.masks[idx]}")
            mask = (mask > 127).astype(np.float32)
            if self.transform:
                sample = self.transform(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            return image, mask.unsqueeze(0).float()

    train_ds = FireDataset(train_imgs, train_masks, train_tf)
    val_ds = FireDataset(val_imgs, val_masks, val_tf)
    print("\n========== DATASET ==========")
    print("Total images:", len(image_paths))
    print("Train:", len(train_ds), "| Validation:", len(val_ds))
    print("Fire images with masks:", sum(m is not None for m in mask_paths))
    print("Non-fire images with empty masks:", sum(m is None for m in mask_paths))
    return train_ds, val_ds


def make_loader(dataset, shuffle: bool, batch_size: int, workers: int):
    from torch.utils.data import DataLoader
    kwargs = dict(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
        # Linux Kaggle/Colab supports fork. If running on Windows, omit this.
        if sys.platform.startswith("linux"):
            kwargs["multiprocessing_context"] = "fork"
    return DataLoader(**kwargs)


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    return (1.0 - ((2.0 * intersection + eps) / (denom + eps))).mean()


def batch_iou(logits, targets, eps: float = 1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersection = (pred * targets).sum()
    union = pred.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def train_model(data_root: str, epochs: int, batch_size: int,
                workers: int, save_path: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA in this script.")
    if not FUSED_SCAN_AVAILABLE:
        raise RuntimeError(
            "Fused Mamba selective_scan is unavailable. Refusing to start "
            "slow full-resolution training with the Python reference scan. "
            f"Import error: {FUSED_SCAN_IMPORT_ERROR}"
        )

    train_ds, val_ds = build_datasets(data_root)
    train_loader = make_loader(train_ds, True, batch_size, workers)
    val_loader = make_loader(val_ds, False, batch_size, workers)

    model = MyUnet().cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val = float("inf")

    print("\n========== TRAINING ==========")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} | Batch size: {batch_size}")
    print(f"Workers: {workers} | AMP: FP16 | Fused scan: {FUSED_SCAN_AVAILABLE}")
    print("Architecture:", MODEL_DIMS, MODEL_DEPTHS)
    print("Four directions: row+, row-, column+, column-")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_iou = 0.0
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for images, masks in train_loader:
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = (F.binary_cross_entropy_with_logits(logits, masks)
                        + dice_loss_from_logits(logits, masks))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.detach().item())
            tr_iou += float(batch_iou(logits.detach(), masks).item())
            del images, masks, logits, loss

        tr_loss /= max(1, len(train_loader))
        tr_iou /= max(1, len(train_loader))

        model.eval()
        va_loss = va_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = (F.binary_cross_entropy_with_logits(logits, masks)
                            + dice_loss_from_logits(logits, masks))
                va_loss += float(loss.item())
                va_iou += float(batch_iou(logits, masks).item())
                del images, masks, logits, loss

        va_loss /= max(1, len(val_loader))
        va_iou /= max(1, len(val_loader))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(va_iou)

        _sync()
        elapsed = time.perf_counter() - start
        peak = (torch.cuda.max_memory_allocated() / 1e9)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {tr_loss:.5f} IoU {tr_iou:.5f} | "
            f"val loss {va_loss:.5f} IoU {va_iou:.5f} | "
            f"{elapsed:.1f}s | peak VRAM {peak:.2f}GB"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "image_size": IMAGE_SIZE,
                "batch_size": batch_size,
                "model_dims": MODEL_DIMS,
                "model_depths": MODEL_DEPTHS,
                "fused_selective_scan": FUSED_SCAN_AVAILABLE,
            }, save_path)
            print("Saved best checkpoint:", save_path)

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        _clear_cuda()

    print("\nTraining finished. Best validation loss:", best_val)
    print("Checkpoint:", save_path)
    print("History:", HISTORY_PATH)


# --------------------------- TOP-LEVEL WORKFLOWS ---------------------------

def run_validation(max_batch: int = BATCH_SIZE, image_size: int = IMAGE_SIZE):
    env = environment_report()
    print("\n========== INSTALLED CUDA PACKAGES ==========")
    print("causal-conv1d CUDA extension:", CAUSAL_CONV_AVAILABLE)
    if CAUSAL_CONV_IMPORT_ERROR:
        print("causal-conv1d import error:", CAUSAL_CONV_IMPORT_ERROR)
    print("mamba selective_scan CUDA extension:", FUSED_SCAN_AVAILABLE)
    if FUSED_SCAN_IMPORT_ERROR:
        print("mamba import error:", FUSED_SCAN_IMPORT_ERROR)

    causal_ok = test_causal_conv1d()
    scan_ok = test_selective_scan()
    four_dir_ok = test_four_direction_ss2d()

    # Capacity testing is deliberately opt-in for the expensive full network.
    # --max-batch 1 is a quick end-to-end smoke test; 128 performs the full sweep.
    full_results = batch_size_capacity_test(max_batch, image_size)
    report = {
        "environment": env,
        "causal_conv1d": causal_ok,
        "selective_scan": scan_ok,
        "four_direction_ss2d": four_dir_ok,
        "full_model_batch_tests": full_results,
    }
    with open("vmamba_cuda_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nValidation report saved: vmamba_cuda_validation.json")
    return report


def run_benchmark(batch_size: int = BATCH_SIZE, repeats: int = 3):
    print("\n========== FULL MODEL BENCHMARK ==========")
    if not torch.cuda.is_available() or not FUSED_SCAN_AVAILABLE:
        raise RuntimeError("Benchmark needs CUDA and fused selective_scan.")
    model = MyUnet().cuda().train()
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
    target = torch.rand(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    timings = []
    _clear_cuda()
    for i in range(repeats + 1):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
        _sync()
        elapsed = time.perf_counter() - start
        if i > 0:
            timings.append(elapsed)
        print(f"iteration {i}: {elapsed:.3f} sec")
    print("mean step time:", round(sum(timings) / len(timings), 3), "sec")
    print("peak allocated VRAM:",
          round(torch.cuda.max_memory_allocated() / 1e9, 3), "GB")
    del model, x, target, optimizer
    _clear_cuda()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="VMamba FireSeg CUDA validation and training"
    )
    parser.add_argument("--mode", choices=["validate", "benchmark", "train"],
                        default="validate")
    parser.add_argument("--data-root", default="/content",
                        help="Folder containing Image/ and Segmentation_Mask/")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-batch", type=int, default=1,
                        help="Max batch for full-model validation. Set 128 for sweep.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--save-path", default=CHECKPOINT_PATH)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything()
    environment_report()

    if args.mode == "validate":
        run_validation(max_batch=args.max_batch, image_size=args.image_size)
    elif args.mode == "benchmark":
        run_benchmark(batch_size=args.batch_size,
                      repeats=args.benchmark_repeats)
    elif args.mode == "train":
        if args.image_size != IMAGE_SIZE:
            raise ValueError(
                f"This model setup is configured for {IMAGE_SIZE}x{IMAGE_SIZE}; "
                "change IMAGE_SIZE in the source only if intentionally changing it."
            )
        train_model(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    main()
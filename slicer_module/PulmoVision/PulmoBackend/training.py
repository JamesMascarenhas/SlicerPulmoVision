# PulmoBackend/training.py

import os
import math
import random
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from .unet3d import UNet3D


def create_synthetic_tumor_volume(
    shape: Tuple[int, int, int] = (64, 64, 64),
    num_tumors_range: Tuple[int, int] = (1, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple synthetic CT-like volume with 1–3 bright ellipsoidal 'tumors'
    inside a low-intensity 'lung' background.

    Returns:
        volume: float32 array, shape (H, W, D), range ~ [-1000, 500] HU (fake)
        mask:   uint8 array,  shape (H, W, D), {0, 1}
    """
    H, W, D = shape
    volume = np.random.normal(loc=-800.0, scale=50.0, size=shape).astype(np.float32)  # 'lung'
    mask = np.zeros(shape, dtype=np.uint8)

    num_tumors = random.randint(*num_tumors_range)

    for _ in range(num_tumors):
        # random center
        cz = random.randint(D // 4, 3 * D // 4)
        cy = random.randint(H // 4, 3 * H // 4)
        cx = random.randint(W // 4, 3 * W // 4)

        # random radii
        rz = random.randint(D // 12, D // 6)
        ry = random.randint(H // 12, H // 6)
        rx = random.randint(W // 12, W // 6)

        zz, yy, xx = np.ogrid[:D, :H, :W]
        ellipsoid = (
            ((zz - cz) ** 2) / (rz**2 + 1e-6)
            + ((yy - cy) ** 2) / (ry**2 + 1e-6)
            + ((xx - cx) ** 2) / (rx**2 + 1e-6)
        ) <= 1.0

        ellipsoid = np.transpose(ellipsoid, (1, 2, 0))  # to (H, W, D)
        mask = np.logical_or(mask, ellipsoid).astype(np.uint8)

    # "Tumor" intensities higher than background
    volume[mask == 1] = np.random.normal(loc=100.0, scale=50.0, size=np.sum(mask)).astype(
        np.float32
    )

    # Clip to at least look like HU
    volume = np.clip(volume, -1000.0, 500.0)

    return volume, mask


class SyntheticLungTumorDataset(Dataset):
    """
    On-the-fly synthetic dataset. Generates volume/mask pairs when indexed.
    """

    def __init__(self, n_samples: int = 100, shape=(64, 64, 64)):
        self.n_samples = n_samples
        self.shape = shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        volume, mask = create_synthetic_tumor_volume(self.shape)
        # Normalize volume to [0, 1]
        vol = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
        vol = vol.astype(np.float32)
        m = mask.astype(np.float32)

        # Convert to NCDHW later; here we return CHWD (C=1)
        vol = vol[None, ...]  # 1 x H x W x D
        m = m[None, ...]
        return vol, m


def dice_loss(pred, target, smooth: float = 1.0):
    """
    Dice loss for binary segmentation, pred and target are probabilities in [0, 1].
    Uses reshape instead of view to handle non-contiguous tensors.
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    denom = pred_flat.sum() + target_flat.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def train_unet3d(
    epochs: int = 5,
    batch_size: int = 2,
    n_samples: int = 100,
    lr: float = 1e-3,
    device: str = None,
    save_path: str = None,
):
    """
    Train UNet3D on synthetic data. This is meant to be quick and simple,
    just to produce a reasonable model for the pipeline demo.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if save_path is None:
        # Save inside PulmoBackend/checkpoints/
        base_dir = os.path.dirname(__file__)
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "unet3d_synthetic.pth")

    dataset = SyntheticLungTumorDataset(n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0

        for vol, m in dataloader:
            # vol, m are (B, 1, H, W, D); convert to NCDHW
            vol = vol.permute(0, 1, 4, 2, 3)  # (B,1,D,H,W)
            m = m.permute(0, 1, 4, 2, 3)

            vol = vol.to(device)
            m = m.to(device)

            optimizer.zero_grad()
            logits = model(vol)
            prob = torch.sigmoid(logits)

            loss_bce = bce(logits, m)
            loss_dice = dice_loss(prob, m)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(1, len(dataloader))
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved UNet3D weights to: {save_path}")


def generate_stub_checkpoint(save_path: Optional[str] = None) -> str:
    """
    Write a tiny placeholder checkpoint for offline environments.

    This keeps the expected file in place so the backend can gracefully
    fall back to percentile segmentation when trained weights are absent.
    """
    if save_path is None:
        base_dir = os.path.dirname(__file__)
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "unet3d_synthetic.pth")

    checkpoint = {
        "state_dict": {},
        "meta": {
            "description": "Placeholder checkpoint generated without training",
            "source": "generate_stub_checkpoint",
        },
    }

    with open(save_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Saved stub UNet3D checkpoint to: {save_path}")
    return save_path


if __name__ == "__main__":
    # Example quick run from a regular Python env
    train_unet3d(
        epochs=3,
        batch_size=2,
        n_samples=40,
    )


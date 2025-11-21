# PulmoBackend/training.py

import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from .unet3d import UNet3D
from .inference import get_default_unet3d_checkpoint_path
from .msd_lung_dataset import MSDTask06LungDataset, get_default_msd_root


JAMES_DATA_ROOT = os.path.abspath(
    "/Users/jamesmascarenhas/Desktop/courses/881/group_project/Task06_Lung"
)


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


def _resolve_data_root(data_root: Optional[str], training_on_james: bool = False) -> str:
    """Resolve where the MSD Task06 dataset should live.

    If ``training_on_james`` is True, the path is forced to James's local
    dataset location. Otherwise, we honor an explicit ``data_root`` or prompt
    the user when not provided.
    """

    if training_on_james:
        candidate = JAMES_DATA_ROOT
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"James-specific dataset path does not exist: {candidate}. Please verify the download."
        )

    if data_root is None:
        prompt = input(
            "Path to MSD Task06 Lung dataset (dataset.json, imagesTr, labelsTr): "
        ).strip()
        data_root = prompt or None

    if data_root is not None:
        candidate = os.path.abspath(data_root)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            "Provided data_root does not exist: "
            f"{candidate}. Set MSD_LUNG_DATA_ROOT or pass a valid path to --data-root."
        )
    
    candidate = get_default_msd_root()
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        "Please provide data_root or set MSD_LUNG_DATA_ROOT to point to the MSD Task06 data. "
        f"Expected default location: {candidate}"
    )


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
        save_path = get_default_unet3d_checkpoint_path()
        ckpt_dir = os.path.dirname(save_path)
        os.makedirs(ckpt_dir, exist_ok=True)

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

    checkpoint = {
        "state_dict": model.state_dict(),
        "base_channels": 16,
        "note": "Trained UNet3D on synthetic lung tumor dataset",
    }
    torch.save(checkpoint, save_path)
    print(f"Saved UNet3D weights to: {save_path}")
    return save_path


def train_msd_unet3d(
    data_root: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    lr: float = 1e-4,
    num_workers: int = 2,
    device: str = None,
    save_path: str = None,
    augment: bool = True,
    training_on_james: bool = False,
):
    """
    Train UNet3D on real MSD Task06 Lung data using random 3D patches.

    Args:
        data_root: Folder containing dataset.json, imagesTr, labelsTr, etc.
        epochs: Number of epochs to train.
        batch_size: Training batch size (number of patches).
        patch_size: Spatial size of extracted 3D patches.
        lr: Learning rate for Adam optimizer.
        num_workers: DataLoader workers.
        device: Override compute device; defaults to CUDA if available.
        save_path: Optional path to save checkpoint; defaults to inference path.
        augment: Whether to apply simple random flips.
        training_on_james: If True, force the dataset path to James's local copy.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_root = _resolve_data_root(data_root, training_on_james=training_on_james)
    if not os.path.exists(resolved_root):
        raise FileNotFoundError(f"MSD data root does not exist: {resolved_root}")

    if save_path is None:
        save_path = get_default_unet3d_checkpoint_path()
        ckpt_dir = os.path.dirname(save_path)
        os.makedirs(ckpt_dir, exist_ok=True)

    dataset = MSDTask06LungDataset(
        data_root=str(resolved_root),
        split="train",
        patch_size=patch_size,
        augment=augment,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            probs = torch.sigmoid(logits)

            loss_bce = bce(logits, masks)
            loss_dice = dice_loss(probs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(dataloader))
        print(f"[Epoch {epoch+1}/{epochs}] MSD train loss: {avg_loss:.4f}")

    checkpoint = {
        "state_dict": model.state_dict(),
        "base_channels": 16,
        "note": "Trained UNet3D on MSD Task06 Lung patches",
    }
    torch.save(checkpoint, save_path)
    print(f"Saved UNet3D weights to: {save_path}")
    return save_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Train UNet3D for lung segmentation")
    parser.add_argument("--train-msd", action="store_true", help="Train on MSD Task06 Lung instead of synthetic data")
    parser.add_argument("--data-root", type=str, default=None, help="Path to MSD Task06 Lung dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Patch size used for MSD training (D H W)",
    )
    parser.add_argument(
        "--on-james",
        action="store_true",
        help="Use James's hardcoded MSD dataset path instead of prompting",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-augment", action="store_true", help="Disable simple flipping augmentation")
    parser.add_argument("--output", type=str, default=None, help="Where to save the checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.train_msd:
        train_msd_unet3d(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=tuple(args.patch_size),
            lr=args.lr,
            num_workers=args.num_workers,
            save_path=args.output,
            augment=not args.no_augment,
            training_on_james=args.on_james,
        )
    else:
        train_unet3d(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.output,
        )
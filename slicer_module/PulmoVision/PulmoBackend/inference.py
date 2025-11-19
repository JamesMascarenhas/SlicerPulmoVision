"""
PulmoVision Backend - Inference

This module provides:
- Simple percentile-based placeholder segmentation.
- UNet3D-based segmentation using a trained model.

All functions operate on NumPy arrays, typically preprocessed volumes with
values in [0, 1].
"""

import os
from typing import Optional

import numpy as np
import torch

from .unet3d import UNet3D


# -------------------------------------------------------------------------
# Percentile-based segmentation (existing placeholder)
# -------------------------------------------------------------------------


def percentile_threshold_segmentation(volume, percentile=99.0):
    """
    Segment a volume by thresholding at a given intensity percentile.

    Intended as a placeholder for a real tumor segmentation model.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed volume, dtype float32, shape (H, W, D).
        Values are assumed to be in a finite numeric range (e.g., [0, 1]).
    percentile : float, optional
        Percentile (0–100) to use as the threshold. Voxels >= this value
        are labeled as 1, others as 0.

    Returns
    -------
    mask : np.ndarray
        Binary mask, same shape as input, dtype uint8 (0 or 1).
    """
    vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError("percentile_threshold_segmentation expects a 3D volume (H, W, D)")

    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100]")

    thresh = float(np.percentile(vol, percentile))
    mask = (vol >= thresh).astype(np.uint8)
    return mask


# -------------------------------------------------------------------------
# UNet3D-based segmentation
# -------------------------------------------------------------------------


def get_default_unet3d_checkpoint_path() -> str:
    """
    Default location of the UNet3D weights file.
    """
    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    return os.path.join(ckpt_dir, "unet3d_synthetic.pth")


def load_unet3d_model(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> UNet3D:
    """
    Load UNet3D with saved weights.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights_path is None:
        weights_path = get_default_unet3d_checkpoint_path()

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"UNet3D weights not found at {weights_path}. "
            f"Train the model first via PulmoBackend.training."
        )

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_unet3d_segmentation(
    volume: np.ndarray,
    *,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run UNet3D-based segmentation on a preprocessed CT volume.

    Args:
        volume: H x W x D float32 numpy array in [0, 1] or similar.
        weights_path: Optional path to .pth file. If None, uses default checkpoints path.
        device: 'cpu' or 'cuda'.
        threshold: probability threshold for binarizing the output.

    Returns:
        mask: H x W x D uint8 array with {0,1}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if volume.ndim != 3:
        raise ValueError(f"Expected volume of shape (H, W, D), got {volume.shape}")

    model = load_unet3d_model(weights_path=weights_path, device=device)

    # Normalize volume to [0, 1] if needed
    v = volume.astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)

    # Our convention in the backend: volume is H x W x D.
    # PyTorch expects N x C x D x H x W.
    v_dhw = np.transpose(v, (2, 0, 1))  # D,H,W
    v_tensor = torch.from_numpy(v_dhw)[None, None, ...]  # 1,1,D,H,W
    v_tensor = v_tensor.to(device)

    with torch.no_grad():
        logits = model(v_tensor)
        prob = torch.sigmoid(logits)

    prob_np = prob.cpu().numpy()[0, 0]  # D,H,W
    prob_hwd = np.transpose(prob_np, (1, 2, 0))  # back to H,W,D

    mask = (prob_hwd >= threshold).astype(np.uint8)
    return mask


# -------------------------------------------------------------------------
# Entry point (used by pipeline.py)
# -------------------------------------------------------------------------


def run_placeholder_segmentation(
    volume,
    method="percentile",
    **kwargs,
):
    """
    Entry point for segmentation.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed CT volume (typically output of preprocess_ct),
        expected shape (H, W, D), dtype float32.
    method : str, optional
        Segmentation method name. Supported:
        - "percentile": uses percentile_threshold_segmentation
          kwargs: percentile=...
        - "unet3d": uses run_unet3d_segmentation
          kwargs: weights_path=..., device=..., threshold=...

    Returns
    -------
    mask : np.ndarray
        Binary segmentation mask, same shape as input, dtype uint8.
    """
    method = method.lower()

    if method == "percentile":
        return percentile_threshold_segmentation(volume, **kwargs)
    elif method == "unet3d":
        return run_unet3d_segmentation(volume, **kwargs)
    else:
        raise ValueError(f"Unsupported segmentation method: {method!r}")

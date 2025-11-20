"""
PulmoVision Backend - Inference

This module provides:
- Simple percentile-based placeholder segmentation.
- UNet3D-based segmentation using a trained model.

All functions operate on NumPy arrays, typically preprocessed volumes with
values in [0, 1].
"""

import os
import warnings
from typing import Optional, Dict, Tuple

import numpy as np

# Optional PyTorch / UNet3D support
try:
    import torch  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    try:
        from .unet3d import UNet3D
    except Exception:
        UNet3D = None  # type: ignore[assignment]
else:
    UNet3D = None  # type: ignore[assignment]


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
    Default location of the UNet3D weights file (PyTorch .pt)
    """
    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    return os.path.join(ckpt_dir, "unet3d_synthetic.pt")


def ensure_default_unet3d_checkpoint(base_channels: int = 2) -> Tuple[str, bool]:
    """
    Make sure a lightweight synthetic checkpoint exists on disk.

    Returns the path and a boolean indicating whether the file was created.
    This keeps the default file extension aligned with future trained models
    while allowing the demo pipeline to run without bundled weights.
    """
    weights_path = get_default_unet3d_checkpoint_path()
    if os.path.exists(weights_path):
        return weights_path, False

    ckpt_dir = os.path.dirname(weights_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        torch.manual_seed(0)
        model = UNet3D(in_channels=1, out_channels=1, base_channels=base_channels)
        checkpoint = {
            "state_dict": model.state_dict(),
            "base_channels": base_channels,
            "note": "Synthetic placeholder weights for PulmoVision demo.",
        }
        torch.save(checkpoint, weights_path)
    except Exception as exc:  # noqa: BLE001 - propagate for status checks
        raise RuntimeError(f"Unable to create synthetic checkpoint at {weights_path}: {exc}") from exc

    return weights_path, True


def get_default_checkpoint_status(device: Optional[str] = None) -> Dict[str, object]:
    """
    Report whether the default checkpoint exists and is loadable.
    """
    try:
        default_path, _ = ensure_default_unet3d_checkpoint()
    except Exception as exc:  # noqa: BLE001 - propagate error details
        return {
            "path": get_default_unet3d_checkpoint_path(),
            "exists": False,
            "loads": False,
            "error": str(exc),
        }

    return get_checkpoint_status(default_path, device=device, prepare_default=False)


def load_unet3d_model(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    *,
    strict: bool = False,
    seed: int = 0,
    base_channels: Optional[int] = None,
    state_dict: Optional[Dict[str, object]] = None,
) -> Tuple[object, bool]:
    """
    Load UNet3D with saved weights.

    Returns the model and a boolean indicating whether any parameters
    were successfully loaded from the checkpoint.
    """
    if not _TORCH_AVAILABLE or UNet3D is None:
        raise RuntimeError(
            "UNet3D segmentation requires PyTorch, which is not available in this "
            "Slicer Python environment."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure we have a checkpoint path
    if weights_path is None:
        # This should return a valid path and a metadata dict
        weights_path, _ = ensure_default_unet3d_checkpoint(
            base_channels=base_channels or 16
        )

    base_channels_from_ckpt: Optional[int] = None

    # Load checkpoint if caller didn't pass a state_dict directly
    if state_dict is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"UNet3D weights not found at {weights_path}. "
                f"Train the model first via PulmoBackend.training."
            )

        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict):
            base_channels_from_ckpt = checkpoint.get("meta", {}).get(
                "base_channels", checkpoint.get("base_channels")
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            state_dict = checkpoint

    # If base_channels not explicitly given, try from checkpoint, else default 16
    if base_channels is None:
        base_channels = int(base_channels_from_ckpt) if base_channels_from_ckpt else 16

    torch.manual_seed(seed)
    model = UNet3D(in_channels=1, out_channels=1, base_channels=base_channels)

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise ValueError(
            f"Checkpoint at {weights_path} does not contain model parameters"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        warnings.warn(
            f"Missing parameters when loading UNet3D checkpoint: {missing}. "
            "Model will use default initialization for those layers.",
        )
    if unexpected:
        warnings.warn(
            f"Unexpected parameters in UNet3D checkpoint: {unexpected}. They were ignored."
        )

    model.to(device)
    model.eval()

    loaded_any = len(state_dict) > 0 and len(missing) < len(model.state_dict())
    return model, loaded_any


def get_checkpoint_status(
    weights_path: Optional[str],
    device: Optional[str] = None,
    *,
    prepare_default: bool = True,
) -> Dict[str, object]:
    """
    Evaluate the availability and loadability of a checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_path = weights_path or get_default_unet3d_checkpoint_path()
    if prepare_default and weights_path is None:
        try:
            resolved_path, _ = ensure_default_unet3d_checkpoint()
        except Exception as exc:  # noqa: BLE001 - surface creation error
            return {
                "path": resolved_path,
                "exists": False,
                "loads": False,
                "error": str(exc),
            }
    status: Dict[str, object] = {
        "path": resolved_path,
        "exists": os.path.exists(resolved_path),
        "loads": False,
        "error": None,
    }

    if not status["exists"]:
        status["error"] = "Checkpoint file is missing"
        return status

    try:
        _, loaded_any = load_unet3d_model(
            weights_path=resolved_path,
            device=device,
            strict=False,
            seed=0,
        )
    except Exception as exc:  # noqa: BLE001 - propagate root cause in status
        status["error"] = str(exc)
        return status

    status["loads"] = bool(loaded_any)
    return status


def run_unet3d_segmentation(
    volume: np.ndarray,
    *,
    weights_path: Optional[str] = None,
    model: Optional[object] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
    seed: int = 0,
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
    if volume.ndim != 3:
        raise ValueError(f"Expected volume of shape (H, W, D), got {volume.shape}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model, _ = load_unet3d_model(
            weights_path=weights_path,
            device=device,
            seed=seed,
            strict=False,
        )

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
    *,
    return_metadata: bool = False,
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
        metadata : dict, optional
        When return_metadata=True, a dictionary describing which segmentation
        strategy was used and any fallback messages.
    """
    method = (method or "").lower().strip() or "auto"
    percentile = float(kwargs.pop("percentile", 99.0))
    device = kwargs.get("device")
    seed = kwargs.get("seed", 0)
    metadata: Dict[str, object] = {
        "requested_method": method,
        "used_method": None,
        "weights_path": None,
        "checkpoint_exists": False,
        "checkpoint_loaded": False,
        "messages": [],
    }

    if method == "percentile":
        metadata["used_method"] = "percentile"
        mask = percentile_threshold_segmentation(volume, percentile=percentile)
        return (mask, metadata) if return_metadata else mask

    if method in {"unet3d", "auto"}:
        weights_path = kwargs.get("weights_path") or None
        if weights_path is None:
            try:
                weights_path, _ = ensure_default_unet3d_checkpoint()
            except Exception as exc:  # noqa: BLE001 - surface creation issues
                metadata["weights_path"] = get_default_unet3d_checkpoint_path()
                metadata["messages"].append(str(exc))
                weights_path = metadata["weights_path"]
            else:
                metadata["weights_path"] = weights_path
        else:
            metadata["weights_path"] = weights_path

        model = None
        try:
            model, loaded_any = load_unet3d_model(
                weights_path=weights_path,
                device=device,
                seed=seed,
                strict=False,
            )
            metadata["checkpoint_exists"] = True
            metadata["checkpoint_loaded"] = bool(loaded_any)
        except FileNotFoundError as exc:
            metadata["checkpoint_exists"] = False
            metadata["messages"].append(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface load errors
            metadata["checkpoint_exists"] = True
            metadata["messages"].append(str(exc))
        else:
            try:
                mask = run_unet3d_segmentation(
                    volume,
                    model=model,
                    device=device,
                    threshold=float(kwargs.get("threshold", 0.5)),
                    seed=seed,
                )
                metadata["used_method"] = "unet3d"
                return (mask, metadata) if return_metadata else mask
            except Exception as exc:  # noqa: BLE001 - signal fallback cause
                metadata["messages"].append(
                    f"UNet3D segmentation unavailable ({exc}). Falling back to percentile heuristic."
                )

        if method == "unet3d" and metadata["used_method"] != "unet3d":
            metadata["messages"].append(
                "UNet3D was requested but usable weights were not found; using percentile segmentation instead."
            )

        metadata["used_method"] = "percentile"
        mask = percentile_threshold_segmentation(volume, percentile=percentile)
        return (mask, metadata) if return_metadata else mask

    raise ValueError(f"Unsupported segmentation method: {method!r}")

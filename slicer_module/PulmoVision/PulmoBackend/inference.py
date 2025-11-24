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
from typing import Optional, Dict, Tuple, Any
from .msd_lung_dataset import _normalize_intensity, DEFAULT_CLIP_RANGE


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
# HU threshold segmentation (classical baseline)
# -------------------------------------------------------------------------


def hu_threshold_segmentation(volume, threshold_hu: float = -300.0) -> np.ndarray:
    """
    Heuristic HU-based lung tumor segmentation.

    Strategy:
      1. Seed lung air as very low HU (HU < -700).
      2. Dilate a few times to approximate full lung parenchyma.
      3. Inside this lung ROI, mark voxels denser than lung parenchyma
         (HU > -150) as tumor candidates.
      4. Keep only the largest connected component (6-connected).

    Note: `threshold_hu` is kept in the signature for compatibility with
    callers, but the internal thresholds are tuned for MSD-style lung CT.

    Parameters
    ----------
    volume : np.ndarray
        Raw CT volume in Hounsfield Units, shape (H, W, D).
    threshold_hu : float, optional
        Unused in this heuristic (kept for call compatibility).

    Returns
    -------
    mask : np.ndarray
        Binary mask, same shape as input, dtype uint8 (0 or 1).
    """
    import numpy as _np

    vol = _np.asarray(volume, dtype=_np.float32)
    if vol.ndim != 3:
        raise ValueError("hu_threshold_segmentation expects a 3D volume (H, W, D)")

    H, W, D = vol.shape

    # ------------------------------------------------------------------
    # 1) Lung air seed: clearly air-like voxels
    # ------------------------------------------------------------------
    lung_seed = vol < -700.0  # stricter than -300 to avoid grabbing the whole body

    # ------------------------------------------------------------------
    # 2) Dilate a bit to approximate full lungs
    # ------------------------------------------------------------------
    lung_mask = lung_seed.copy()
    for _ in range(3):  # fewer iterations: keep ROI near lungs
        padded = _np.pad(lung_mask, 1, mode="constant", constant_values=False)
        expanded = _np.zeros_like(lung_mask, dtype=bool)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    expanded |= padded[
                        1 + dx : 1 + dx + H,
                        1 + dy : 1 + dy + W,
                        1 + dz : 1 + dz + D,
                    ]
        lung_mask = expanded

    # ------------------------------------------------------------------
    # 3) Solid-ish candidates inside lung ROI
    # ------------------------------------------------------------------
    solid_candidates = vol > -150.0  # denser than parenchyma, but includes tumor & vessels
    candidates = lung_mask & solid_candidates

    # Debug summary
    print(
        "DEBUG HU (v2):",
        "lung_seed_voxels =", int(lung_seed.sum()),
        "lung_mask_voxels =", int(lung_mask.sum()),
        "candidate_voxels =", int(candidates.sum()),
    )

    if not _np.any(candidates):
        return _np.zeros_like(vol, dtype=_np.uint8)

    # ------------------------------------------------------------------
    # 4) Largest connected component (6-connected) in candidates
    # ------------------------------------------------------------------
    visited = _np.zeros_like(candidates, dtype=bool)
    largest_component = _np.zeros_like(candidates, dtype=bool)
    max_size = 0

    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for x in range(H):
        for y in range(W):
            for z in range(D):
                if candidates[x, y, z] and not visited[x, y, z]:
                    stack = [(x, y, z)]
                    visited[x, y, z] = True
                    component_voxels = [(x, y, z)]

                    while stack:
                        cx, cy, cz = stack.pop()
                        for dx, dy, dz in neighbors:
                            nx, ny, nz = cx + dx, cy + dy, cz + dz
                            if (
                                0 <= nx < H
                                and 0 <= ny < W
                                and 0 <= nz < D
                                and candidates[nx, ny, nz]
                                and not visited[nx, ny, nz]
                            ):
                                visited[nx, ny, nz] = True
                                stack.append((nx, ny, nz))
                                component_voxels.append((nx, ny, nz))

                    comp_size = len(component_voxels)
                    if comp_size > max_size:
                        max_size = comp_size
                        largest_component.fill(False)
                        for vx, vy, vz in component_voxels:
                            largest_component[vx, vy, vz] = True

    return largest_component.astype(_np.uint8)



# -------------------------------------------------------------------------
# UNet3D-based segmentation
# -------------------------------------------------------------------------


def get_default_unet3d_checkpoint_path() -> str:
    """
    Default location of the UNet3D weights file (PyTorch .pt)
    for the sythetic placeholder model used in initial demos
    """
    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    return os.path.join(ckpt_dir, "unet3d_synthetic.pt")


def get_default_msd_unet3d_checkpoint_path() -> str:
    """Default location for UNet3D weights trained on MSD Task06 Lung."""
    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    return os.path.join(ckpt_dir, "unet3d_msd.pt")


def _checkpoint_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    base_channels = meta.get("base_channels") or state.get("base_channels")
    return {
        "schema_version": meta.get("schema_version", 1),
        "trained_on": meta.get("trained_on", "unknown"),
        "checkpoint_type": meta.get("checkpoint_type", "unknown"),
        "package_version": meta.get("package_version"),
        "git_sha": meta.get("git_sha"),
        "base_channels": base_channels,
        "notes": meta.get("notes"),
    }


def ensure_default_unet3d_checkpoint(base_channels: int = 2) -> Tuple[str, bool]:
    """
    Preserve the legacy synthetic checkpoint for explicit demo use.

    Default segmentation now expects a real MSD checkpoint; this helper is kept
    for tests or controlled fallbacks.
    """
    weights_path = get_default_unet3d_checkpoint_path()
    if os.path.exists(weights_path):
        return weights_path, False
    
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "UNet3D requires PyTorch, which is not installed in this Slicer environment."
            "To enable UNet3D segmentation, install the official 'PyTorch' extension from:"
            "  Slicer → View → Extensions Manager → Search: PyTorch → Install"
            "Then restart Slicer and try again."
        )   

    ckpt_dir = os.path.dirname(weights_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        torch.manual_seed(0)
        model = UNet3D(in_channels=1, out_channels=1, base_channels=base_channels)
        checkpoint = {
            "state_dict": model.state_dict(),
            "meta": {
                "schema_version": 1,
                "trained_on": "synthetic demo",
                "checkpoint_type": "synthetic-demo",
                "base_channels": base_channels,
                "notes": "Synthetic placeholder weights for PulmoVision demo.",
            },
        }
        torch.save(checkpoint, weights_path)
    except Exception as exc:  # noqa: BLE001 - propagate for status checks
        raise RuntimeError(f"Unable to create synthetic checkpoint at {weights_path}: {exc}") from exc

    return weights_path, True


def get_default_checkpoint_status(device: Optional[str] = None) -> Dict[str, object]:
    """
    Report whether a default checkpoint exists and is loadable.

    Priority:
      1) MSD-trained checkpoint (unet3d_msd.pt)
      2) Synthetic fallback checkpoint (unet3d_synthetic.pt)
    """
    if device is None:
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    msd_path = get_default_msd_unet3d_checkpoint_path()
    synthetic_path = get_default_unet3d_checkpoint_path()

    status: Dict[str, object] = {
        "path": None,
        "exists": False,
        "loads": False,
        "error": None,
        "is_msd": False,
        "is_synthetic": False,
        "metadata": None,
    }

    # --- 1) Prefer MSD checkpoint -----------------------------------------
    if os.path.exists(msd_path):
        status["path"] = msd_path
        status["exists"] = True
        status["is_msd"] = True

        try:
            _, loaded_any = load_unet3d_model(
                weights_path=msd_path,
                device=device,
                strict=False,
            )
            status["loads"] = bool(loaded_any)
            checkpoint = torch.load(msd_path, map_location=device)
            status["metadata"] = _checkpoint_metadata(checkpoint)
        except Exception as exc:  # noqa: BLE001 - surface load error
            status["error"] = str(exc)

        return status
    

    synthetic_path = get_default_unet3d_checkpoint_path()
    if os.path.exists(synthetic_path):
        status["path"] = synthetic_path
        status["exists"] = True
        status["is_synthetic"] = True
        try:
            checkpoint = torch.load(synthetic_path, map_location=device)
            _, loaded_any = load_unet3d_model(
                weights_path=synthetic_path,
                device=device,
                strict=False,
            )
            status["loads"] = bool(loaded_any)
            status["metadata"] = _checkpoint_metadata(checkpoint)
        except Exception as exc:  # noqa: BLE001 - surface load error
            status["error"] = str(exc)
    else:
        status["error"] = "MSD checkpoint missing; synthetic fallback not prepared"

    return status


def load_unet3d_model(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    *,
    strict: bool = False,
    seed: int = 0,
    base_channels: Optional[int] = None,
    state_dict: Optional[Dict[str, object]] = None,
    allow_synthetic_fallback: bool = False,
) -> Tuple[object, bool]:
    """
    Load UNet3D with saved weights.

    Returns the model and a boolean indicating whether any parameters
    were successfully loaded from the checkpoint.
    """
    if not _TORCH_AVAILABLE or UNet3D is None:
        raise RuntimeError(
            "UNet3D segmentation requires PyTorch. It is not currently available."
            "Install the official 'PyTorch' extension via the Slicer Extensions Manager:"
            "  View → Extensions Manager → PyTorch → Install"
            "Restart Slicer afterward."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure we have a checkpoint path. Default now requires MSD weights and will
    # raise a clear error if they are missing.
    if weights_path is None:
        msd_default = get_default_msd_unet3d_checkpoint_path()
        if os.path.exists(msd_default):
            weights_path = msd_default
        elif allow_synthetic_fallback:
            weights_path, _ = ensure_default_unet3d_checkpoint(
                base_channels=base_channels or 16
            )
        else:
            raise FileNotFoundError(
                f"UNet3D weights not found at default path {msd_default}. "
                "Download Task06_Lung via data/prepare_msd_lung.py and train PulmoBackend.training."
            )

    base_channels_from_ckpt: Optional[int] = None

    meta_info: Dict[str, Any] = {}

    # Load checkpoint if caller didn't pass a state_dict directly
    if state_dict is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"UNet3D weights not found at {weights_path}. "
                f"Train the model first via PulmoBackend.training."
            )

        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict):
            meta_info = _checkpoint_metadata(checkpoint)
            base_channels_from_ckpt = meta_info.get("base_channels")
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
    if meta_info:
        model._pulmovision_meta = meta_info  # type: ignore[attr-defined]

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

    if weights_path is None:
        msd_default = get_default_msd_unet3d_checkpoint_path()
        if os.path.exists(msd_default):
            resolved_path = msd_default
        else:
            resolved_path = get_default_unet3d_checkpoint_path()
            if prepare_default:
                try:
                    resolved_path, _ = ensure_default_unet3d_checkpoint()
                except Exception as exc:  # noqa: BLE001 - surface creation error
                    return {
                        "path": resolved_path,
                        "exists": False,
                        "loads": False,
                        "error": str(exc),
                    }
    else:
        resolved_path = weights_path
    status: Dict[str, object] = {
        "path": resolved_path,
        "exists": os.path.exists(resolved_path),
        "loads": False,
        "error": None,
        "metadata": None,
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
            allow_synthetic_fallback=True,
        )
    except Exception as exc:  # noqa: BLE001 - propagate root cause in status
        status["error"] = str(exc)
        return status

    status["loads"] = bool(loaded_any)
    try:
        checkpoint = torch.load(resolved_path, map_location=device)
        status["metadata"] = _checkpoint_metadata(checkpoint)
    except Exception:
        status["metadata"] = None
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
    Run UNet3D-based segmentation on a CT volume.

    Args:
        volume: H x W x D float32 NumPy array in Hounsfield Units.
                It will be clipped to DEFAULT_CLIP_RANGE and normalized
                to [0, 1] using the same logic as the MSD dataset loader.
        weights_path: Optional path to .pt file. If None, uses default checkpoints path.
        device: 'cpu' or 'cuda'.
        threshold: probability threshold for binarizing the output.

    Returns:
        mask: H x W x D uint8 array with {0,1}.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume of shape (H, W, D), got {volume.shape}")

    if device is None:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    if model is None:
        model, _ = load_unet3d_model(
            weights_path=weights_path,
            device=device,
            seed=seed,
            strict=False,
        )

    # Match MSD training preprocessing
    v = np.asarray(volume, dtype=np.float32)
    v = _normalize_intensity(v, DEFAULT_CLIP_RANGE)   # [0,1] after HU clip

    # Backend convention: volume is H x W x D.
    # PyTorch expects N x C x D x H x W.
    v_dhw = np.transpose(v, (2, 0, 1))  # D,H,W
    v_tensor = torch.tensor(v_dhw.tolist(), dtype=torch.float32)[None, None, ...]  # 1,1,D,H,W
    v_tensor = v_tensor.to(device)

    with torch.no_grad():
        logits = model(v_tensor)
        prob = torch.sigmoid(logits)

    prob_np = np.array(prob.cpu().tolist(), dtype=np.float32)[0, 0]  # D,H,W
    prob_hwd = np.transpose(prob_np, (1, 2, 0))  # H,W,D

    print(
        "UNet3D Debug — Probability stats:",
        "min=", float(prob_np.min()),
        "max=", float(prob_np.max()),
        "mean=", float(prob_np.mean()),
    )

    mask = (prob_hwd >= threshold).astype(np.uint8)
    return mask


# -------------------------------------------------------------------------
# Entry point (used by pipeline.py)
# -------------------------------------------------------------------------


def run_placeholder_segmentation(
    volume,
    method="unet3d",
    *,
    return_metadata: bool = False,
    allow_fallback_to_percentile: bool = False,
    allow_hu_threshold_fallback: bool = False,
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
    method = (method or "").lower().strip() or "unet3d"
    percentile = float(kwargs.pop("percentile", 99.0))
    threshold_hu = float(kwargs.get("threshold_hu", -300.0))
    device = kwargs.get("device")
    seed = kwargs.get("seed", 0)
    metadata: Dict[str, object] = {
        "requested_method": method,
        "used_method": None,
        "weights_path": None,
        "checkpoint_exists": False,
        "checkpoint_loaded": False,
        "messages": [],
        "checkpoint_metadata": None,
    }

    if method == "percentile":
        metadata["used_method"] = "percentile"
        mask = percentile_threshold_segmentation(volume, percentile=percentile)
        return (mask, metadata) if return_metadata else mask
    
    torch_ready = _TORCH_AVAILABLE and UNet3D is not None

    if method == "unet3d":
        if not torch_ready:
            metadata["messages"].append(
                "UNet3D segmentation unavailable: PyTorch is not installed."
                "Install the official 'PyTorch' extension using Slicer's Extensions Manager:"
                "  View → Extensions Manager → Search 'PyTorch' → Install"
                "After installing and restarting Slicer, UNet3D segmentation will be enabled"
            )
            metadata["used_method"] = "percentile"
            mask = percentile_threshold_segmentation(volume, percentile=percentile)
            return (mask, metadata) if return_metadata else mask
        
        weights_path = kwargs.get("weights_path") or None
        if weights_path is None:
            weights_path = get_default_msd_unet3d_checkpoint_path()
        metadata["weights_path"] = weights_path

        model = None
        try:
            model, loaded_any = load_unet3d_model(
                weights_path=weights_path,
                device=device,
                seed=seed,
                strict=False,
                allow_synthetic_fallback=allow_fallback_to_percentile,
            )
            metadata["checkpoint_exists"] = True
            metadata["checkpoint_loaded"] = bool(loaded_any)
            if hasattr(model, "_pulmovision_meta"):
                metadata["checkpoint_metadata"] = getattr(model, "_pulmovision_meta")
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
        if allow_hu_threshold_fallback:
            metadata["messages"].append(
                "UNet3D was requested but usable weights were not found; using HU Threshold segmentation instead."
            )
            metadata["used_method"] = "hu_threshold"
            try:
                mask = hu_threshold_segmentation(volume, threshold_hu=threshold_hu)
                return (mask, metadata) if return_metadata else mask
            except Exception as exc:  # noqa: BLE001 - surface HU errors
                metadata["messages"].append(
                    f"HU threshold segmentation failed ({exc})."
                )
                # If HU fallback failed and percentile fallback is allowed, drop through
                # to the percentile handler below. Otherwise, we'll raise with the
                # accumulated messages.

        if allow_fallback_to_percentile:
            if method == "unet3d" and metadata["used_method"] != "unet3d":
                metadata["messages"].append(
                    "UNet3D was requested but usable weights were not found; using percentile segmentation instead."
                )

            metadata["used_method"] = "percentile"
            mask = percentile_threshold_segmentation(volume, percentile=percentile)
            return (mask, metadata) if return_metadata else mask

        messages = "; ".join(metadata.get("messages", [])) or "UNet3D checkpoint unavailable"
        raise RuntimeError(messages)

    raise ValueError(f"Unsupported segmentation method: {method!r}")
"""
PulmoVision Backend - End-to-End Pipeline

This module ties together preprocessing, segmentation, and postprocessing
into a single callable function `run_pulmo_pipeline`, which will be invoked
from the 3D Slicer-facing PulmoVision module.

Design goals:
- Pure NumPy operations here (no Slicer or SimpleITK imports).
- Clean API that can later swap in a 3D U-Net model without changing
  the Slicer-facing logic.
"""

import numpy as np

from .preprocessing import preprocess_ct
from .inference import run_placeholder_segmentation
from .postprocessing import postprocess_mask


def run_pulmo_pipeline(volume,
                       *,
                       # reasonable lung defaults; can be overridden from Slicer
                       window_center=-600.0,
                       window_width=1500.0,
                       normalize=True,
                       norm_out_min=0.0,
                       norm_out_max=1.0,
                       segmentation_method="auto",
                       segmentation_kwargs=None,
                       postprocess=True,
                       postprocess_kwargs=None,
                       return_intermediates=False):
    """
    Run the PulmoVision end-to-end pipeline on a CT volume.

    Current stages:
    1. Preprocessing:
       - CT windowing (center/width).
       - Min–max normalization to [norm_out_min, norm_out_max].
    2. Segmentation:
       - Placeholder segmentation based on simple heuristics.
    3. Postprocessing:
       - Ensure clean binary mask.

    Parameters
    ----------
    volume : np.ndarray
        Raw CT volume, expected shape (H, W, D). Values are typically HU,
        but any numeric range is accepted.
    window_center : float, optional
        CT window center (level). Default is -600 (lung window).
    window_width : float, optional
        CT window width. Default is 1500 (lung window).
    normalize : bool, optional
        If True, apply min–max normalization after windowing.
    norm_out_min : float, optional
        Lower bound of normalization range (default 0.0).
    norm_out_max : float, optional
        Upper bound of normalization range (default 1.0).
    segmentation_method : str, optional
        Segmentation method name for the backend. Currently:
        - "percentile": use run_placeholder_segmentation with percentile threshold.
        - "unet3d": attempt UNet3D inference; fall back to percentile if weights missing.
        - "auto": try UNet3D if usable weights are available, otherwise percentile.
    segmentation_kwargs : dict, optional
        Extra keyword arguments forwarded to the segmentation function.
        For method "percentile", you can pass {"percentile": 99.0}, etc.
        For method "unet3d"/"auto", pass {"weights_path": ..., "device": ..., "threshold": ...}.
    postprocess : bool, optional
        If True, apply postprocess_mask to the raw mask.
    postprocess_kwargs : dict, optional
        Extra keyword arguments forwarded to postprocess_mask.
    return_intermediates : bool, optional
        If True, return a dictionary including the preprocessed volume
        and the mask. If False, return only the mask.

    Returns
    -------
    If return_intermediates is False:
        mask : np.ndarray
            Binary segmentation mask, same shape as input, dtype uint8.
    If return_intermediates is True:
        outputs : dict
            {
                "preprocessed_volume": np.ndarray (float32),
                "raw_mask": np.ndarray (uint8),
                "final_mask": np.ndarray (uint8),
            }
    """
    vol = np.asarray(volume)

    if vol.ndim != 3:
        raise ValueError("run_pulmo_pipeline expects a 3D volume (H, W, D)")

    # 1. Preprocessing
    preprocessed = preprocess_ct(
        vol,
        window_center=window_center,
        window_width=window_width,
        normalize=normalize,
        norm_out_min=norm_out_min,
        norm_out_max=norm_out_max,
    )

    # 2. Segmentation (placeholder, to be replaced by 3D U-Net later)
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    else:
        segmentation_kwargs = dict(segmentation_kwargs)
    raw_mask = run_placeholder_segmentation(
        preprocessed,
        method=segmentation_method,
        **segmentation_kwargs,
    )

    # 3. Postprocessing
    if postprocess:
        if postprocess_kwargs is None:
            postprocess_kwargs = {}
        final_mask = postprocess_mask(raw_mask, **postprocess_kwargs)
    else:
        final_mask = raw_mask

    if return_intermediates:
        return {
            "preprocessed_volume": preprocessed,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
        }
    else:
        return final_mask


# This file lives in: slicer_module/PulmoVision/Testing/Python
import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels: .../PulmoVision
MODULE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Allow "from PulmoBackend..." imports
sys.path.insert(0, MODULE_DIR)

from PulmoBackend.inference import get_default_unet3d_checkpoint_path
from PulmoBackend.pipeline import run_pulmo_pipeline


DEF_CKPT = get_default_unet3d_checkpoint_path()


def make_deterministic_volume():
    """A simple synthetic volume with a bright central cube."""
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    vol[2:6, 2:6, 2:6] = 100.0
    vol[0, 0, 0] = -1000.0
    return vol


def run_and_check(volume, *, method, kwargs):
    mask = run_pulmo_pipeline(
        volume,
        segmentation_method=method,
        segmentation_kwargs=kwargs,
        postprocess=False,
        normalize=True,
    )

    assert mask.shape == volume.shape
    assert mask.dtype == np.uint8
    unique_vals = np.unique(mask)
    assert np.all(np.isin(unique_vals, [0, 1]))
    assert mask.sum() > 0
    return mask


def main():
    volume = make_deterministic_volume()

    percentile_mask = run_and_check(
        volume,
        method="percentile",
        kwargs={"percentile": 99.0},
    )
    assert percentile_mask.sum() == 64, "Percentile heuristic mask size changed"

    unet_kwargs = {
        "weights_path": DEF_CKPT,
        "device": "cpu",
        "threshold": 0.5,
        "percentile": 99.0,  # fallback path uses this
    }
    unet_mask = run_and_check(volume, method="unet3d", kwargs=unet_kwargs)

    print("Percentile foreground voxels:", int(percentile_mask.sum()))
    print("UNet3D foreground voxels:", int(unet_mask.sum()))
    print("Masks equal:", bool(np.array_equal(percentile_mask, unet_mask)))


if __name__ == "__main__":
    main()

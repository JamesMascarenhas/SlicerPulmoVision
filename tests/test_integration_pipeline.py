import os
import pytest
import numpy as np

try:
    import torch
except Exception:
    torch = None

from PulmoBackend.inference import get_default_msd_unet3d_checkpoint_path
from PulmoBackend.pipeline import run_pulmo_pipeline
from tests.conftest import make_demo_volume


@pytest.mark.skipif(torch is None, reason="PyTorch required for UNet3D smoke test")
def test_pipeline_runs_with_msd_checkpoint():
    vol = make_demo_volume()

    result = run_pulmo_pipeline(
        vol,
        segmentation_method="unet3d",
        segmentation_kwargs={
            "weights_path": get_default_msd_unet3d_checkpoint_path(),
            "allow_hu_threshold_fallback": False,
            "threshold": 0.5,
        },
        postprocess=False,
        normalize=True,
        return_metadata=True,
    )

    mask = result["mask"]
    meta = result["segmentation_metadata"]

    assert mask.shape == vol.shape
    assert mask.dtype == np.uint8
    assert meta["used_method"] == "unet3d"
    assert meta["checkpoint_metadata"]["trained_on"]

def test_pipeline_falls_back_when_checkpoint_missing():
    vol = make_demo_volume()

    missing_ckpt = os.path.join(
        "slicer_module", "PulmoVision", "PulmoBackend", "checkpoints", "missing_file.pt"
    )

    result = run_pulmo_pipeline(
        vol,
        segmentation_method="unet3d",
        segmentation_kwargs={"weights_path": missing_ckpt},
        return_metadata=True,
    )

    mask = result["mask"]
    meta = result["segmentation_metadata"]

    assert mask.shape == vol.shape
    assert mask.dtype == np.uint8
    assert meta["used_method"] == "hu_threshold"
    assert any("not found" in msg.lower() for msg in meta["messages"])
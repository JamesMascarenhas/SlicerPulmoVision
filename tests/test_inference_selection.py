import os
import pytest

try:
    import torch
except Exception:
    torch = None

from PulmoBackend import inference
from conftest import make_demo_volume


@pytest.mark.skipif(torch is None, reason="PyTorch required for UNet3D")
def test_missing_checkpoint_raises_without_fallback():
    missing = os.path.join(
        "slicer_module", "PulmoVision", "PulmoBackend",
        "checkpoints", "does_not_exist.pt"
    )
    vol = make_demo_volume()

    with pytest.raises(RuntimeError):
        inference.run_placeholder_segmentation(
            vol,
            method="unet3d",
            weights_path=missing,
            allow_fallback_to_percentile=False,
        )


@pytest.mark.skipif(torch is None, reason="PyTorch required")
def test_status_reports_metadata_for_msd_checkpoint():
    status = inference.get_default_checkpoint_status(device="cpu")

    assert status["path"].endswith("unet3d_msd.pt")
    assert status["exists"] is True
    assert status["metadata"] is not None
    assert status["metadata"].get("trained_on")

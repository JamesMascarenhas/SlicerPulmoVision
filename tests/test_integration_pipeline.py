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
            "allow_fallback_to_percentile": False,
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

import numpy as np

from PulmoBackend.radiomics import compute_radiomics_features


def test_basic_radiomics_outputs_keys():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[1:3, 1:3, 1:3] = 5.0

    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1

    features = compute_radiomics_features(vol, mask, voxel_spacing=(1.0, 1.0, 1.0))
    assert "Volume (mm^3)" in features
    assert "Surface Area (mm^2)" in features
    assert features["Volume (mm^3)"] > 0
    assert features["Surface Area (mm^2)"] > 0

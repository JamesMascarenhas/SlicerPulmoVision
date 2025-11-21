import numpy as np

from PulmoBackend.preprocessing import (
    normalize_minmax,
    preprocess_ct,
    window_ct,
)


def test_window_ct_limits_values():
    vol = np.array([[[-1000, -200], [0, 1200]]], dtype=np.float32)
    windowed = window_ct(vol, window_center=-600, window_width=1500)
    assert windowed.min() >= -1350
    assert windowed.max() <= 150


def test_preprocess_ct_normalizes_to_unit_interval():
    vol = np.linspace(-1000, 500, num=8).reshape((2, 2, 2)).astype(np.float32)
    processed = preprocess_ct(vol, normalize=True)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_normalize_minmax_with_custom_range():
    vol = np.array([[[0.0, 5.0, 10.0]]], dtype=np.float32)
    scaled = normalize_minmax(vol, 0.0, 10.0, -1.0, 1.0)
    assert np.isclose(scaled.min(), -1.0)
    assert np.isclose(scaled.max(), 1.0)

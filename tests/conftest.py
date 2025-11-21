import sys
import os
import numpy as np

# Make slicer_module importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SLICER_MODULE = os.path.join(REPO_ROOT, "slicer_module")

if SLICER_MODULE not in sys.path:
    sys.path.insert(0, SLICER_MODULE)

# Allow ``import PulmoBackend`` without relying on PYTHONPATH
PULMOVISION_ROOT = os.path.join(SLICER_MODULE, "PulmoVision")
if PULMOVISION_ROOT not in sys.path:
    sys.path.insert(0, PULMOVISION_ROOT)


def make_demo_volume():
    """Tiny volume for smoke/inference tests."""
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    vol[2:6, 2:6, 2:6] = 100.0
    vol[0, 0, 0] = -1000.0
    return vol
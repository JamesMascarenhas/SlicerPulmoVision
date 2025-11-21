"""Patch-based MSD Task06 Lung dataset loader for PyTorch."""

import json
import os
import random
from typing import Optional, Tuple

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CLIP_RANGE = (-1000.0, 400.0)


def get_default_msd_root() -> str:
    """Return a best-effort default path for the MSD dataset.

    The path can be overridden with the ``MSD_LUNG_DATA_ROOT`` environment
    variable. If not set, we fall back to ``../group_project/Task06_Lung``
    relative to the repository root.
    """

    env_override = os.environ.get("MSD_LUNG_DATA_ROOT")
    if env_override:
        return os.path.abspath(env_override)

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    )
    return os.path.abspath(os.path.join(os.path.dirname(repo_root), "group_project", "Task06_Lung"))


def _resolve_path(base: str, relative_path: str) -> str:
    """Resolve image/label paths listed in dataset.json.

    Handles common MSD paths such as ``./imagesTr/lung_001.nii.gz`` while
    tolerating pre-unzipped ``.nii`` files.
    """

    candidate = os.path.abspath(os.path.join(base, relative_path))
    if os.path.exists(candidate):
        return candidate

    if candidate.endswith(".nii.gz"):
        alt = candidate[: -len(".gz")]
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(f"Could not find file listed in dataset.json: {candidate}")


def _normalize_intensity(volume: np.ndarray, clip_range: Tuple[float, float]) -> np.ndarray:
    volume = np.clip(volume, *clip_range)
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    return volume.astype(np.float32)


def _pad_to_minimum(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Pad a 3D array to at least target_shape (D, H, W)."""

    depth, height, width = array.shape
    td, th, tw = target_shape
    pad_d = max(0, td - depth)
    pad_h = max(0, th - height)
    pad_w = max(0, tw - width)

    padding = (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )
    if any(p > 0 for pair in padding for p in pair):
        array = np.pad(array, padding, mode="constant", constant_values=0)
    return array


def _random_crop_pair(
    image: np.ndarray, label: Optional[np.ndarray], patch_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Randomly crop a patch of ``patch_size`` from the volume and label."""

    d, h, w = image.shape
    pd, ph, pw = patch_size

    max_d = max(0, d - pd)
    max_h = max(0, h - ph)
    max_w = max(0, w - pw)

    start_d = random.randint(0, max_d) if max_d > 0 else 0
    start_h = random.randint(0, max_h) if max_h > 0 else 0
    start_w = random.randint(0, max_w) if max_w > 0 else 0

    image_patch = image[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    label_patch = None
    if label is not None:
        label_patch = label[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    return image_patch, label_patch


def _center_crop_pair(
    image: np.ndarray, label: Optional[np.ndarray], patch_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d, h, w = image.shape
    pd, ph, pw = patch_size

    start_d = max(0, (d - pd) // 2)
    start_h = max(0, (h - ph) // 2)
    start_w = max(0, (w - pw) // 2)

    image_patch = image[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    label_patch = None
    if label is not None:
        label_patch = label[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
    return image_patch, label_patch


class MSDTask06LungDataset(Dataset):
    """Patch-based dataset for MSD Task06 Lung volumes."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        normalize: bool = True,
        augment: bool = True,
        clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE,
    ) -> None:
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment
        self.clip_range = clip_range

        dataset_json = os.path.join(self.data_root, "dataset.json")
        if not os.path.exists(dataset_json):
            raise FileNotFoundError(f"dataset.json not found in {self.data_root}")

        with open(dataset_json, "r") as fp:
            meta = json.load(fp)

        if split == "train":
            self.samples = [
                (
                    _resolve_path(self.data_root, entry["image"]),
                    _resolve_path(self.data_root, entry["label"]),
                )
                for entry in meta.get("training", [])
            ]
        elif split == "test":
            self.samples = [
                (_resolve_path(self.data_root, p), None) for p in meta.get("test", [])
            ]
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not self.samples:
            raise ValueError(f"No samples found for split={split} in {dataset_json}")

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_augment(self, image: np.ndarray, label: Optional[np.ndarray]):
        if not self.augment:
            return image, label

        # Random flips along each axis
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis)
                if label is not None:
                    label = np.flip(label, axis=axis)
        return image, label

    def __getitem__(self, idx: int):
        image_path, label_path = self.samples[idx]
        image = sitk.ReadImage(image_path)
        image_data = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)

        label_data = None
        if label_path is not None:
            label_image = sitk.ReadImage(label_path)
            label_data = sitk.GetArrayFromImage(label_image).astype(np.float32)

        image_data = _pad_to_minimum(image_data, self.patch_size)
        if label_data is not None:
            label_data = _pad_to_minimum(label_data, self.patch_size)

        if self.split == "train":
            image_patch, label_patch = _random_crop_pair(image_data, label_data, self.patch_size)
        else:
            image_patch, label_patch = _center_crop_pair(image_data, label_data, self.patch_size)

        image_patch, label_patch = self._maybe_augment(image_patch, label_patch)

        if self.normalize:
            image_patch = _normalize_intensity(image_patch, self.clip_range)

        # Ensure contiguous memory layout after flips/crops so torch.from_numpy works
        image_patch = np.ascontiguousarray(image_patch)
        if label_patch is not None:
            label_patch = np.ascontiguousarray(label_patch)
        
        image_tensor = torch.from_numpy(image_patch[None, ...])  # (1, D, H, W)
        label_tensor = None
        if label_patch is not None:
            label_tensor = torch.from_numpy(label_patch[None, ...])
        else:
            label_tensor = torch.zeros_like(image_tensor)

        return image_tensor, label_tensor
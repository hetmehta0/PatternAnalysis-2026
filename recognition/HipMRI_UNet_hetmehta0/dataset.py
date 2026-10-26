"""HipMRI data loading: frozen patient-level manifest -> 2D axial slices.

Pipeline
    1. ``load_manifest`` reads the existing ``metadata.csv`` (one row per scan),
       checks that no patient appears in more than one split, and checks the
       frozen 26/6/6 patient split and its SHA-256. No new splits are created.
    2. ``load_volume_pair`` loads one NIfTI image/mask pair in canonical (RAS)
       orientation, checks shape, affine and labels, and normalises the image.
    3. ``VolumeSet`` preloads every scan of one split into memory.
    4. ``SliceDataset`` indexes every axial slice ``[:, :, z]`` of a VolumeSet
       and returns ``image [1, H, W] float32`` and ``mask [H, W] int64``.

No augmentation is applied for the initial baseline. ``SliceDataset`` accepts
an optional ``transform(image, mask) -> (image, mask)`` hook for later work.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 6
REQUIRED_COLUMNS = ("patient_id", "scan_id", "image_path", "mask_path", "split")
SPLITS = ("train", "val", "test")

# Frozen split from COMP3710-Project/data/processed/dataset_report.json.
EXPECTED_PATIENTS_PER_SPLIT = {"train": 26, "val": 6, "test": 6}
EXPECTED_SPLIT_SHA256 = "4da182221410f102ad9282c9b1167ccc357273fe5703517da9a361f792dd7886"


# =============================================================================
# Manifest and split verification
# =============================================================================
def split_sha256(frame: pd.DataFrame) -> str:
    """Hash of sorted ``patient_id,split`` lines (same definition as the manifest report)."""
    pairs = sorted(set(zip(frame["patient_id"], frame["split"], strict=True)))
    return hashlib.sha256("\n".join(f"{p},{s}" for p, s in pairs).encode()).hexdigest()


def load_manifest(
    path: str | Path,
    expected_patients: dict[str, int] | None = EXPECTED_PATIENTS_PER_SPLIT,
    expected_sha256: str | None = EXPECTED_SPLIT_SHA256,
) -> pd.DataFrame:
    """Read and verify the scan manifest. Raises ValueError on any violation."""
    frame = pd.read_csv(path, dtype={"patient_id": str, "scan_id": str})
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["scan_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate scan IDs")
    invalid = set(frame["split"]) - set(SPLITS)
    if invalid:
        raise ValueError(f"Invalid split values: {sorted(invalid)}")

    splits_per_patient = frame.groupby("patient_id")["split"].nunique()
    leaking = sorted(splits_per_patient[splits_per_patient > 1].index)
    if leaking:
        raise ValueError(f"Patients appear in more than one split: {leaking}")

    if expected_patients is not None:
        counts = frame.groupby("split")["patient_id"].nunique().to_dict()
        counts = {split: int(counts.get(split, 0)) for split in SPLITS}
        if counts != expected_patients:
            raise ValueError(f"Patients per split {counts} != expected {expected_patients}")
    if expected_sha256 is not None:
        digest = split_sha256(frame)
        if digest != expected_sha256:
            raise ValueError(f"Split hash {digest} != frozen hash {expected_sha256}")
    return frame.reset_index(drop=True)


# =============================================================================
# Volume loading and preprocessing
# =============================================================================
def normalise_mri(image: np.ndarray) -> np.ndarray:
    """Per-volume robust z-score over non-zero voxels; zero voxels stay zero.

    Clips to the 0.5-99.5 percentile of non-zero voxels, then standardises.
    Ported from the author's earlier COMP3710-Project ``volume_io.normalise_mri``.
    """
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    nonzero = finite & (image != 0)
    values = image[nonzero]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(values, (0.5, 99.5))
    clipped = np.clip(np.where(finite, image, 0), low, high)
    foreground = clipped[nonzero]
    std = float(foreground.std())
    if std < 1e-8:
        std = 1.0
    result = (clipped - float(foreground.mean())) / std
    result[~nonzero] = 0.0
    return result.astype(np.float32, copy=False)


def _load_canonical(path: str | Path) -> nib.Nifti1Image:
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) == 4 and image.shape[-1] == 1:
        image = image.slicer[..., 0]
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {image.shape}")
    return image


def load_volume_pair(
    image_path: str | Path,
    mask_path: str | Path,
    num_classes: int = NUM_CLASSES,
    normalise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image float32 [H, W, D], mask uint8 [H, W, D]) in RAS orientation."""
    image_nii = _load_canonical(image_path)
    mask_nii = _load_canonical(mask_path)
    if image_nii.shape != mask_nii.shape:
        raise ValueError(
            f"Image/mask shape mismatch {image_nii.shape} vs {mask_nii.shape}: {mask_path}"
        )
    if not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-3):
        raise ValueError(f"Image/mask affine mismatch: {image_path} vs {mask_path}")

    raw_mask = np.asanyarray(mask_nii.dataobj)
    rounded = np.rint(raw_mask)
    if not np.array_equal(rounded, raw_mask):
        raise ValueError(f"Mask contains non-integer labels: {mask_path}")
    if rounded.min() < 0 or rounded.max() >= num_classes:
        labels = np.unique(rounded).astype(int).tolist()
        raise ValueError(f"Mask labels {labels} outside [0, {num_classes - 1}]: {mask_path}")
    mask = rounded.astype(np.uint8)
    del raw_mask, rounded

    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    if normalise:
        image = normalise_mri(image)
    return image, mask


@dataclass
class VolumeSet:
    """All scans of one split, preloaded into memory."""

    scan_ids: list[str]
    patient_ids: list[str]
    images: list[np.ndarray]  # each float32 [H, W, D]
    masks: list[np.ndarray]  # each uint8 [H, W, D]
    load_seconds: float = 0.0

    @classmethod
    def from_manifest(
        cls,
        frame: pd.DataFrame,
        split: str,
        scan_ids: Sequence[str] | None = None,
        normalise: bool = True,
        verbose: bool = False,
    ) -> VolumeSet:
        rows = frame[frame["split"] == split]
        if scan_ids is not None:
            rows = rows[rows["scan_id"].isin(set(scan_ids))]
        if rows.empty:
            raise ValueError(f"No scans selected for split {split!r}")
        start = time.perf_counter()
        images, masks = [], []
        for i, row in enumerate(rows.itertuples(index=False), start=1):
            image, mask = load_volume_pair(row.image_path, row.mask_path, normalise=normalise)
            images.append(image)
            masks.append(mask)
            if verbose and (i % 20 == 0 or i == len(rows)):
                print(f"  loaded {i}/{len(rows)} {split} volumes", flush=True)
        return cls(
            scan_ids=rows["scan_id"].tolist(),
            patient_ids=rows["patient_id"].tolist(),
            images=images,
            masks=masks,
            load_seconds=time.perf_counter() - start,
        )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_slices(self) -> int:
        return sum(image.shape[2] for image in self.images)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.images) + sum(m.nbytes for m in self.masks)


# =============================================================================
# 2D slice dataset
# =============================================================================
SliceIndex = tuple[int, int]  # (volume index, axial slice index z)


def all_slice_indices(volumes: VolumeSet) -> list[SliceIndex]:
    """Every axial slice of every volume (depth taken from each volume's shape)."""
    return [(v, z) for v, image in enumerate(volumes.images) for z in range(image.shape[2])]


def select_overfit_slices(volumes: VolumeSet, count: int, seed: int) -> list[SliceIndex]:
    """Seeded choice of ``count`` slices that contain at least one foreground pixel."""
    candidates = [
        (v, z)
        for v, mask in enumerate(volumes.masks)
        for z in np.flatnonzero(mask.any(axis=(0, 1)))
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} foreground slices available, need {count}")
    chosen = np.random.default_rng(seed).choice(len(candidates), size=count, replace=False)
    return sorted((int(candidates[i][0]), int(candidates[i][1])) for i in chosen)


class SliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """2D axial slices drawn from a preloaded VolumeSet."""

    def __init__(
        self,
        volumes: VolumeSet,
        indices: Sequence[SliceIndex] | None = None,
        transform: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.volumes = volumes
        self.indices = list(indices) if indices is not None else all_slice_indices(volumes)
        self.transform = transform  # None for the baseline: no augmentation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        v, z = self.indices[index]
        image = self.volumes.images[v][:, :, z]
        mask = self.volumes.masks[v][:, :, z]
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))
        return image, mask

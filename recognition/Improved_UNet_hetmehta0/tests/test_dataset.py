import copy
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from dataset import (HipMRIDataset, audit_rows, fingerprint, load_manifest, load_pair,
                     normalize, patient_split, read_pairs, read_slice, write_json)
from synthetic import make_fixture


@pytest.fixture
def fixture(tmp_path):
    path = make_fixture(tmp_path / "synthetic")
    return path, load_manifest(path)


def test_reproducible_patient_partitions(fixture):
    _, manifest = fixture
    rows = manifest["rows"]
    assert patient_split(rows) == patient_split(list(reversed(rows)))
    partitions = [{r["patient_id"] for r in rows if r["split"] == s}
                  for s in ("train", "validation", "test")]
    assert all(partitions)
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert sum(map(len, partitions)) == 10


def test_fixed_channels_and_missing_class(fixture):
    _, manifest = fixture
    sample = HipMRIDataset(manifest, "train", size=(33, 35))[0]
    assert sample["one_hot"].shape == (3, 33, 35)
    assert torch.equal(sample["one_hot"].sum(0), torch.ones(33, 35))
    assert torch.equal(sample["one_hot"].argmax(0), sample["target"])
    row = manifest["rows"][0]
    nii = nib.load(row["mask"])
    data = np.zeros(nii.shape, np.int16)
    nib.save(nib.Nifti1Image(data, nii.affine), row["mask"])
    _, _, target = load_pair(row, manifest["classes"])
    assert target.max() == 0


@pytest.mark.parametrize("fault", ["affine", "shape", "unknown", "fractional", "nonfinite"])
def test_invalid_pairs_rejected(fixture, fault):
    _, manifest = fixture
    row = manifest["rows"][0]
    nii = nib.load(row["mask"])
    data, affine = nii.get_fdata(), nii.affine.copy()
    if fault == "affine":
        affine[0, 3] += 1
    elif fault == "shape":
        data = data[:-1]
    elif fault == "unknown":
        data[0, 0, 0] = 99
    elif fault == "fractional":
        data[0, 0, 0] = 0.5
    else:
        data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, affine), row["mask"])
    with pytest.raises(ValueError):
        load_pair(row, manifest["classes"])


def test_no_volume_truncation(tmp_path):
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((32, 32, 2)), np.eye(4)), path)
    with pytest.raises(ValueError, match="Expected 2D"):
        read_slice(path)


def test_constant_normalization():
    assert np.array_equal(normalize(np.ones((32, 32))), np.zeros((32, 32)))
    x = normalize(np.arange(1024, dtype=np.float32).reshape(32, 32))
    assert abs(x.mean()) < 1e-6
    assert abs(x.std() - 1) < 1e-6


def test_file_and_manifest_tampering(fixture):
    path, manifest = fixture
    changed = copy.deepcopy(manifest)
    changed["seed"] += 1
    write_json(path, changed)
    with pytest.raises(ValueError, match="fingerprint"):
        load_manifest(path)
    write_json(path, manifest)
    with open(manifest["rows"][0]["image"], "ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path)


def test_leakage_even_with_recomputed_hash(fixture):
    path, manifest = fixture
    first = manifest["rows"][0]
    second = next(r for r in manifest["rows"] if r["patient_id"] == first["patient_id"]
                  and r["case_id"] != first["case_id"])
    second["split"] = "test" if first["split"] != "test" else "train"
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items()
                                           if k != "fingerprint"})
    write_json(path, manifest)
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path)


def test_reused_files_and_duplicate_pixels_rejected(fixture):
    _, manifest = fixture
    rows = copy.deepcopy(manifest["rows"])
    rows[1]["image"] = rows[0]["image"]
    with pytest.raises(ValueError, match="Reused"):
        audit_rows(rows, manifest["classes"])
    rows = copy.deepcopy(manifest["rows"])
    other = next(r for r in rows if r["patient_id"] != rows[0]["patient_id"])
    nib.save(nib.load(rows[0]["image"]), other["image"])
    with pytest.raises(ValueError, match="Identical image pixels"):
        audit_rows(rows, manifest["classes"])


def test_augmentation_is_paired_and_train_only(fixture):
    _, manifest = fixture
    with pytest.raises(ValueError, match="only"):
        HipMRIDataset(manifest, "test", augment=True)
    original = HipMRIDataset(manifest, "train", size=(33, 35))[0]
    torch.manual_seed(5)
    augmented = HipMRIDataset(manifest, "train", size=(33, 35), augment=True)[0]
    assert any(torch.equal(augmented["image"], original["image"].flip(axes))
               and torch.equal(augmented["target"],
                               original["target"].flip(tuple(a - 1 for a in axes)))
               for axes in ((), (1,), (2,), (1, 2)))


def test_small_cohort_rejected(fixture):
    _, manifest = fixture
    with pytest.raises(ValueError, match="three patients"):
        patient_split(manifest["rows"][:2])

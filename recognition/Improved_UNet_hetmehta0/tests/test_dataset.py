import copy
import json
import shutil

import nibabel as nib
import numpy as np
import pytest
import torch

from dataset import (HIPMRI_CLASSES, HipMRIDataset, audit_rows, fingerprint,
                     index_hipmri, inventory, load_manifest, load_pair, normalize,
                     patient_split, prepare_manifest, read_pairs, read_slice,
                     write_json)
from fixture_data import make_test_fixture


@pytest.fixture
def fixture(tmp_path):
    path = make_test_fixture(tmp_path / "fixture")
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


def test_inventory_reports_observed_labels_without_interpreting_them(fixture):
    path, _ = fixture
    report = inventory(path.parent)
    assert report["nifti_count"] == 40
    masks = [r for r in report["files"] if "_mask" in r["path"]]
    assert all(r["unique_values_if_at_most_32"] == [0, 2, 5] for r in masks)
    assert all(r["nonfinite_pixels"] == 0 for r in report["files"])


def test_official_hipmri_indexer_pairs_prefixes_and_preserves_splits(fixture, tmp_path):
    _, manifest = fixture
    root = tmp_path / "HipMRI-2D"
    source_rows = manifest["rows"][:3]
    for number, (folder_split, split) in enumerate(
            (("train", "train"), ("validate", "validation"), ("test", "test")), 4):
        image_dir = root / f"keras_slices_{folder_split}"
        mask_dir = root / f"keras_slices_seg_{folder_split}"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        key = f"{number:03d}_week_0_slice_0.nii.gz"
        shutil.copyfile(source_rows[number - 4]["image"], image_dir / f"case_{key}")
        shutil.copyfile(source_rows[number - 4]["mask"], mask_dir / f"seg_{key}")
    rows = index_hipmri(root)
    assert [row["split"] for row in rows] == ["train", "validation", "test"]
    assert {row["patient_id"] for row in rows} == {"case_004", "case_005", "case_006"}
    prepared = prepare_manifest(rows, HIPMRI_CLASSES, preserve_splits=True)
    assert prepared["split_method"] == "supplied patient-disjoint partitions"
    assert prepared["fractions"] is None


def test_official_hipmri_indexer_rejects_missing_pair(fixture, tmp_path):
    path, _ = fixture
    root = tmp_path / "HipMRI-2D"
    for split in ("train", "validate", "test"):
        (root / f"keras_slices_{split}").mkdir(parents=True)
        (root / f"keras_slices_seg_{split}").mkdir(parents=True)
    shutil.copyfile(next(path.parent.glob("*_image.nii.gz")),
                    root / "keras_slices_train/case_004_week_0_slice_0.nii.gz")
    with pytest.raises(ValueError, match="Unpaired"):
        index_hipmri(root)

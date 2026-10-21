import argparse
import copy

import nibabel as nib
import numpy as np
import pytest
import torch

from dataset import HipMRIDataset, load_manifest, load_pair
from metrics import segmentation_loss
from modules import build_model
from predict import export_case
from fixture_data import make_test_fixture
from train import evaluate, fit, load_checkpoint, set_seed


@pytest.mark.parametrize("architecture", ["baseline", "improved"])
def test_train_reload_evaluate_export(tmp_path, architecture):
    path = make_test_fixture(tmp_path / "fixture")
    run = tmp_path / architecture
    args = argparse.Namespace(manifest=path, model=architecture, out=run,
                              epochs=1, batch_size=4, base_channels=2,
                              size=[32, 32], lr=1e-3, seed=3710,
                              threads=1, device="cpu")
    summary = fit(args)
    assert not summary["test_evaluated"]
    assert (run / "learning_curves.png").stat().st_size > 0
    manifest = load_manifest(path)
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(run / "best.pt", manifest, device)
    ds = HipMRIDataset(manifest, "test", (32, 32))
    report = evaluate(model, ds, device)
    assert report["evaluated_slices"] == 2
    assert report["metric_grid"] == "original Nifti image/mask grid"
    assert sum(v["ground_truth_pixels"] for v in report["pooled"].values()) == 2 * 33 * 35
    assert report["peak_cuda_allocated_bytes"] is None
    assert checkpoint["epoch"] == 1
    again, _ = load_checkpoint(run / "best.pt", manifest, device)
    x = ds[0]["image"][None]
    with torch.no_grad():
        assert torch.equal(model(x), again(x))
    # Known prediction verifies raw noncontiguous label encoding and geometry.
    row = ds.rows[0]
    nii, image, target = load_pair(row, manifest["classes"])
    exported = export_case(run, 0, row, nii, image, target, target,
                           manifest["classes"], 1)
    saved = nib.load(run / exported["file"])
    assert saved.shape == nii.shape
    assert np.allclose(saved.affine, nii.affine)
    assert np.array_equal(saved.get_fdata(), nib.load(row["mask"]).get_fdata())
    assert (run / exported["file"].replace("_prediction.nii.gz", "_panel.png")).exists()
    changed = copy.deepcopy(manifest)
    changed["fingerprint"] = "different split"
    with pytest.raises(ValueError, match="do not match"):
        load_checkpoint(run / "best.pt", changed, device)


@pytest.mark.parametrize("architecture", ["baseline", "improved"])
def test_repeated_batch_can_reduce_loss(architecture):
    """Optimization sanity check, NOT generalization or HipMRI performance."""
    set_seed(3710)
    model = build_model(architecture, 2, 2)
    x = torch.zeros(2, 1, 32, 32)
    x[:, :, 10:22, 10:22] = 1
    target = x[:, 0].long()
    onehot = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    model.eval()  # Disable dropout to isolate an optimizer wiring check.
    with torch.no_grad():
        initial = float(segmentation_loss(model(x), target, onehot))
    for _ in range(40):
        optimizer.zero_grad()
        loss = segmentation_loss(model(x), target, onehot)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(segmentation_loss(model(x), target, onehot))
    assert final < initial * .9, (initial, final)

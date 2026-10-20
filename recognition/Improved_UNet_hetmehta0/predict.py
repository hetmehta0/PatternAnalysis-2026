"""Load a trained checkpoint, export native-grid Nifti masks and failure panels."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dataset import HipMRIDataset, load_manifest, write_json
from train import device_from_name, evaluate, load_checkpoint, set_seed


def export_case(out, index, row, nii, image, target, prediction, classes,
                synthetic, max_panels):
    # Never use untrusted case IDs as filesystem paths.
    stem = f"case_{index:04d}_{hashlib.sha256(row['case_id'].encode()).hexdigest()[:10]}"
    raw_labels = np.array([c["value"] for c in classes], dtype=np.int16)[prediction]
    if len(nii.shape) == 3:
        raw_labels = raw_labels[..., None]
    header = nii.header.copy()
    header.set_data_dtype(np.int16)
    output = nib.Nifti1Image(raw_labels, nii.affine, header)
    for form in ("qform", "sform"):
        transform, code = getattr(nii, f"get_{form}")(coded=True)
        getattr(output, f"set_{form}")(transform, int(code))
    nib.save(output, out / f"{stem}_prediction.nii.gz")
    if index < max_panels:
        prostate = next(i for i, cls in enumerate(classes) if cls["name"] == "prostate")
        gt, pred = target == prostate, prediction == prostate
        overlay = np.zeros((*image.shape, 4), np.float32)
        overlay[pred & ~gt] = (1, 0.1, 0.1, 0.8)
        overlay[gt & ~pred] = (0.1, 0.5, 1, 0.8)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
        for ax in axes:
            ax.axis("off")
        axes[0].imshow(image.T, cmap="gray", origin="lower")
        axes[1].imshow(target.T, vmin=0, vmax=len(classes) - 1, interpolation="nearest",
                       cmap="viridis", origin="lower")
        axes[2].imshow(prediction.T, vmin=0, vmax=len(classes) - 1, interpolation="nearest",
                       cmap="viridis", origin="lower")
        axes[3].imshow(image.T, cmap="gray", origin="lower")
        axes[3].imshow(overlay.transpose(1, 0, 2), origin="lower")
        for ax, title in zip(axes, ("Input", "Ground truth (all classes)",
                                   "Prediction (all classes)", "Prostate: red FP / blue FN")):
            ax.set_title(title, fontsize=10)
        prefix = "SYNTHETIC — NOT HipMRI | " if synthetic else ""
        fig.suptitle(prefix + row["case_id"], fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_panel.png", dpi=150)
        plt.close(fig)
    return {"case_id": row["case_id"], "file": f"{stem}_prediction.nii.gz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--max-panels", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    set_seed(3710)
    manifest = load_manifest(args.manifest)
    device = device_from_name(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, manifest, device)
    dataset = HipMRIDataset(manifest, args.split, checkpoint["config"]["size"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    exports = []

    def save(index, row, nii, image, target, prediction):
        exports.append(export_case(out, index, row, nii, image, target, prediction,
                                   manifest["classes"], manifest["synthetic"], args.max_panels))

    report = evaluate(model, dataset, device, on_case=save)
    report.update({"synthetic": manifest["synthetic"], "split": args.split,
                   "manifest_fingerprint": manifest["fingerprint"], "exports": exports,
                   "checkpoint_epoch": checkpoint["epoch"]})
    write_json(out / "metrics.json", report)
    print(json.dumps({"synthetic": manifest["synthetic"],
                      "patient_macro": report["patient_macro"],
                      "exported_slices": len(exports)}, indent=2))


if __name__ == "__main__":
    main()

"""Generate tiny SYNTHETIC Nifti fixtures, never HipMRI evidence or training data."""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from dataset import prepare_manifest, read_pairs, write_json


def make_fixture(root, patients=10, slices=2, seed=3710):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    classes = [{"value": 0, "name": "background"},
               {"value": 2, "name": "synthetic_other"},
               {"value": 5, "name": "prostate"}]
    rows = []
    yy, xx = np.mgrid[:33, :35]
    affine = np.diag([0.8, 0.9, 3.0, 1.0])
    for patient in range(patients):
        for section in range(slices):
            case = f"SYNTHETIC_p{patient:02d}_s{section:02d}"
            target = np.zeros((33, 35), np.int16)
            target[(yy - 9)**2 + (xx - 9)**2 < 16] = 2
            target[(yy - 19 - section)**2 + (xx - 19)**2 < 25 + patient] = 5
            image = (rng.normal(0, 0.15, target.shape) + target / 5).astype(np.float32)
            ip, mp = root / f"{case}_image.nii.gz", root / f"{case}_mask.nii.gz"
            nib.save(nib.Nifti1Image(image[..., None], affine), ip)
            nib.save(nib.Nifti1Image(target[..., None], affine), mp)
            rows.append({"patient_id": f"SYNTHETIC_p{patient:02d}", "case_id": case,
                         "image": ip.name, "mask": mp.name})
    with (root / "pairs.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["patient_id", "case_id", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "classes.json", classes)
    manifest = prepare_manifest(read_pairs(root / "pairs.csv"), classes, seed, synthetic=True)
    write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory; refuses to overwrite")
    args = parser.parse_args()
    print(make_fixture(args.out))

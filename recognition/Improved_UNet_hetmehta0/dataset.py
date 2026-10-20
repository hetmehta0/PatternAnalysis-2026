"""COMP3710 Nifti pipeline. AI assistance is disclosed in README.md.

Pairing/patient identity comes from a reviewed CSV, never guessed from filenames.
Only 2D or (H,W,1) Nifti is accepted. No first-slice truncation of 3D volumes.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_classes(path):
    """Ordered JSON [{value: raw_integer, name: meaning}, ...]; index 0 is background."""
    classes = json.loads(Path(path).read_text())
    validate_classes(classes)
    return classes


def validate_classes(classes):
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("Provide a fixed vocabulary with at least two classes")
    values = [c["value"] for c in classes]
    names = [c["name"] for c in classes]
    if any(type(v) is not int or v < 0 or v > 32767 for v in values):
        raise ValueError("Raw labels must be nonnegative int16 integers")
    if len(set(values)) != len(values) or len(set(names)) != len(names):
        raise ValueError("Class values and names must be unique")
    if names[0] != "background" or "prostate" not in names:
        raise ValueError("First class must be background; a prostate class is required")


def read_pairs(path):
    """CSV: patient_id,case_id,image,mask. Relative paths resolve beside the CSV."""
    path = Path(path).resolve()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not {"patient_id", "case_id", "image", "mask"} <= set(reader.fieldnames or []):
            raise ValueError("CSV needs patient_id,case_id,image,mask columns")
        rows = []
        for row in reader:
            entry = {key: row[key].strip() for key in
                     ("patient_id", "case_id", "image", "mask")}
            if not all(entry.values()):
                raise ValueError("Empty patient ID, case ID or file path")
            for key in ("image", "mask"):
                p = Path(entry[key])
                entry[key] = str((path.parent / p).resolve())
                if not Path(entry[key]).is_file():
                    raise FileNotFoundError(entry[key])
            rows.append(entry)
    validate_rows(rows)
    return rows


def validate_rows(rows):
    if not rows:
        raise ValueError("No cases")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case_id")
    paths = [str(Path(r[k]).resolve()) for r in rows for k in ("image", "mask")]
    if len(set(paths)) != len(paths):
        raise ValueError("Reused image/mask path (including symlinks)")


def read_slice(path):
    nii = nib.load(path)
    shape = nii.shape
    if not (len(shape) == 2 or (len(shape) == 3 and shape[2] == 1)):
        raise ValueError(f"Expected 2D or (H,W,1), got {shape}: {path}")
    data = nii.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[:, :, 0]
    if min(data.shape) < 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid dimensions or nonfinite intensities: {path}")
    if not np.isfinite(nii.affine).all() or abs(np.linalg.det(nii.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid Nifti affine: {path}")
    # Reject contradictory coded transforms rather than silently choosing one.
    q, qc = nii.get_qform(coded=True)
    s, sc = nii.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-4, rtol=0):
        raise ValueError(f"Conflicting qform/sform: {path}")
    return nii, data


def load_pair(row, classes):
    image_nii, image = read_slice(row["image"])
    mask_nii, mask = read_slice(row["mask"])
    if image_nii.shape != mask_nii.shape or not np.allclose(
            image_nii.affine, mask_nii.affine, atol=1e-4, rtol=0):
        raise ValueError(f"Image/mask spatial alignment mismatch: {row['case_id']}")
    if not np.equal(mask, np.rint(mask)).all():
        raise ValueError(f"Noninteger mask labels: {row['case_id']}")
    raw_values = {c["value"] for c in classes}
    unknown = set(np.unique(mask).astype(int)) - raw_values
    if unknown:
        raise ValueError(f"Unknown mask labels {sorted(unknown)}: {row['case_id']}")
    encoded = np.empty(mask.shape, dtype=np.int64)
    for index, cls in enumerate(classes):
        encoded[mask == cls["value"]] = index
    return image_nii, image, encoded


def patient_split(rows, seed=3710, val_fraction=0.15, test_fraction=0.15):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to < 1")
    patients = sorted({r["patient_id"] for r in rows})
    if len(patients) < 3:
        raise ValueError("At least three patients are needed for three nonempty splits")
    random.Random(seed).shuffle(patients)
    nv = max(1, int(len(patients) * val_fraction))
    nt = max(1, int(len(patients) * test_fraction))
    if nv + nt >= len(patients):
        raise ValueError("Not enough training patients for requested fractions")
    partition = {p: "validation" for p in patients[:nv]}
    partition.update({p: "test" for p in patients[nv:nv + nt]})
    partition.update({p: "train" for p in patients[nv + nt:]})
    return [dict(r, split=partition[r["patient_id"]])
            for r in sorted(rows, key=lambda row: row["case_id"])]


def audit_rows(rows, classes):
    """Inspect every pair, count original pixels, and seal file identities."""
    validate_classes(classes)
    validate_rows(rows)
    shapes, counts, spacings = Counter(), Counter(), Counter()
    patient_content = {}
    sealed = []
    for row in rows:
        nii, image, target = load_pair(row, classes)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        owner = patient_content.setdefault(digest, row["patient_id"])
        if owner != row["patient_id"]:
            raise ValueError("Identical image pixels assigned to different patients")
        shapes[str(tuple(image.shape))] += 1
        spacings[str(tuple(float(v) for v in nii.header.get_zooms()[:2]))] += 1
        for index, cls in enumerate(classes):
            counts[cls["name"]] += int((target == index).sum())
        sealed.append(dict(row, image_sha256=file_hash(row["image"]),
                           mask_sha256=file_hash(row["mask"])))
    report = {
        "patients": len({r["patient_id"] for r in rows}), "slices": len(rows),
        "shape_counts": dict(shapes), "in_plane_spacing_counts": dict(spacings),
        "original_pixel_counts": dict(counts),
        "identity_warning": "Patient IDs and label semantics require human source verification.",
        "alignment": "All pairs passed shape/affine checks; semantic pairing still requires visual review.",
    }
    return sealed, report


def prepare_manifest(pairs, classes, seed=3710, synthetic=False,
                     val_fraction=0.15, test_fraction=0.15):
    rows, audit = audit_rows(pairs, classes)
    rows = patient_split(rows, seed, val_fraction, test_fraction)
    audit["splits"] = {
        s: {"patients": len({r["patient_id"] for r in rows if r["split"] == s}),
            "slices": sum(r["split"] == s for r in rows)}
        for s in ("train", "validation", "test")
    }
    # Ensure the optimization and selection labels exist in their partitions.
    prostate = next(i for i, c in enumerate(classes) if c["name"] == "prostate")
    for split in ("train", "validation", "test"):
        if not any((load_pair(r, classes)[2] == prostate).any()
                   for r in rows if r["split"] == split):
            raise ValueError(f"No prostate pixels in {split}; review patient stratification")
    result = {"schema_version": 1, "synthetic": bool(synthetic), "seed": seed,
              "fractions": {"validation": val_fraction, "test": test_fraction},
              "classes": classes, "rows": rows, "audit": audit}
    result["fingerprint"] = fingerprint(result)
    return result


def load_manifest(path, verify_files=True):
    manifest = json.loads(Path(path).read_text())
    expected = manifest["fingerprint"]
    if fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != expected:
        raise ValueError("Manifest fingerprint mismatch")
    validate_classes(manifest["classes"])
    validate_rows(manifest["rows"])
    patient_partitions = {}
    present = set()
    for row in manifest["rows"]:
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Invalid split name")
        present.add(split)
        old = patient_partitions.setdefault(row["patient_id"], split)
        if old != split:
            raise ValueError("Patient leakage across splits")
        if verify_files:
            for key in ("image", "mask"):
                if file_hash(row[key]) != row[f"{key}_sha256"]:
                    raise ValueError(f"File changed since audit: {row[key]}")
    if present != {"train", "validation", "test"}:
        raise ValueError("All three splits must be nonempty")
    return manifest


def normalize(image):
    centered = image.astype(np.float32) - np.float32(image.mean(dtype=np.float64))
    scale = float(centered.std(dtype=np.float64))
    return centered / max(scale, 1e-8)


class HipMRIDataset(Dataset):
    def __init__(self, manifest, split, size=(256, 256), augment=False):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if augment and split != "train":
            raise ValueError("Augmentation is only allowed on training data")
        if len(size) != 2 or min(size) < 32:
            raise ValueError("Use two spatial dimensions >= 32")
        self.rows = [r for r in manifest["rows"] if r["split"] == split]
        self.classes = manifest["classes"]
        self.size = tuple(size)
        self.augment = augment
        self.seed = manifest["seed"]
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        _, image, target = load_pair(row, self.classes)
        x = torch.from_numpy(normalize(image))[None]
        y = torch.from_numpy(target)
        x = F.interpolate(x[None], size=self.size, mode="bilinear",
                          align_corners=False)[0]
        y = F.interpolate(y[None, None].float(), size=self.size,
                          mode="nearest")[0, 0].long()
        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
            for axis in (0, 1):
                if torch.rand((), generator=generator) < 0.5:
                    x, y = x.flip(axis + 1), y.flip(axis)
        one_hot = F.one_hot(y, num_classes=len(self.classes)).permute(2, 0, 1).float()
        return {"image": x, "target": y, "one_hot": one_hot,
                "patient_id": row["patient_id"], "case_id": row["case_id"]}


def inventory(root):
    """Read-only unpaired inventory. Does not invent patient IDs or class meanings."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory inaccessible: {root}")
    files = sorted(p for p in root.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
    items = []
    for path in files:
        nii = nib.load(path)
        data = nii.get_fdata(dtype=np.float32)
        finite = np.isfinite(data)
        values = np.unique(data[finite])
        items.append({"path": str(path.relative_to(root)), "shape": list(nii.shape),
                      "dtype": str(nii.get_data_dtype()),
                      "spacing": [float(v) for v in nii.header.get_zooms()],
                      "nonfinite_pixels": int((~finite).sum()),
                      "min": float(values[0]) if len(values) else None,
                      "max": float(values[-1]) if len(values) else None,
                      "unique_values_if_at_most_32": values.tolist() if len(values) <= 32 else None})
    return {"root": str(root.resolve()), "nifti_count": len(items), "files": items,
            "warning": "Unpaired inventory only. Verify patient IDs and label ontology before prepare."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inventory")
    inspect.add_argument("--root", required=True)
    inspect.add_argument("--out", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--pairs", required=True)
    prepare.add_argument("--classes", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--seed", type=int, default=3710)
    prepare.add_argument("--val-fraction", type=float, default=0.15)
    prepare.add_argument("--test-fraction", type=float, default=0.15)
    prepare.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.root)
    else:
        result = prepare_manifest(read_pairs(args.pairs), load_classes(args.classes),
                                  args.seed, args.synthetic,
                                  args.val_fraction, args.test_fraction)
    write_json(args.out, result)
    print(json.dumps(result.get("audit", {"nifti_count": result.get("nifti_count")}), indent=2))


if __name__ == "__main__":
    main()

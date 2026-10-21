"""Run both complete CLIs on a small, real HipMRI pipeline-check subset."""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from dataset import load_manifest, load_pair, prepare_manifest, write_json


def real_smoke_subset(manifest, positive_per_patient=2, negative_per_patient=1):
    """Keep a small deterministic mixture of positive/negative slices per patient.

    This intentionally changes the natural class distribution and is therefore
    only suitable for checking execution, never for estimating performance.
    """
    prostate = next(i for i, cls in enumerate(manifest["classes"])
                    if cls["name"] == "prostate")
    grouped = defaultdict(list)
    for row in manifest["rows"]:
        grouped[row["patient_id"]].append(row)
    selected = []
    for patient, rows in sorted(grouped.items()):
        positive, negative = [], []
        for row in sorted(rows, key=lambda item: item["case_id"]):
            target = load_pair(row, manifest["classes"])[2]
            bucket = positive if (target == prostate).any() else negative
            limit = positive_per_patient if bucket is positive else negative_per_patient
            if len(bucket) < limit:
                bucket.append(row)
            if (len(positive) == positive_per_patient and
                    len(negative) == negative_per_patient):
                break
        selected.extend(positive + negative)
    return prepare_manifest(
        selected, manifest["classes"], manifest["seed"], preserve_splits=True,
        source={"name": "HipMRI real-data pipeline smoke subset",
                "parent_manifest_fingerprint": manifest["fingerprint"],
                "selection": {"positive_slices_per_patient": positive_per_patient,
                              "negative_slices_per_patient": negative_per_patient},
                "warning": "Class-balanced subset; not representative and not a benchmark."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Full audited HipMRI manifest")
    parser.add_argument("--out", required=True, help="New ignored run directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--size", nargs=2, type=int, default=[64, 64])
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    full_manifest = load_manifest(args.manifest)
    manifest = real_smoke_subset(full_manifest)
    manifest_path = root / "real-smoke-manifest.json"
    write_json(manifest_path, manifest)
    script_dir = Path(__file__).resolve().parent
    summary = {
        "data_source": "HipMRI",
        "representative_experiment": False,
        "claim": "Real HipMRI execution check on a deliberately balanced subset; not a benchmark.",
        "full_manifest_fingerprint": full_manifest["fingerprint"],
        "smoke_manifest_fingerprint": manifest["fingerprint"],
        "smoke_audit": manifest["audit"],
        "models": {},
    }

    def run(script, *options):
        command = [sys.executable, str(script_dir / script), *map(str, options)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        with (root / "commands.log").open("a") as log:
            log.write("COMMAND: " + " ".join(command) + "\n" + result.stdout + result.stderr + "\n")
        print(result.stdout, flush=True)

    for model in ("baseline", "improved"):
        output = root / model
        run("train.py", "fit", "--manifest", manifest_path, "--model", model,
            "--out", output, "--epochs", args.epochs, "--base-channels", 2,
            "--batch-size", 8, "--size", *args.size, "--device", args.device)
        run("train.py", "evaluate", "--manifest", manifest_path,
            "--checkpoint", output / "best.pt", "--split", "test",
            "--out", output / "test_metrics.json", "--device", args.device)
        run("predict.py", "--manifest", manifest_path,
            "--checkpoint", output / "best.pt", "--split", "test",
            "--out", output / "predictions", "--device", args.device)
        training = json.loads((output / "training_summary.json").read_text())
        test = json.loads((output / "test_metrics.json").read_text())
        prediction = json.loads((output / "predictions/metrics.json").read_text())
        if test["pooled"] != prediction["pooled"]:
            raise RuntimeError("Evaluation and prediction metrics disagree")
        summary["models"][model] = {
            "training": training, "native_grid_test": test["patient_macro"],
            "mean_forward_seconds": test["latency"]["mean_seconds_per_slice"],
            "prediction_exports": len(prediction["exports"]),
        }
    write_json(root / "smoke_summary.json", summary)
    print("REAL HIPMRI CLI SMOKE: both models trained, reloaded, evaluated and exported.")
    print("This balanced subset is not a performance benchmark.")


if __name__ == "__main__":
    main()

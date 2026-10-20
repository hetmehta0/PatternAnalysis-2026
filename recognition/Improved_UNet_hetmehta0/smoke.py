"""Run both complete CLIs on synthetic data and save an auditable summary."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from dataset import write_json
from synthetic import make_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory outside Git or under ignored runs/")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = make_fixture(root / "SYNTHETIC_DATA")
    script_dir = Path(__file__).resolve().parent
    summary = {"synthetic": True, "claim": "Software smoke test only; no HipMRI experiments.",
               "models": {}}

    def run(script, *options):
        command = [sys.executable, str(script_dir / script), *map(str, options)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        with (root / "commands.log").open("a") as log:
            log.write("COMMAND: " + " ".join(command) + "\n" + result.stdout + result.stderr + "\n")
        print(result.stdout, flush=True)

    for model in ("baseline", "improved"):
        output = root / model
        run("train.py", "fit", "--manifest", manifest, "--model", model,
            "--out", output, "--epochs", args.epochs, "--base-channels", 2,
            "--batch-size", 4, "--size", 32, 32, "--device", "cpu")
        run("train.py", "evaluate", "--manifest", manifest, "--checkpoint", output / "best.pt",
            "--split", "test", "--out", output / "test_metrics.json")
        run("predict.py", "--manifest", manifest, "--checkpoint", output / "best.pt",
            "--split", "test", "--out", output / "predictions")
        train = json.loads((output / "training_summary.json").read_text())
        test = json.loads((output / "test_metrics.json").read_text())
        prediction = json.loads((output / "predictions/metrics.json").read_text())
        if test["pooled"] != prediction["pooled"]:
            raise RuntimeError("Evaluation and prediction metrics disagree")
        summary["models"][model] = {
            "training": train, "native_grid_test": test["patient_macro"],
            "mean_forward_seconds": test["latency"]["mean_seconds_per_slice"],
            "prediction_exports": len(prediction["exports"]),
        }
    write_json(root / "smoke_summary.json", summary)
    print("SYNTHETIC CLI SMOKE: both models trained, reloaded, evaluated and exported.")


if __name__ == "__main__":
    main()

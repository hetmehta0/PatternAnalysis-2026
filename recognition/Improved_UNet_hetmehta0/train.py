"""Train on train patients, select on validation, explicitly evaluate held-out test.

AI assistance and architectural adaptations are described in README.md.
"""
import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, load_manifest, load_pair, write_json
from metrics import confusion_matrix, segmentation_loss, summarize_counts
from modules import build_model


def set_seed(seed, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def device_from_name(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda"
            else platform.processor() or platform.machine()}


@torch.inference_mode()
def evaluate(model, dataset, device, on_case=None):
    """Native-grid pixel metrics, patient aggregation and warm model-only timing.

on_case receives native prediction for exports; no metrics are computed on
resized masks. Patient aggregates include all supplied slices/scans for that ID.
"""
    model.eval()
    counts, latencies, losses = {}, [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Warm-up is excluded from reported forward latency.
    sample = dataset[0]
    warm = sample["image"][None].to(device)
    model(warm)
    synchronize(device)
    for index, row in enumerate(dataset.rows):
        sample = dataset[index]
        nii, image, target_array = load_pair(row, dataset.classes)
        x = sample["image"][None].to(device)
        target = sample["target"][None].to(device)
        one_hot = sample["one_hot"][None].to(device)
        synchronize(device)
        started = time.perf_counter()
        logits = model(x)
        synchronize(device)
        seconds = time.perf_counter() - started
        latencies.append({"case_id": row["case_id"], "forward_seconds": seconds})
        losses.append(float(segmentation_loss(logits, target, one_hot)))
        native_logits = F.interpolate(logits, size=target_array.shape,
                                      mode="bilinear", align_corners=False)
        pred = native_logits.argmax(1)[0].cpu()
        matrix = confusion_matrix(pred, torch.from_numpy(target_array), len(dataset.classes))
        patient = row["patient_id"]
        counts[patient] = counts.get(patient, torch.zeros_like(matrix)) + matrix
        if on_case is not None:
            on_case(index, row, nii, image, target_array, pred.numpy())
    result = summarize_counts(counts, dataset.classes)
    result.update({
        "mean_loss_resized_grid": sum(losses) / len(losses),
        "evaluated_slices": len(dataset),
        "metric_grid": "original Nifti image/mask grid",
        "latency": {"scope": "batch=1 warmed forward only; excludes IO, preprocessing, export",
                    "mean_seconds_per_slice": sum(v["forward_seconds"] for v in latencies) / len(latencies),
                    "per_case": latencies},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" else None,
    })
    return result


def save_curves(history, path, synthetic):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set_ylabel("CE + soft Dice loss")
    axes[1].plot(epochs, [r["validation_prostate_dice"] for r in history], label="Validation")
    axes[1].set_ylabel("Patient-macro prostate Dice (native grid)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle("SYNTHETIC SOFTWARE SMOKE TEST — NOT HipMRI" if synthetic else "Training history")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        raise ValueError("Epochs, batch size, learning rate and threads must be positive")
    manifest = load_manifest(args.manifest)
    set_seed(args.seed, args.threads)
    device = device_from_name(args.device)
    train_data = HipMRIDataset(manifest, "train", args.size, augment=True)
    validation_data = HipMRIDataset(manifest, "validation", args.size)
    model = build_model(args.model, len(manifest["classes"]), args.base_channels).to(device)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "model": args.model, "num_classes": len(manifest["classes"]),
        "base_channels": args.base_channels, "size": list(args.size),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "weight_decay": 1e-5,
        "normalization": "per-slice z-score on all original pixels",
        "augmentation": "paired independent axis flips, deterministic per case/epoch",
    }
    write_json(out / "config.json", dict(config, environment=environment(device),
                                        synthetic=manifest["synthetic"],
                                        manifest_fingerprint=manifest["fingerprint"]))
    history, best, training_peak = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_data.epoch = epoch
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_loss, seen = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device)
            target, onehot = batch["target"].to(device), batch["one_hot"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = segmentation_loss(logits, target, onehot)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            seen += len(x)
        if device.type == "cuda":
            training_peak = max(training_peak, torch.cuda.max_memory_allocated(device))
        validation = evaluate(model, validation_data, device)
        score = validation["patient_macro"]["prostate"]["dice"]
        if score is None:
            raise ValueError("Undefined validation prostate Dice")
        record = {"epoch": epoch, "train_loss": total_loss / seen,
                  "validation_loss": validation["mean_loss_resized_grid"],
                  "validation_prostate_dice": score}
        history.append(record)
        print(json.dumps(dict(record, synthetic=manifest["synthetic"])), flush=True)
        if score > best:
            best = score
            torch.save({
                "schema_version": 1, "state_dict": model.state_dict(),
                "config": config, "classes": manifest["classes"],
                "manifest_fingerprint": manifest["fingerprint"],
                "synthetic": manifest["synthetic"], "epoch": epoch,
                "validation_prostate_dice": score,
            }, out / "best.pt")
            write_json(out / "best_validation.json", dict(validation, synthetic=manifest["synthetic"]))
        write_json(out / "history.json", history)
    synchronize(device)
    result = {
        "synthetic": manifest["synthetic"], "environment": environment(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_plus_validation_seconds": time.perf_counter() - started,
        "training_peak_cuda_allocated_bytes": training_peak if device.type == "cuda" else None,
        "best_validation_prostate_dice": best,
        "test_evaluated": False, "manifest_fingerprint": manifest["fingerprint"],
        "warning": "Synthetic results are software checks, not HipMRI evidence."
        if manifest["synthetic"] else "No test-set performance measured during fit.",
    }
    write_json(out / "training_summary.json", result)
    save_curves(history, out / "learning_curves.png", manifest["synthetic"])
    print(json.dumps(result, indent=2))
    return result


def load_checkpoint(path, manifest, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and manifest do not match")
    if checkpoint["classes"] != manifest["classes"] or checkpoint["synthetic"] != manifest["synthetic"]:
        raise ValueError("Checkpoint class vocabulary/data provenance mismatch")
    config = checkpoint["config"]
    model = build_model(config["model"], config["num_classes"], config["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--manifest", required=True)
    fitting.add_argument("--model", choices=("baseline", "improved"), required=True)
    fitting.add_argument("--out", required=True, help="New run directory")
    fitting.add_argument("--epochs", type=int, default=50)
    fitting.add_argument("--batch-size", type=int, default=8)
    fitting.add_argument("--base-channels", type=int, default=16)
    fitting.add_argument("--size", nargs=2, type=int, default=[256, 256])
    fitting.add_argument("--lr", type=float, default=1e-3)
    fitting.add_argument("--seed", type=int, default=3710)
    fitting.add_argument("--threads", type=int, default=1)
    fitting.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    testing = sub.add_parser("evaluate")
    testing.add_argument("--manifest", required=True)
    testing.add_argument("--checkpoint", required=True)
    testing.add_argument("--split", choices=("validation", "test"), required=True)
    testing.add_argument("--out", required=True)
    testing.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.command == "fit":
        fit(args)
    else:
        set_seed(3710)
        manifest = load_manifest(args.manifest)
        device = device_from_name(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, manifest, device)
        dataset = HipMRIDataset(manifest, args.split, checkpoint["config"]["size"])
        report = evaluate(model, dataset, device)
        report.update({"synthetic": manifest["synthetic"], "split": args.split,
                       "checkpoint_epoch": checkpoint["epoch"],
                       "manifest_fingerprint": manifest["fingerprint"],
                       "environment": environment(device)})
        write_json(args.out, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

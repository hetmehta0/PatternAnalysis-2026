"""Loss and count-based multiclass metrics; no empty-class score inflation."""
import torch
import torch.nn.functional as F


def segmentation_loss(logits, target, one_hot):
    probabilities = logits.softmax(dim=1)
    axes = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(axes)
    denominator = probabilities.sum(axes) + one_hot.sum(axes)
    # Foreground soft Dice plus CE. CE still penalizes false positives on empty masks.
    dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6))[1:].mean()
    return 0.5 * F.cross_entropy(logits, target) + 0.5 * dice_loss


def confusion_matrix(prediction, target, num_classes):
    prediction, target = prediction.reshape(-1).long(), target.reshape(-1).long()
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("Metric inputs must have equal nonempty shapes")
    if (prediction < 0).any() or (prediction >= num_classes).any() or (
            target < 0).any() or (target >= num_classes).any():
        raise ValueError("Metric label outside fixed class vocabulary")
    return torch.bincount(target * num_classes + prediction,
                          minlength=num_classes**2).reshape(num_classes, num_classes).cpu()


def scores(matrix, classes):
    matrix = matrix.double()
    result = {}
    for i, cls in enumerate(classes):
        tp = float(matrix[i, i])
        gt, pred = float(matrix[i].sum()), float(matrix[:, i].sum())
        denom = gt + pred
        result[cls["name"]] = {
            "dice": 2 * tp / denom if denom else None,
            "iou": tp / (denom - tp) if denom else None,
            "ground_truth_pixels": int(gt), "predicted_pixels": int(pred),
            "false_positive_pixels": int(pred - tp),
            "false_negative_pixels": int(gt - tp),
        }
    return result


def summarize_counts(patient_counts, classes):
    total = sum(patient_counts.values())
    per_patient = {patient: scores(counts, classes)
                   for patient, counts in sorted(patient_counts.items())}
    macro = {}
    for cls in classes:
        name = cls["name"]
        macro[name] = {}
        for metric in ("dice", "iou"):
            values = [p[name][metric] for p in per_patient.values()
                      if p[name][metric] is not None]
            macro[name][metric] = sum(values) / len(values) if values else None
            macro[name]["contributing_patients"] = len(values)
    return {"pooled": scores(total, classes), "patient_macro": macro,
            "per_patient": per_patient,
            "empty_class_policy": "Both absent => null/excluded; false positives on absent truth => 0."}

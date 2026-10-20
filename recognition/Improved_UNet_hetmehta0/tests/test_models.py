import pytest
import torch

from metrics import confusion_matrix, scores, segmentation_loss, summarize_counts
from modules import build_model

torch.set_num_threads(1)


@pytest.mark.parametrize("architecture", ["baseline", "improved"])
def test_forward_backward_odd_shape_and_all_heads(architecture):
    torch.manual_seed(3710)
    model = build_model(architecture, 3, 2)
    x = torch.randn(2, 1, 33, 35)
    y = torch.randint(0, 3, (2, 33, 35))
    onehot = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float()
    prediction = model(x)
    assert prediction.shape == (2, 3, 33, 35)
    loss = segmentation_loss(prediction, y, onehot)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
    if architecture == "improved":
        assert all(h.weight.grad.abs().sum() > 0 for h in model.segmentation_heads)


def test_metrics_hand_computed_and_empty_policy():
    classes = [{"name": "background"}, {"name": "prostate"}, {"name": "absent"}]
    # Prostate: TP=1, FP=1, FN=1 -> Dice=.5, IoU=1/3.
    target = torch.tensor([0, 1, 1, 0])
    pred = torch.tensor([0, 0, 1, 1])
    counts = confusion_matrix(pred, target, 3)
    value = scores(counts, classes)
    assert value["prostate"]["dice"] == 0.5
    assert value["prostate"]["iou"] == pytest.approx(1 / 3)
    assert value["absent"]["dice"] is None
    assert scores(confusion_matrix(target, target, 3), classes)["prostate"]["dice"] == 1
    result = summarize_counts({"patient_A": counts, "patient_B": counts}, classes)
    assert result["patient_macro"]["prostate"]["dice"] == .5


def test_false_positive_on_empty_truth_is_zero():
    counts = confusion_matrix(torch.ones(4), torch.zeros(4), 2)
    assert scores(counts, [{"name": "background"}, {"name": "prostate"}])["prostate"]["dice"] == 0

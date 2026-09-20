import torch

from cfwam.losses import counterfactual_loss
from cfwam.model import CounterfactualAttributor
from cfwam.decision import decode_attribution
from cfwam.runtime import AbstainPolicy


def test_model_and_preregistered_losses_support_backward():
    model = CounterfactualAttributor(residual_dim=12, node_feature_dim=7, edge_type_count=5)
    output = model(
        residual=torch.randn(2, 12),
        node_features=torch.randn(2, 10, 7),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        edge_type=torch.tensor([0, 1, 2, 3, 4]),
    )
    losses = counterfactual_loss(
        output["cause_logits"], output["mask_logits"], output["hypothesis_residuals"],
        residual_target=torch.randn(2, 12), cause_target=torch.tensor([1, 3]),
        mask_target=torch.zeros(2, 10),
    )
    losses["total"].backward()
    assert set(losses) == {"total", "cause", "mask", "counterfactual"}
    assert torch.isfinite(losses["total"])


def test_decoder_uses_unknown_safety_exit():
    outputs = {
        "cause_logits": torch.zeros(1, 5),
        "mask_logits": torch.zeros(1, 3),
        "hypothesis_residuals": torch.zeros(1, 5, 4),
    }
    result = decode_attribution(
        outputs, torch.zeros(1, 4), ["a", "b", "c"], AbstainPolicy(0.9, 1.8, 5.0)
    )
    assert result.abstained
    assert result.cause.value == "unknown"

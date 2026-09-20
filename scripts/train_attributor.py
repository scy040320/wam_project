"""Train the lightweight attribution GAT on grouped pre-extracted records."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cfwam.losses import counterfactual_loss
from cfwam.model import CounterfactualAttributor


def collate(batch):
    keys = ("residual", "node_features", "edge_index", "edge_type", "cause", "mask")
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def run_epoch(model, loader, optimizer=None):
    total, count = 0.0, 0
    for batch in loader:
        outputs = model(batch["residual"], batch["node_features"], batch["edge_index"], batch["edge_type"])
        losses = counterfactual_loss(
            outputs["cause_logits"], outputs["mask_logits"], outputs["hypothesis_residuals"],
            batch["residual"], batch["cause"], batch["mask"],
        )
        if optimizer:
            optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
        total += float(losses["total"].detach()) * len(batch["cause"]); count += len(batch["cause"])
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train = torch.load(args.train, weights_only=False)
    validation = torch.load(args.validation, weights_only=False)
    train_states = {x["initial_state_id"] for x in train}; val_states = {x["initial_state_id"] for x in validation}
    if not train or not validation or train_states & val_states:
        raise ValueError("empty split or initial state leaked across grouped train/validation split")
    first = train[0]
    model = CounterfactualAttributor(first["residual"].numel(), first["node_features"].shape[-1], 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=32, shuffle=False, collate_fn=collate)
    best = (float("inf"), None)
    for epoch in range(args.epochs):
        model.train(); train_loss = run_epoch(model, train_loader, optimizer)
        model.eval()
        with torch.no_grad(): val_loss = run_epoch(model, val_loader)
        print(f"epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best[0]: best = (val_loss, {k: v.cpu() for k, v in model.state_dict().items()})
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], output / "attributor_best.pt")
    (output / "training_manifest.json").write_text(json.dumps({
        "best_validation_loss": best[0], "train_initial_states": sorted(train_states),
        "validation_initial_states": sorted(val_states), "loss": "cause + mask + 0.5 * counterfactual",
        "thresholds": "calibrate once on validation and freeze before final test",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

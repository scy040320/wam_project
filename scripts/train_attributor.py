"""Train the lightweight attribution GAT on grouped pre-extracted records."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cfwam.losses import counterfactual_loss
from cfwam.model import CounterfactualAttributor


def collate(batch):
    static = ("edge_index", "edge_type")
    stacked = ("residual", "node_features", "edge_mask", "node_valid", "cause", "mask")
    payload = {key: torch.stack([item[key] for item in batch]) for key in stacked}
    for key in static:
        if not all(torch.equal(item[key], batch[0][key]) for item in batch[1:]):
            raise ValueError(f"{key} must be canonical and identical across a batch")
        payload[key] = batch[0][key]
    return payload


def run_epoch(model, loader, device, optimizer=None):
    total, count = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch["residual"], batch["node_features"], batch["edge_index"], batch["edge_type"], batch["edge_mask"], batch["node_valid"])
        losses = counterfactual_loss(
            outputs["cause_logits"], outputs["mask_logits"], outputs["hypothesis_residuals"],
            batch["residual"], batch["cause"], batch["mask"], node_valid=batch["node_valid"],
        )
        if optimizer:
            optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
        total += float(losses["total"].detach()) * len(batch["cause"]); count += len(batch["cause"])
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train = torch.load(args.train, map_location="cpu", weights_only=False)
    validation = torch.load(args.validation, map_location="cpu", weights_only=False)
    train_states = {x["initial_state_id"] for x in train}; val_states = {x["initial_state_id"] for x in validation}
    if not train or not validation or train_states & val_states:
        raise ValueError("empty split or initial state leaked across grouped train/validation split")
    first = train[0]
    model = CounterfactualAttributor(first["residual"].numel(), first["node_features"].shape[-1], 5).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=32, shuffle=False, collate_fn=collate)
    best = (float("inf"), None, -1)
    bad_epochs, history = 0, []
    for epoch in range(args.epochs):
        model.train(); train_loss = run_epoch(model, train_loader, args.device, optimizer)
        model.eval()
        with torch.no_grad(): val_loss = run_epoch(model, val_loader, args.device)
        print(f"epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best[0] - args.min_delta:
            best = (val_loss, {k: v.cpu() for k, v in model.state_dict().items()}, epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early_stop epoch={epoch:03d} best_epoch={best[2]:03d} patience={args.patience}")
                break
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], output / "attributor_best.pt")
    (output / "training_manifest.json").write_text(json.dumps({
        "best_validation_loss": best[0], "best_epoch": best[2], "max_epochs": args.epochs, "seed": args.seed,
        "early_stopping_patience": args.patience, "min_delta": args.min_delta,
        "train_initial_states": sorted(train_states),
        "validation_initial_states": sorted(val_states), "loss": "cause + mask + 0.5 * counterfactual",
        "thresholds": "calibrate once on validation and freeze before final test",
    }, indent=2), encoding="utf-8")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

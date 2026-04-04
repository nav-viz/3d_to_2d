"""
training/train.py
=================
Fixed training script for MFCAD++ with variable-sized graphs.
Supports HGT + GraphSAGE, dynamic feature dim (5), and proper variable-node batching.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from loguru import logger

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.mfcad_h5_dataset import make_dataloader, NUM_CLASSES
from models.hgt import CADHGT
from models.graphsage import build_model as build_graphsage

try:
    from sklearn.metrics import f1_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_f1(preds: list[int], labels: list[int], n_cls: int) -> float:
    if _SKLEARN:
        return f1_score(labels, preds, average="micro", zero_division=0)
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / max(len(labels), 1)


def _make_scheduler(optimiser, cfg, num_epochs):
    sched = cfg["training"].get("scheduler", "cosine")
    if sched == "cosine":
        return CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1e-6)
    elif sched == "step":
        return StepLR(optimiser, step_size=cfg["training"].get("step_size", 30),
                      gamma=cfg["training"].get("gamma", 0.1))
    return None


# ---------------------------------------------------------------------------
# Training Loop (Fixed for variable graph sizes)
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    model_type: str,
) -> None:
    tc = cfg["training"]
    lr = tc["learning_rate"]
    wd = tc["weight_decay"]
    epochs = tc["epochs"]
    patience = tc["patience"]
    log_every = tc["log_every"]

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    log_dir = Path(cfg["paths"]["logs"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")   # Change to "cuda" later when DGL CUDA is installed
    logger.info(f"Training on device: {device}")
    model = model.to(device)

    # === Optimiser ===
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = _make_scheduler(optimiser, cfg, epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    no_improve = 0
    log_path = log_dir / f"train_{model_type}.jsonl"
    ckpt_path = ckpt_dir / f"best_{model_type}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        # === Train Loop - handles variable batch format ===
        for batch_items in train_loader:
            # batch_items is a list of (g, labels) tuples
            for item in batch_items if isinstance(batch_items, list) else [batch_items]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    g, labels = item
                else:
                    continue

                import dgl
                if not isinstance(g, dgl.DGLHeteroGraph):
                    continue

                g = g.to(device)
                labels = labels.to(device).view(-1)

                logits = model(g)
                loss = criterion(logits, labels)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        train_f1 = compute_f1(all_preds, all_labels, NUM_CLASSES)

        # === Validation ===
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss / max(len(train_loader), 1):.4f}  "
                f"train_f1={train_f1:.4f}  "
                f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  "
                f"({elapsed:.1f}s)"
            )

        # Log and save best
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss / max(len(train_loader), 1), 6),
            "train_f1": round(train_f1, 6),
            "val_loss": round(val_loss, 6),
            "val_f1": round(val_f1, 6),
        }
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_f1": val_f1,
                "model_type": model_type,
                "cfg": cfg,
            }, ckpt_path)
            logger.success(f"Checkpoint saved → {ckpt_path} (val_f1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

    logger.success(f"Training complete. Best val_f1={best_val_f1:.4f}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    import dgl
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_items in loader:
        for item in batch_items if isinstance(batch_items, list) else [batch_items]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                g, labels = item
            else:
                continue
            if not isinstance(g, dgl.DGLHeteroGraph):
                continue

            g = g.to(device)
            labels = labels.to(device).view(-1)
            logits = model(g)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = total_loss / max(len(loader), 1)
    f1 = compute_f1(all_preds, all_labels, NUM_CLASSES)
    return mean_loss, f1


# ---------------------------------------------------------------------------
# Model Building (with dynamic feat_dim)
# ---------------------------------------------------------------------------

def build_model(cfg: dict, feat_dim: int):
    """Build the correct model based on type - fixed for stable CADHGT"""
    model_type = cfg["model"]["type"]
    from models.hgt import CADHGT   # force fresh import
    if model_type == "hgt":
        logger.info(f"Building CADHGT (stable) with in_feats={feat_dim}")
        return CADHGT(
            in_feats=feat_dim,
            hidden_dim=cfg["model"].get("hidden_dim", 256),
            out_feats=NUM_CLASSES,
            num_layers=cfg["model"].get("num_layers", 3),
            dropout=cfg["model"].get("dropout", 0.2),
        )
    else:
        # GraphSAGE fallback
        logger.info(f"Building GraphSAGE with in_feats={feat_dim}")
        return build_graphsage(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train GNN on MFCAD++ dataset")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model_type", default=None, choices=["hgt", "graphsage"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main():
    import yaml
    args = _parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # CLI overrides
    if args.model_type:
        cfg["model"]["type"] = args.model_type
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    model_type = cfg["model"]["type"]
    set_seed(cfg["training"]["seed"])

    logger.remove()
    logger.add(sys.stderr, level=cfg["logging"]["level"], colorize=cfg["logging"].get("colorize", True))

    # Data loaders
    bs = cfg["training"]["batch_size"]
    train_loader = make_dataloader(
        cfg["paths"]["train_h5"], batch_size=bs, shuffle=True,
        normalise=cfg["graph"]["normalise_features"], max_samples=args.max_samples
    )
    val_loader = make_dataloader(
        cfg["paths"]["val_h5"], batch_size=bs, shuffle=False,
        normalise=cfg["graph"]["normalise_features"], max_samples=args.max_samples
    )

    # Get actual feature dimension
    from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
    temp_ds = MFCAD_H5_Dataset(cfg["paths"]["train_h5"], normalise=False, max_samples=1)
    feat_dim = temp_ds.feat_dim
    del temp_ds

    # Build model with correct input size
    model = build_model(cfg, feat_dim=feat_dim)

    # Train
    train(model, train_loader, val_loader, cfg, model_type)


if __name__ == "__main__":
    main()
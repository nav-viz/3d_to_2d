"""
training/train.py
=================
Improved training script with fixed batching for variable-sized graphs + support for 10-dim features.
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
from utils.logger import logger

import torch.nn.functional as F

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

class FocalLoss(nn.Module):
    """Focal Loss for better handling of class imbalance in multi-class setting."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
    
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


# training/train.py  —  REPLACE train() and evaluate()

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    model_type: str,
) -> None:
    """Stable training with Focal Loss."""
    tc = cfg["training"]
    lr = float(tc.get("learning_rate", 0.0005))
    wd = float(tc.get("weight_decay", 1e-4))
    epochs = int(tc.get("epochs", 150))
    patience = int(tc.get("patience", 20))
    log_every = int(tc.get("log_every", 5))

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{model_type}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    no_improve = 0
    min_epochs = 10

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                g, labels = batch
            else:
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

        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        if epoch % log_every == 0 or val_f1 > best_val_f1:
            logger.info(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss / max(len(train_loader), 1):.4f}  "
                f"train_f1={train_f1:.4f}  "
                f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  "
                f"({elapsed:.1f}s)"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": val_f1,
                "model_type": model_type,
                "cfg": cfg,
                "in_feats": model.input_proj.in_features, 
            }, ckpt_path)
            logger.success(f"✓ Best checkpoint saved (val_f1={val_f1:.4f})")
        else:
            no_improve += 1
            if epoch >= min_epochs and no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

    logger.success(f"Training complete. Best val_f1 = {best_val_f1:.4f}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Clean evaluation with proper batch unpacking."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            g, labels = batch
        else:
            logger.warning(f"Unexpected val batch format: {type(batch)}")
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


def build_model(cfg: dict, feat_dim: int):
    model_type = cfg["model"]["type"]
    if model_type == "hgt":
        logger.info(f"Building CADHGT (stable) with in_feats={feat_dim}")
        return CADHGT(
            in_feats=feat_dim,
            hidden_dim=cfg["model"].get("hidden_dim", 768),   # Increased default
            out_feats=NUM_CLASSES,
            num_layers=cfg["model"].get("num_layers", 8),     # Increased default
            dropout=cfg["model"].get("dropout", 0.2),
        )
    else:
        logger.info(f"Building GraphSAGE with in_feats={feat_dim}")
        return build_graphsage(cfg)


def _parse_args():
    p = argparse.ArgumentParser(description="Train GNN on MFCAD++ dataset")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model_type", default=None, choices=["hgt", "graphsage"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main():
    import yaml
    args = _parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # === FIX: Force correct types ===
    if args.model_type:
        cfg["model"]["type"] = args.model_type

    if args.epochs:
        cfg["training"]["epochs"] = int(args.epochs)

    if args.max_samples is not None:
        pass  # already handled in dataloader

    # Ensure numeric types
    tc = cfg["training"]
    tc["learning_rate"] = float(tc.get("learning_rate", 0.001))
    tc["weight_decay"]  = float(tc.get("weight_decay", 1e-5))
    tc["epochs"]        = int(tc.get("epochs", 50))
    tc["patience"]      = int(tc.get("patience", 12))
    tc["batch_size"]    = int(tc.get("batch_size", 4))

    model_type = cfg["model"]["type"]
    set_seed(cfg["training"]["seed"])

    logger.remove()
    logger.add(sys.stderr, level=cfg["logging"]["level"], colorize=True)

    bs = cfg["training"]["batch_size"]

    train_loader = make_dataloader(
        cfg["paths"]["train_h5"],
        batch_size=bs,
        shuffle=True,
        normalise=cfg["graph"]["normalise_features"],
        max_samples=args.max_samples,
    )
    val_loader = make_dataloader(
        cfg["paths"]["val_h5"],
        batch_size=bs,
        shuffle=False,
        normalise=cfg["graph"]["normalise_features"],
        max_samples=args.max_samples,
    )

    # Get actual feature dimension
    from graph.mfcad_h5_dataset import MFCAD_H5_Dataset
    temp_ds = MFCAD_H5_Dataset(cfg["paths"]["train_h5"], normalise=False, max_samples=1)
    feat_dim = temp_ds.feat_dim
    del temp_ds

    model = build_model(cfg, feat_dim=feat_dim)

    train(model, train_loader, val_loader, cfg, model_type)


if __name__ == "__main__":
    main()
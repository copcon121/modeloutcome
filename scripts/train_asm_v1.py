#!/usr/bin/env python3
"""
ASM Training Script v1
======================
Train Auction State Model (GRU-based) for VA-shift prediction.

Reference: PLAN_AuctionStateModel_v1.md

Usage:
    python scripts/train_asm_v1.py
    python scripts/train_asm_v1.py --hidden_dim 128 --num_layers 2 --lr 0.0005
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
TRAIN_PATH = Path("output/asm_dataset_v1/asm_dataset_v1_train.pt")
VAL_PATH = Path("output/asm_dataset_v1/asm_dataset_v1_val.pt")
STATS_PATH = Path("output/asm_dataset_v1/asm_dataset_v1_stats.json")
OUTPUT_DIR = Path("output/asm_models_v1")

# Label mapping
LABEL_NAMES = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}
NUM_CLASSES = 3

# Default hyperparameters
DEFAULT_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 50,
    "patience": 7,
    "weight_method": "sqrt_inv",  # 'inv' or 'sqrt_inv'
}


# ==============================================================================
# DATASET
# ==============================================================================


class ASMDataset(Dataset):
    """ASM Dataset for VA-shift prediction."""

    def __init__(self, data_path: Path):
        data = torch.load(data_path, weights_only=False)
        self.X = data["X"]  # (N, 60, 100)
        self.y = data["y"]  # (N,)
        self.meta = data.get("meta", None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# MODEL
# ==============================================================================


class ASMGRUClassifier(nn.Module):
    """GRU-based classifier for Auction State Model."""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classifier head
        gru_output_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # GRU output: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        _, h_n = self.gru(x)

        # Take last layer hidden states from both directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2, :, :]  # (batch, hidden_dim)
            h_backward = h_n[-1, :, :]  # (batch, hidden_dim)
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1, :, :]  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(h_final)
        return logits


# ==============================================================================
# METRICS
# ==============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N, 3) for [UP, DOWN, NEUTRAL]

    Returns:
        Dict with accuracy, macro_f1, per-class F1, AUCs
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(np.mean(y_true == y_pred))

    # Macro F1
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    metrics["f1_UP"] = float(f1_per_class[0])
    metrics["f1_DOWN"] = float(f1_per_class[1])
    metrics["f1_NEUTRAL"] = float(f1_per_class[2])

    # AUC: UP vs rest
    try:
        y_up_binary = (y_true == 0).astype(int)
        metrics["auc_UP"] = float(roc_auc_score(y_up_binary, y_probs[:, 0]))
    except ValueError:
        metrics["auc_UP"] = 0.5

    # AUC: DOWN vs rest
    try:
        y_down_binary = (y_true == 1).astype(int)
        metrics["auc_DOWN"] = float(roc_auc_score(y_down_binary, y_probs[:, 1]))
    except ValueError:
        metrics["auc_DOWN"] = 0.5

    # AUC: SHIFT (UP or DOWN) vs NEUTRAL
    try:
        y_shift_binary = (y_true != 2).astype(int)  # 1 if UP or DOWN, 0 if NEUTRAL
        p_shift = y_probs[:, 0] + y_probs[:, 1]  # P(UP) + P(DOWN)
        metrics["auc_SHIFT"] = float(roc_auc_score(y_shift_binary, p_shift))
    except ValueError:
        metrics["auc_SHIFT"] = 0.5

    return metrics


# ==============================================================================
# TRAINING
# ==============================================================================


def compute_class_weights(stats: Dict, method: str = "sqrt_inv") -> torch.Tensor:
    """Compute class weights from label distribution."""
    train_dist = stats["train"]["label_distribution"]

    # Order: UP=0, DOWN=1, NEUTRAL=2
    counts = np.array(
        [train_dist.get("UP", 1), train_dist.get("DOWN", 1), train_dist.get("NEUTRAL", 1)],
        dtype=np.float32,
    )

    if method == "inv":
        weights = 1.0 / counts
    elif method == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    else:
        weights = np.ones(3)

    # Normalize
    weights = weights / weights.sum() * 3

    print(f"Class weights ({method}): UP={weights[0]:.3f}, DOWN={weights[1]:.3f}, NEUTRAL={weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def train(config: Dict):
    """Main training function."""
    print("=" * 60)
    print("ASM Training v1")
    print("=" * 60)
    print(f"Config: {config}")
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load stats for class weights
    with open(STATS_PATH) as f:
        stats = json.load(f)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ASMDataset(TRAIN_PATH)
    val_dataset = ASMDataset(VAL_PATH)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Model
    input_dim = train_dataset.X.shape[2]
    model = ASMGRUClassifier(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=config["dropout"],
    ).to(device)

    print(f"\nModel: ASMGRUClassifier")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Dropout: {config['dropout']}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Loss with class weights
    class_weights = compute_class_weights(stats, config["weight_method"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_metric = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = OUTPUT_DIR / "asm_gru64_best.pt"

    for epoch in range(1, config["max_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Monitor metric (macro F1 or AUC_SHIFT)
        monitor_metric = val_metrics["auc_SHIFT"]

        # Print epoch summary
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.3f} | "
            f"F1: {val_metrics['macro_f1']:.3f} | "
            f"AUC_SHIFT: {val_metrics['auc_SHIFT']:.3f} | "
            f"AUC_UP: {val_metrics['auc_UP']:.3f} | "
            f"AUC_DOWN: {val_metrics['auc_DOWN']:.3f}"
        )

        # Check for improvement
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": best_metrics,
                },
                best_model_path,
            )
            print(f"  → New best! Saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={config['patience']})")
                break

    # Final summary
    print("\n" + "=" * 60)
    print("[BEST RESULT]")
    print("=" * 60)
    print(f"  Epoch:     {best_epoch}")
    print(f"  Macro F1:  {best_metrics['macro_f1']:.4f}")
    print(f"  F1_UP:     {best_metrics['f1_UP']:.4f}")
    print(f"  F1_DOWN:   {best_metrics['f1_DOWN']:.4f}")
    print(f"  F1_NEUTRAL:{best_metrics['f1_NEUTRAL']:.4f}")
    print(f"  AUC_UP:    {best_metrics['auc_UP']:.4f}")
    print(f"  AUC_DOWN:  {best_metrics['auc_DOWN']:.4f}")
    print(f"  AUC_SHIFT: {best_metrics['auc_SHIFT']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")

    # Save metrics JSON
    metrics_path = OUTPUT_DIR / "asm_gru64_best_metrics.json"
    best_metrics["config"] = config
    best_metrics["created_at"] = datetime.now().isoformat()
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Model saved to:   {best_model_path}")


# ==============================================================================
# MAIN
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASM v1 Model")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num_layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument(
        "--weight_method",
        type=str,
        default=DEFAULT_CONFIG["weight_method"],
        choices=["inv", "sqrt_inv", "none"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_method": args.weight_method,
    }

    train(config)


if __name__ == "__main__":
    main()

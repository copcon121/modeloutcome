"""
Train Sequence Quality Model (GRU) - v2

Supports multiple configs, focal loss, LR warmup/cosine, and unique model saves.
Data loading/splits remain unchanged.
"""

import sys
import os
from pathlib import Path
from typing import Callable, Dict, List
import json
import argparse
import shutil

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase6_seq_quality.dataset_seq import QualitySequenceDataset, compute_normalizer_stats
from phase6_seq_quality.model_seq import QualitySeqGRU

# Paths (env overridable)
ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("PHASE4_DIR_OVERRIDE", ROOT / "output/phase4_quality"))
OUTPUT_DIR = Path(os.getenv("PHASE6_OUTPUT_DIR", ROOT / "output/phase6_seq_quality"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_loss_fn(use_focal: bool, gamma: float, pos_weight: torch.Tensor) -> Callable:
    """
    Build loss function with optional focal term on top of weighted BCE.
    """
    def bce_loss(logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )

    if not use_focal:
        def loss_fn(logits, targets):
            return bce_loss(logits, targets).mean()
        return loss_fn

    gamma = float(gamma)

    def loss_fn(logits, targets):
        bce = bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0, probs, 1.0 - probs)
        focal_weight = (1.0 - pt).pow(gamma)
        return (focal_weight * bce).mean()

    return loss_fn


def make_warmup_cosine_scheduler(optimizer, num_epochs: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(num_epochs * warmup_ratio))

    def lr_lambda(current_epoch):
        if current_epoch < warmup_steps:
            return float(current_epoch + 1) / float(warmup_steps)
        progress = (current_epoch - warmup_steps) / max(1, (num_epochs - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_config(config: Dict, normalizer_stats: Dict, promote_config: str = None) -> Dict:
    """
    Train a single config and return summary metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = DATA_DIR / "dataset_p2_quality_v1_train.pt"
    val_path = DATA_DIR / "dataset_p2_quality_v1_val.pt"

    print(f"\n=== Training config: {config['name']} ===")
    print(f"  lr={config['lr']} use_focal={config['use_focal_loss']} gamma={config['focal_gamma']}")
    print(f"  hidden_dim={config['hidden_dim']} dropout={config['dropout']} layers={config['num_layers']}")
    print(f"  batch_size={config['batch_size']} max_epochs={config['max_epochs']} patience={config['patience']}")
    print(f"  device={device}")

    # Datasets / loaders
    train_dataset = QualitySequenceDataset(train_path, normalizer_stats)
    val_dataset = QualitySequenceDataset(val_path, normalizer_stats)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # Class balance
    y_train = train_dataset.y
    pos_count = (y_train == 1).sum().item()
    neg_count = (y_train == 0).sum().item()
    pos_weight_val = neg_count / max(1, pos_count)
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    print(f"  Class balance: KEEP={pos_count}, DROP={neg_count}, pos_weight={pos_weight_val:.2f}")

    # Model - get input_dim from data
    input_dim = train_dataset.X.shape[-1]  # Get feature dimension from data
    model = QualitySeqGRU(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Loss / optimizer / scheduler
    loss_fn = make_loss_fn(
        use_focal=config["use_focal_loss"],
        gamma=config["focal_gamma"],
        pos_weight=pos_weight,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer, num_epochs=config["max_epochs"], warmup_ratio=config["warmup_ratio"]
    )

    best_f1 = 0.0
    best_auc = 0.0
    patience_counter = 0
    training_log: List[Dict] = []

    # Unique model path for this config
    model_path_unique = OUTPUT_DIR / f"model_seq_quality_{config['name']}.pt"
    # Default/promoted model path (optional)
    model_path_default = OUTPUT_DIR / "model_seq_quality_v1_best.pt"

    for epoch in range(config["max_epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X = batch["X_seq"].to(device)
            side = batch["side"].to(device)
            y = batch["y"].float().to(device)

            logits = model(X, side).squeeze(-1)  # Keep batch dim
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X_seq"].to(device)
                side = batch["side"].to(device)
                y = batch["y"].float().to(device)

                logits = model(X, side).squeeze(-1)  # Keep batch dim
                loss = loss_fn(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                # Handle scalar case (batch size 1)
                if probs.ndim == 0:
                    probs = [probs.item()]
                    preds = [preds.item()]

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0

        log_entry = {
            "config": config["name"],
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_auc": auc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_log.append(log_entry)

        print(
            f"  Epoch {epoch+1:2d}: train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}"
        )

        # Early stopping on F1
        if f1 > best_f1:
            best_f1 = f1
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), model_path_unique)
            # Promote to default if requested
            if promote_config and promote_config == config["name"]:
                torch.save(model.state_dict(), model_path_default)
            print(f"    [BEST] F1={f1:.4f}, AUC={auc:.4f} - Saved model -> {model_path_unique}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"    Early stopping at epoch {epoch+1} (patience={config['patience']})")
                break

        scheduler.step()

    # Save training log per config
    log_df = pd.DataFrame(training_log)
    log_path = OUTPUT_DIR / f"training_log_seq_{config['name']}.csv"
    log_df.to_csv(log_path, index=False)
    print(f"  Saved training log: {log_path}")

    return {
        "config": config["name"],
        "best_f1": best_f1,
        "best_auc": best_auc,
        "epochs": len(training_log),
        "log_path": str(log_path),
        "model_path": str(model_path_unique),
    }


def promote_from_summary(summary_path: Path, promote_config: str, target_path: Path):
    if not summary_path.exists():
        print(f"[WARN] Summary not found: {summary_path}")
        return False
    data = json.loads(summary_path.read_text())
    for entry in data:
        if entry.get("config") == promote_config:
            src = Path(entry["model_path"])
            if not src.exists():
                print(f"[WARN] Model to promote not found: {src}")
                return False
            shutil.copyfile(src, target_path)
            print(f"[PROMOTE] {promote_config} -> {target_path}")
            return True
    print(f"[WARN] Config {promote_config} not found in summary.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Train Sequence Quality Model v2")
    parser.add_argument("--run-all", action="store_true", help="Train all configs")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated config names to train (overrides --run-all if set)")
    parser.add_argument("--promote-config", type=str, default="v2_lr1e-3_focal1.5",
                        help="Config name to promote to model_seq_quality_v1_best.pt")
    parser.add_argument("--no-train", action="store_true", help="Skip training and only promote from summary")
    args = parser.parse_args()
    print("=" * 80)
    print("PHASE 6: TRAIN SEQUENCE QUALITY MODEL (v2)")
    print("=" * 80)

    # Compute normalizer (always ensure fresh stats exist)
    print("\n[1/4] Computing normalizer stats...")
    train_path = DATA_DIR / "dataset_p2_quality_v1_train.pt"
    normalizer_stats = compute_normalizer_stats(train_path)
    normalizer_path = OUTPUT_DIR / "normalizer_stats_seq.pt"
    torch.save(normalizer_stats, normalizer_path)
    print(f"  Saved: {normalizer_path}")

    # Base config shared fields - optimized for small dataset
    base_config = {
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 4,  # Early stopping after 4 epochs no improvement
        "weight_decay": 1e-3,  # Strong L2 regularization
        "warmup_ratio": 0.1,
        "num_layers": 1,
    }
    
    # 3 configs to compare - all with smaller models to reduce overfitting
    configs = [
        {
            "name": "v2_1_gru64_dropout0.3_wd1e-3",
            "lr": 1e-3,
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "hidden_dim": 64,
            "dropout": 0.3,
            **base_config,
        },
        {
            "name": "v2_2_gru64_dropout0.4_wd1e-3",
            "lr": 1e-3,
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "hidden_dim": 64,
            "dropout": 0.4,
            **base_config,
        },
        {
            "name": "v2_3_gru128_dropout0.3_wd1e-3",
            "lr": 1e-3,
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "hidden_dim": 128,
            "dropout": 0.3,
            **base_config,
        },
    ]

    if args.no_train:
        print("\n[SKIP TRAIN] --no-train specified; only promoting model if requested.")
        summary_path = OUTPUT_DIR / "training_summary_seq_v2.json"
        if args.promote_config:
            promote_from_summary(summary_path, args.promote_config, OUTPUT_DIR / "model_seq_quality_v1_best.pt")
        return

    # Control: select configs
    if args.configs:
        selected = set([c.strip() for c in args.configs.split(",") if c.strip()])
        run_list = [cfg for cfg in configs if cfg["name"] in selected]
        if not run_list:
            print(f"[WARN] No matching configs for: {selected}. Nothing to train.")
            return
    else:
        run_list = configs if args.run_all else [configs[0]]

    summaries = []
    for cfg in run_list:
        summaries.append(train_one_config(cfg, normalizer_stats, promote_config=args.promote_config))

    # Save summary JSON with unique model paths
    summary_path = OUTPUT_DIR / "training_summary_seq_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Optional promotion after training using summary (to ensure latest file)
    if args.promote_config:
        promote_from_summary(summary_path, args.promote_config, OUTPUT_DIR / "model_seq_quality_v1_best.pt")


if __name__ == "__main__":
    main()

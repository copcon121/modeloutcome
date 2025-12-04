"""
Training loop for STATE-ENC v1

Plain PyTorch training without external frameworks.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from ..src.config import StateEncTrainConfig, StateEncModelConfig
from ..src.model.state_enc_model import StateEncModel
from ..src.dataset_encoder import StateEncDataset, collate_fn, create_dataloaders
from .losses import MultiHeadLoss
from .eval_metrics import MetricsAccumulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class StateEncTrainer:
    """
    Trainer for STATE-ENC model.
    
    Features:
    - Multi-head loss training
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(self,
                 model: StateEncModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: StateEncTrainConfig,
                 model_config: StateEncModelConfig):
        """
        Args:
            model: StateEncModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            model_config: Model configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_config = model_config
        
        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Loss function
        self.loss_fn = MultiHeadLoss(loss_weights=config.loss_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.max_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_epochs / config.max_epochs,
            anneal_strategy='cos'
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode="max"  # Maximize validation accuracy
        )
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dir_acc": [],
            "val_regime_acc": [],
            "lr": []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        accumulator = MetricsAccumulator()
        
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            X = batch["X"].to(self.device)
            targets = {
                "future_dir_5": batch["future_dir_5"].to(self.device),
                "future_return_5": batch["future_return_5"].to(self.device),
                "regime_hint": batch["regime_hint"].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X)
            
            # Compute loss
            losses = self.loss_fn(outputs, targets)
            loss = losses["loss_total"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate metrics
            accumulator.update(outputs, targets, losses)
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{total_batches}] "
                    f"Loss: {loss.item():.4f} LR: {current_lr:.6f}"
                )
        
        return accumulator.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        accumulator = MetricsAccumulator()
        
        for batch in self.val_loader:
            X = batch["X"].to(self.device)
            targets = {
                "future_dir_5": batch["future_dir_5"].to(self.device),
                "future_return_5": batch["future_return_5"].to(self.device),
                "regime_hint": batch["regime_hint"].to(self.device)
            }
            
            outputs = self.model(X)
            losses = self.loss_fn(outputs, targets)
            
            accumulator.update(outputs, targets, losses)
        
        return accumulator.compute()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.model_config.__dict__ if hasattr(self.model_config, '__dict__') else self.model_config
        }
        
        # Save periodic checkpoint
        if epoch % self.config.save_every_n_epochs == 0:
            path = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            best_path = self.run_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training summary dict
        """
        logger.info("=" * 60)
        logger.info("Starting STATE-ENC v1 Training")
        logger.info(f"Output directory: {self.run_dir}")
        logger.info(f"Max epochs: {self.config.max_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Get current LR
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update history
            self.history["train_loss"].append(train_metrics.get("avg_loss_total", 0))
            self.history["val_loss"].append(val_metrics.get("avg_loss_total", 0))
            self.history["val_dir_acc"].append(val_metrics.get("dir_accuracy", 0))
            self.history["val_regime_acc"].append(val_metrics.get("regime_accuracy", 0))
            self.history["lr"].append(current_lr)
            
            # Check if best
            val_acc = val_metrics.get("dir_accuracy", 0)
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} "
                f"[{epoch_time:.1f}s] "
                f"Train Loss: {train_metrics.get('avg_loss_total', 0):.4f} "
                f"Val Loss: {val_metrics.get('avg_loss_total', 0):.4f} "
                f"Val Dir Acc: {val_acc:.4f} "
                f"Val Regime Acc: {val_metrics.get('regime_accuracy', 0):.4f} "
                f"{'*BEST*' if is_best else ''}"
            )
            
            # Early stopping
            if self.early_stopping(val_acc):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Save training history
        history_path = self.run_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Summary
        summary = {
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "total_epochs": epoch,
            "total_time_seconds": total_time,
            "run_dir": str(self.run_dir)
        }
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best val accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info("=" * 60)
        
        return summary


def train_state_enc(config_path: str) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        config_path: Path to training config JSON
        
    Returns:
        Training summary
    """
    # Load configs
    train_config = StateEncTrainConfig.from_json(config_path)
    model_config = StateEncModelConfig.from_json(train_config.model_config_path)
    
    # Set seed
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    
    # Create dataloaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=train_config.dataset_path,
        feature_config_path=train_config.feature_config_path,
        batch_size=train_config.batch_size,
        val_split=train_config.val_split,
        test_split=train_config.test_split,
        num_workers=train_config.num_workers,
        seed=train_config.seed
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = StateEncModel.from_config(model_config.__dict__)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = StateEncTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_config=model_config
    )
    
    # Train
    summary = trainer.train()
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train STATE-ENC v1")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    args = parser.parse_args()
    
    summary = train_state_enc(args.config)
    print(json.dumps(summary, indent=2))

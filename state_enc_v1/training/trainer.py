"""
Training loop for STATE-ENC v1.1

Features:
- Data augmentation (noise, dropout, bar masking)
- Cosine annealing LR
- Best model by total loss
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..src.config import StateEncTrainConfig, StateEncModelConfig
from ..src.model.state_enc_model import StateEncModel
from ..src.dataset_encoder import create_dataloaders
from .losses import MultiHeadLoss
from .eval_metrics import MetricsAccumulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = score < self.best_score if self.mode == "min" else score > self.best_score
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class StateEncTrainer:
    """Trainer for STATE-ENC v1.1"""
    
    def __init__(self,
                 model: StateEncModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: StateEncTrainConfig,
                 model_config: StateEncModelConfig):
        
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
        logger.info(f"Device: {self.device}")
        
        # Loss
        self.loss_fn = MultiHeadLoss(loss_weights=config.loss_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler: Cosine annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.patience, mode="min")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        
        # History
        self.history = {"train_loss": [], "val_loss": [], "val_dir_acc": [], "lr": []}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with augmentation"""
        self.model.train()
        accumulator = MetricsAccumulator()
        
        for batch in self.train_loader:
            X = batch["X"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "X"}
            
            self.optimizer.zero_grad()
            
            # Forward with augmentation
            outputs = self.model(X, augment=True)
            
            # Compute loss
            losses = self.loss_fn(outputs, targets)
            loss = losses["loss_total"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            accumulator.update(outputs, targets, losses)
        
        return accumulator.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model (no augmentation)"""
        self.model.eval()
        accumulator = MetricsAccumulator()
        
        for batch in self.val_loader:
            X = batch["X"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "X"}
            
            outputs = self.model(X, augment=False)
            losses = self.loss_fn(outputs, targets)
            
            accumulator.update(outputs, targets, losses)
        
        return accumulator.compute()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.model_config.__dict__ if hasattr(self.model_config, '__dict__') else self.model_config
        }
        
        if is_best:
            self.best_state = self.model.state_dict().copy()
            best_path = self.run_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("=" * 60)
        logger.info("STATE-ENC v1.1 Training")
        logger.info(f"Epochs: {self.config.max_epochs}, LR: {self.config.learning_rate}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update history
            self.history["train_loss"].append(train_metrics.get("avg_loss_total", 0))
            self.history["val_loss"].append(val_metrics.get("avg_loss_total", 0))
            self.history["val_dir_acc"].append(val_metrics.get("dir_accuracy", 0))
            self.history["lr"].append(current_lr)
            
            # Check best
            val_loss = val_metrics.get("avg_loss_total", float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} | "
                f"Train: {train_metrics.get('avg_loss_total', 0):.4f} | "
                f"Val: {val_loss:.4f} | "
                f"DirAcc: {val_metrics.get('dir_accuracy', 0):.4f} "
                f"{'*BEST*' if is_best else ''}"
            )
            
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Save history
        with open(self.run_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch,
            "total_time_seconds": total_time,
            "run_dir": str(self.run_dir),
            "final_metrics": val_metrics
        }


def train_state_enc(config_path: str) -> Dict[str, Any]:
    """Main training function"""
    train_config = StateEncTrainConfig.from_json(config_path)
    model_config = StateEncModelConfig.from_json(train_config.model_config_path)
    
    torch.manual_seed(train_config.seed)
    
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
    
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    model = StateEncModel.from_config(model_config.__dict__)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")
    logger.info(f"Heads: {model.get_head_names()}")
    logger.info(f"z_t dim: {model.get_embedding_dim()}")
    
    trainer = StateEncTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_config=model_config
    )
    
    return trainer.train()

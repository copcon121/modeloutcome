"""Trainer for ASM v2"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from .losses import AsmLoss
from .eval_metrics import AsmMetrics


class AsmTrainer:
    """Trainer for ASM v2 model"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1,
                 label_smoothing: float = 0.05,
                 device: str = "cuda",
                 output_dir: str = "asm_v2/artifacts/final"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss
        self.criterion = AsmLoss(label_smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler will be set in train()
        self.scheduler = None
        self.warmup_ratio = warmup_ratio
        
        # Metrics
        self.train_metrics = AsmMetrics(num_classes)
        self.val_metrics = AsmMetrics(num_classes)
        
        # Best tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
    
    def train(self, epochs: int, save_best: bool = True) -> Dict:
        """Train the model"""
        
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Cosine scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        history = {"train": [], "val": []}
        
        for epoch in range(epochs):
            # Train epoch
            train_results = self._train_epoch(epoch, epochs)
            history["train"].append(train_results)
            
            # Validate
            val_results = self._validate()
            history["val"].append(val_results)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_results['avg_loss']:.4f} | "
                  f"Train Acc: {train_results['accuracy']:.4f} | "
                  f"Train F1: {train_results['macro_f1']:.4f} | "
                  f"Val Acc: {val_results['accuracy']:.4f} | "
                  f"Val F1: {val_results['macro_f1']:.4f}")
            
            # Save best
            if save_best and val_results["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_results["macro_f1"]
                self.best_epoch = epoch + 1
                self._save_checkpoint("asm_v2_gc_m1_v1.pt", val_results)
        
        print(f"\nBest Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        
        return history
    
    def _train_epoch(self, epoch: int, total_epochs: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in pbar:
            x = batch["x"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Loss
            loss_dict = self.criterion(outputs, labels)
            loss = loss_dict["loss"]
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            preds = outputs["logits"].argmax(dim=-1)
            self.train_metrics.update(preds, labels, loss.item())
            
            pbar.set_postfix({"loss": loss.item()})
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def _validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            x = batch["x"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(x)
            loss_dict = self.criterion(outputs, labels)
            
            preds = outputs["logits"].argmax(dim=-1)
            self.val_metrics.update(preds, labels, loss_dict["loss"].item())
        
        return self.val_metrics.compute()
    
    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config(),
            "metrics": metrics,
            "epoch": self.best_epoch
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        
        # Also save config separately
        config_path = self.output_dir / "asm_model_config_v1.json"
        with open(config_path, "w") as f:
            json.dump(self.model.get_config(), f, indent=2)
        
        print(f"Saved checkpoint to {path}")

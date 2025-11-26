"""
Train Outcome Model
Trains a Transformer or MLP model on labeled dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from layer2_feature_engine.utils.config_loader import load_config
from layer2_feature_engine.utils.logging_utils import setup_logger

logger = setup_logger('training', log_file='logs/training.log')


class OutcomeDataset(Dataset):
    """PyTorch Dataset for outcome model"""

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to .pt dataset file
        """
        logger.info(f"Loading dataset from {dataset_path}")
        data = torch.load(dataset_path)

        self.features = data['features']  # [N, context_len, feature_dim]
        self.labels = data['labels']      # [N]
        self.metadata = data.get('metadata', [])
        self.feature_names = data.get('feature_names', [])

        logger.info(f"Dataset loaded: {self.features.shape}, {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleTransformer(nn.Module):
    """Simple Transformer Encoder for sequence classification"""

    def __init__(self, feature_dim: int, config: dict):
        super().__init__()

        d_model = config.get('d_model', 128)
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_encoder_layers', 3)
        dim_feedforward = config.get('dim_feedforward', 256)
        dropout = config.get('dropout', 0.1)
        num_classes = config.get('num_classes', 3)

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 200, d_model))  # Max 200 bars

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_dim]

        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Global mean pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # Classification
        logits = self.fc(x)  # [batch, num_classes]

        return logits


class SimpleMLP(nn.Module):
    """Simple MLP baseline model"""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()

        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        dropout = config.get('dropout', 0.2)
        num_classes = config.get('num_classes', 3)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_dim]

        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, feature_dim = x.shape

        # Flatten
        x = x.view(batch_size, -1)  # [batch, seq_len * feature_dim]

        # Forward
        logits = self.network(x)

        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc='Training'):
        features = features.to(device)
        labels = labels.to(device)

        # Forward
        logits = model(features)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description='Train outcome model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--dataset', type=str, default='data/datasets/outcome_train.pt')
    parser.add_argument('--output', type=str, default='models/outcome_transformer_v1.pt')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_config = config['model']
    training_config = config['training']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = OutcomeDataset(args.dataset)

    # Split train/val
    train_size = int(training_config['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Model
    _, seq_len, feature_dim = dataset.features.shape
    logger.info(f"Input shape: seq_len={seq_len}, feature_dim={feature_dim}")

    architecture = model_config['architecture']

    if architecture == 'transformer':
        model = SimpleTransformer(feature_dim, model_config['transformer'])
    elif architecture == 'mlp':
        input_dim = seq_len * feature_dim
        model = SimpleMLP(input_dim, model_config['mlp'])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01)
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = training_config.get('patience', 5)

    for epoch in range(training_config['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{training_config['epochs']}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, args.output)
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()

"""
Evaluate Sequence Quality Model with Backtest

Applies trained seq ML filter to validation events and computes trading metrics.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Local imports
from phase6_seq_quality.model_seq import QualitySeqGRU


def load_normalizer(normalizer_path):
    norm = torch.load(normalizer_path, map_location="cpu")
    return norm["mean"], norm["std"]


def prepare_dataset(dataset_path, normalizer_mean, normalizer_std, device):
    data = torch.load(dataset_path)
    X = data["X"]  # [N, 60, 66]
    side = data["side"].float().view(-1, 1)  # [N, 1]
    y = data["y_quality"]
    meta = data["meta"]

    # Normalize per feature (broadcast across time)
    X_norm = (X - normalizer_mean) / (normalizer_std + 1e-8)
    return X_norm.to(device), side.to(device), y, meta


def calculate_trading_metrics(events, preds, probs, threshold, name):
    trades = []
    for i, event in enumerate(events):
        if preds[i] == 1:
            trades.append(
                {
                    "hit": event["hit"],
                    "outcome_rr": event["outcome_rr"],
                    "signal_side": event["signal_side"],
                }
            )
    if len(trades) == 0:
        return {
            "name": name,
            "num_trades": 0,
            "winners": 0,
            "losers": 0,
            "winrate": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
            "total_r": 0.0,
            "max_dd_r": 0.0,
            "threshold": threshold,
        }

    winners = sum(1 for t in trades if t["hit"] == "tp")
    losers = sum(1 for t in trades if t["hit"] == "sl")
    none_hits = sum(1 for t in trades if t["hit"] == "none")
    winrate = winners / len(trades) if trades else 0.0
    total_r = sum(t["outcome_rr"] for t in trades)
    avg_r = total_r / len(trades) if trades else 0.0
    expectancy = avg_r

    cumulative_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0
    for t in trades:
        cumulative_r += t["outcome_rr"]
        peak_r = max(peak_r, cumulative_r)
        dd = peak_r - cumulative_r
        max_dd_r = max(max_dd_r, dd)

    return {
        "name": name,
        "num_trades": len(trades),
        "winners": winners,
        "losers": losers,
        "none_hits": none_hits,
        "winrate": winrate,
        "avg_r": avg_r,
        "expectancy": expectancy,
        "total_r": total_r,
        "max_dd_r": max_dd_r,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate seq quality model with backtest")
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/phase4_quality/dataset_p2_quality_v1_val.pt",
        help="Validation dataset path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output/phase6_seq_quality/model_seq_quality_v1_best.pt",
        help="Seq model path",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="output/phase6_seq_quality/normalizer_stats_seq.pt",
        help="Normalizer stats path",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5],
        help="Thresholds to evaluate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SEQ QUALITY BACKTEST")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Norm:    {args.normalizer}")
    print(f"Thresholds: {args.thresholds}")

    # Load normalizer and dataset
    mean, std = load_normalizer(args.normalizer)
    mean = mean.to(device)
    std = std.to(device)

    X, side, y_true, meta = prepare_dataset(args.dataset, mean, std, device)

    # Load model with inferred dimensions from checkpoint
    state = torch.load(args.model, map_location=device)
    inferred_hidden = 128
    inferred_input_dim = 66  # default
    if "gru.weight_ih_l0" in state:
        inferred_hidden = state["gru.weight_ih_l0"].shape[0] // 3
        inferred_input_dim = state["gru.weight_ih_l0"].shape[1] - 1  # subtract side dim
    model = QualitySeqGRU(input_dim=inferred_input_dim, hidden_dim=inferred_hidden, num_layers=1, dropout=0.1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(X, side).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    preds_default = (probs >= 0.5).astype(int)

    # Metrics for default 0.5
    y_true_np = y_true.numpy()
    acc = accuracy_score(y_true_np, preds_default)
    precision = precision_score(y_true_np, preds_default, zero_division=0)
    recall = recall_score(y_true_np, preds_default, zero_division=0)
    f1 = f1_score(y_true_np, preds_default, zero_division=0)
    try:
        auc = roc_auc_score(y_true_np, probs)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_true_np, preds_default)

    print("\nClassification metrics (threshold=0.5):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print("\n  Confusion Matrix:")
    print("                Predicted")
    print("              DROP  KEEP")
    print(f"  Actual DROP  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        KEEP  {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Trading metrics per threshold
    results = []
    for th in args.thresholds:
        preds = (probs >= th).astype(int)
        stats = calculate_trading_metrics(meta, preds, probs, th, name=f"Seq@{th}")
        results.append(stats)

    print("\nTrading metrics:")
    for r in results:
        print(f"\n{r['name']} (threshold={r['threshold']}):")
        print(f"  Trades:     {r['num_trades']:,}")
        print(f"  Winners:    {r['winners']:,}")
        print(f"  Losers:     {r['losers']:,}")
        print(f"  Winrate:    {r['winrate']*100:.1f}%")
        print(f"  Avg R:      {r['avg_r']:+.4f}")
        print(f"  Expectancy: {r['expectancy']:+.4f}R")
        print(f"  Total R:    {r['total_r']:+.2f}R")
        print(f"  Max DD:     {r['max_dd_r']:.2f}R")

    # Export JSON report
    out_path = Path(args.dataset).parent / "seq_backtest_report.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "classification": {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm.tolist(),
                },
                "threshold_results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved seq backtest report: {out_path}")


def evaluate_model_on_dataset(model_path, normalizer_path, dataset_path, threshold=0.5):
    """
    Helper: load model/normalizer, run on dataset, return metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_normalizer(normalizer_path)
    mean = mean.to(device)
    std = std.to(device)

    X, side, y_true, meta = prepare_dataset(dataset_path, mean, std, device)

    # Load model (infer hidden_dim from state dict input_proj weight shape if needed)
    state = torch.load(model_path, map_location=device)
    # Try to infer hidden_dim from state dict (fallback to 128)
    hidden_dim = state["gru.weight_ih_l0"].shape[0] // 3 if "gru.weight_ih_l0" in state else 128
    model = QualitySeqGRU(input_dim=66, hidden_dim=hidden_dim, num_layers=1, dropout=0.1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X, side).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= threshold).astype(int)
    y_true_np = y_true.numpy()

    acc = accuracy_score(y_true_np, preds)
    precision = precision_score(y_true_np, preds, zero_division=0)
    recall = recall_score(y_true_np, preds, zero_division=0)
    f1 = f1_score(y_true_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true_np, probs)
    except Exception:
        auc = 0.0

    trade_metrics = calculate_trading_metrics(meta, preds, probs, threshold, name="seq")
    trade_metrics.update({
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    })
    return trade_metrics


if __name__ == "__main__":
    main()

"""
Python shadow replay / live backtest for QualitySeq v2.

Uses validation quality dataset, applies seq model with official threshold, logs per-event decisions,
and summarizes expectancy/drawdown.
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
import torch
import numpy as np

from phase6_seq_quality.eval_seq_backtest import load_normalizer, prepare_dataset, calculate_trading_metrics
from phase6_seq_quality.model_seq import QualitySeqGRU


def max_drawdown(series):
    peak = -1e9
    max_dd = 0.0
    cum = 0.0
    for x in series:
        cum += x
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)
    return max_dd


def main():
    parser = argparse.ArgumentParser(description="Shadow replay using seq quality model")
    parser.add_argument("--dataset", type=str, default="output/phase4_quality/dataset_p2_quality_v1_val.pt",
                        help="Validation dataset path")
    parser.add_argument("--model", type=str, default="output/phase6_seq_quality/model_seq_quality_v1_best.pt",
                        help="Seq model path (promoted)")
    parser.add_argument("--normalizer", type=str, default="output/phase6_seq_quality/normalizer_stats_seq.pt",
                        help="Normalizer stats path")
    parser.add_argument("--threshold", type=float, default=0.7, help="Quality threshold")
    parser.add_argument("--out-dir", type=str, default="output/phase8_live_replay", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "shadow_trading_log_replay.jsonl"
    summary_path = out_dir / "shadow_replay_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_normalizer(args.normalizer)
    mean = mean.to(device)
    std = std.to(device)

    X, side, y_true, meta = prepare_dataset(args.dataset, mean, std, device)

    # Sort by timestamp for chronological replay
    timestamps = [m.get("timestamp", "") for m in meta]
    order = sorted(range(len(meta)), key=lambda i: timestamps[i])
    X = X[order]
    side = side[order]
    meta = [meta[i] for i in order]

    # Infer hidden_dim if possible
    state = torch.load(args.model, map_location=device)
    hidden_dim = state["gru.weight_ih_l0"].shape[0] // 3 if "gru.weight_ih_l0" in state else 128
    model = QualitySeqGRU(input_dim=66, hidden_dim=hidden_dim, num_layers=1, dropout=0.1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X, side).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    keeps = probs >= args.threshold
    # Write JSONL log
    with open(log_path, "w", encoding="utf-8") as f:
        for i, m in enumerate(meta):
            record = {
                "timestamp_server": datetime.utcnow().isoformat(),
                "model_type": "seq_v2",
                "mode": "seq_conservative",
                "threshold": args.threshold,
                "shadow_only": True,
                "p_keep": float(probs[i]),
                "keep": bool(keeps[i]),
                "side": int(side[i].item()),
                "meta": {
                    "timestamp": m.get("timestamp"),
                    "symbol_root": m.get("symbol_root"),
                    "session": m.get("session"),
                    "event_id": m.get("event_id"),
                },
                "outcome_R": m.get("outcome_rr"),
            }
            f.write(json.dumps(record) + "\n")

    # Summary stats
    outcomes = [m.get("outcome_rr", 0.0) for m in meta]
    outcomes_kept = [o for o, k in zip(outcomes, keeps) if k]
    trades = sum(keeps)
    expectancy = np.mean(outcomes_kept) if trades > 0 else 0.0
    dd = max_drawdown(outcomes_kept) if trades > 0 else 0.0
    winrate = sum(1 for o in outcomes_kept if o > 0) / trades if trades > 0 else 0.0

    summary = {
        "model": args.model,
        "normalizer": args.normalizer,
        "threshold": args.threshold,
        "num_events": len(meta),
        "num_trades": int(trades),
        "expectancy_R": float(expectancy),
        "max_drawdown_R": float(dd),
        "winrate": float(winrate),
        "log_path": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nShadow replay complete:")
    print(f"  Trades: {trades}")
    print(f"  Expectancy: {expectancy:+.4f}R")
    print(f"  Max DD: {dd:.2f}R")
    print(f"  Winrate: {winrate*100:.1f}%")
    print(f"  Log: {log_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

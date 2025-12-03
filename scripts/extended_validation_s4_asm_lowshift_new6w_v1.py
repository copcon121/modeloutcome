#!/usr/bin/env python3
"""
Extended Validation: S4_HighVol_FVG_London + ASM LowShift on NEW DATA 6W
========================================================================
Validate S4_LDN_ASM_LowShift_0.2_v1.0 strategy on out-of-sample new data.

Data Source: data/raw/new_data/ (Apr 28 - Jun 02, 2025)
Model: ASM-GRU64-v1.0-C3 (bar-only, 100 features)
Filter: p_shift <= 0.2 (inverse filter - stable auction)

Usage:
    python scripts/extended_validation_s4_asm_lowshift_new6w_v1.py
"""

import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add project root to path
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar
from asm_inference_v1 import ASMModelV1Loader

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
RAW_DATA_DIR = ROOT / "data/raw/new_data"
FEATURES_OUTPUT_DIR = ROOT / "output/new_data_features_s4_asm"

# Output
OUTPUT_PATH = ROOT / "backtests/s4_asm_lowshift_extval_new6w_v1.json"

# S4 Rule parameters
S4_CONFIG = {
    "session": "London",
    "session_start_hour": 8,
    "session_end_hour": 14,
    "rr_target": 2.0,
    "max_bars_in_trade": 100,
}

# ASM parameters
ASM_SEQ_LEN = 60
ASM_FEATURE_DIM = 100

# Filter thresholds to test
P_SHIFT_THRESHOLDS = [0.15, 0.2, 0.25, 0.3]

# Feature columns to exclude (metadata)
EXCLUDE_COLS = ["timestamp", "bar_index", "global_bar_index", "_source_file", "open", "high", "low", "close"]


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class S4Trade:
    """S4 trade record."""
    bar_index: int
    timestamp: str
    session: str
    side: int  # +1 long, -1 short
    entry_price: float
    sl_price: float
    tp_price: float
    outcome: str  # 'win', 'loss', 'timeout'
    pnl_r: float
    exit_bar: int
    # ASM scores
    asm_p_up: float = 0.0
    asm_p_down: float = 0.0
    asm_p_neutral: float = 0.0
    asm_p_shift: float = 0.0


# ==============================================================================
# DATA LOADING (reuse from process_new_data_features.py)
# ==============================================================================


def load_raw_bar(data: dict) -> RawBar:
    """Convert JSON dict to RawBar."""
    ts_str = data['timestamp']
    ts = datetime.fromisoformat(ts_str)
    
    bar_data = data['bar']
    tick_data = data.get('tick_features', {})
    
    return RawBar(
        symbol=data.get('symbol', 'GC'),
        timeframe=data.get('timeframe', 'M1'),
        timestamp=ts,
        bar_index=data['bar_index'],
        o=bar_data['o'],
        h=bar_data['h'],
        l=bar_data['l'],
        c=bar_data['c'],
        volume=bar_data['volume'],
        delta=bar_data.get('delta', 0),
        buy_volume=bar_data.get('buy_volume', 0),
        sell_volume=bar_data.get('sell_volume', 0),
        best_bid=bar_data.get('best_bid', bar_data['c']),
        best_ask=bar_data.get('best_ask', bar_data['c']),
        tick_speed=tick_data.get('tick_speed', 0),
        aggr_buy_speed=tick_data.get('aggr_buy_speed', 0),
        aggr_sell_speed=tick_data.get('aggr_sell_speed', 0),
        price_speed=tick_data.get('price_speed', bar_data['h'] - bar_data['l']),
        vwap_daily=bar_data.get('vwap_daily', 0.0)
    )


def process_raw_files() -> pd.DataFrame:
    """Process raw JSONL files and generate features."""
    print("=" * 70)
    print("PROCESSING NEW DATA 6W FOR S4 + ASM VALIDATION")
    print("=" * 70)
    
    # Find all raw files
    files = sorted(RAW_DATA_DIR.glob("smc_export_gc_m1_v3_*.jsonl"))
    print(f"\nFound {len(files)} files to process:")
    for f in files:
        print(f"  - {f.name}")
    
    all_bars = []
    
    for input_path in files:
        print(f"\n  Processing {input_path.name}...")
        
        # Initialize Manager (fresh for each file)
        manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    raw_bar = load_raw_bar(data)
                    feature_bar = manager.update(raw_bar)
                    
                    # Convert to dict
                    fb_dict = feature_bar.to_dict()
                    fb_dict['timestamp'] = raw_bar.timestamp.isoformat()
                    fb_dict['bar_index'] = raw_bar.bar_index
                    # Add raw OHLC
                    fb_dict['open'] = raw_bar.o
                    fb_dict['high'] = raw_bar.h
                    fb_dict['low'] = raw_bar.l
                    fb_dict['close'] = raw_bar.c
                    all_bars.append(fb_dict)
                    
                except Exception as e:
                    if line_num < 5:
                        print(f"    Error line {line_num}: {e}")
                    continue
        
        print(f"    Processed, total bars so far: {len(all_bars)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['global_bar_index'] = range(len(df))
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal bars: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_feature_columns(df: pd.DataFrame, target_features: int = 100) -> List[str]:
    """Get exactly N numeric feature columns (excluding metadata).
    
    If not enough features, pad with zeros later.
    """
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS or col.startswith("_"):
            continue
        if col.startswith("weekly_") or col.startswith("daily_va_"):
            continue  # Skip Weekly VA features for ASM v1.0
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)
    return feature_cols[:target_features]  # Return up to target_features


def detect_session(hour: int) -> str:
    """Detect trading session from hour (UTC)."""
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 14:
        return "London"
    else:
        return "NY"


def is_high_vol_regime(row: pd.Series, df: pd.DataFrame, idx: int, lookback: int = 100) -> bool:
    """Check if current bar is in high volatility regime."""
    if idx < lookback:
        return False
    
    recent = df.iloc[max(0, idx - lookback):idx]
    range_q66 = recent["high_low_range"].quantile(0.66)
    avg_vol = recent["volume"].mean()
    
    high_range = row["high_low_range"] > range_q66
    high_volume = row["volume"] > avg_vol * 2.0
    
    return high_range or high_volume


def check_s4_setup(row: pd.Series, df: pd.DataFrame, idx: int, session_filter: str = None) -> Tuple[bool, int]:
    """Check if bar qualifies as S4 setup."""
    hour = row["timestamp"].hour
    session = detect_session(hour)
    
    if session_filter and session != session_filter:
        return False, 0
    
    if not is_high_vol_regime(row, df, idx):
        return False, 0
    
    in_bull_fvg = row.get("in_bull_fvg", 0) == 1
    in_bear_fvg = row.get("in_bear_fvg", 0) == 1
    ext_trend = row.get("ext_trend_dir", 0)
    
    if in_bull_fvg and ext_trend > 0:
        return True, 1
    if in_bear_fvg and ext_trend < 0:
        return True, -1
    
    return False, 0


def calculate_sl_tp(row: pd.Series, side: int, rr: float = 2.0) -> Tuple[float, float]:
    """Calculate SL and TP for a trade."""
    atr_approx = row["high_low_range"] * 2
    
    if side == 1:
        sl = row["low"] - atr_approx * 0.5
        risk = row["close"] - sl
        tp = row["close"] + risk * rr
    else:
        sl = row["high"] + atr_approx * 0.5
        risk = sl - row["close"]
        tp = row["close"] - risk * rr
    
    return sl, tp


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: int, sl: float, tp: float, max_bars: int = 100) -> Tuple[str, float, int]:
    """Simulate trade outcome."""
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        
        if side == 1:
            if low <= sl:
                return "loss", -1.0, i
            if high >= tp:
                return "win", 2.0, i
        else:
            if high >= sl:
                return "loss", -1.0, i
            if low <= tp:
                return "win", 2.0, i
    
    return "timeout", 0.0, entry_idx + max_bars


def get_context_window(df: pd.DataFrame, idx: int, feature_cols: List[str], seq_len: int = 60, target_dim: int = 100) -> np.ndarray:
    """Get context window for ASM inference.
    
    Pads with zeros if not enough features to match target_dim.
    """
    if idx < seq_len:
        return None
    
    context = df.iloc[idx - seq_len:idx][feature_cols].values.astype(np.float32)
    
    # Pad if needed
    if context.shape[1] < target_dim:
        padding = np.zeros((seq_len, target_dim - context.shape[1]), dtype=np.float32)
        context = np.concatenate([context, padding], axis=1)
    
    return context


def calculate_stats(trades: List[S4Trade]) -> Dict:
    """Calculate trading statistics."""
    if not trades:
        return {"trades": 0, "winrate": 0.0, "expectancy": 0.0, "maxdd": 0.0, "total_R": 0.0}
    
    pnls = [t.pnl_r for t in trades]
    wins = sum(1 for t in trades if t.outcome == "win")
    
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    maxdd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    
    return {
        "trades": len(trades),
        "winrate": round(100 * wins / len(trades), 2),
        "expectancy": round(float(np.mean(pnls)), 4),
        "maxdd": round(maxdd, 2),
        "total_R": round(sum(pnls), 2),
    }


def calculate_stats_by_session(trades: List[S4Trade]) -> Dict:
    """Calculate stats broken down by session."""
    sessions = {"Asia": [], "London": [], "NY": []}
    for t in trades:
        if t.session in sessions:
            sessions[t.session].append(t)
    
    return {s: calculate_stats(ts) for s, ts in sessions.items()}


# ==============================================================================
# MAIN VALIDATION
# ==============================================================================


def run_extended_validation():
    """Run extended validation on new 6W data."""
    print("=" * 70)
    print("EXTENDED VALIDATION: S4_LDN + ASM LowShift on NEW DATA 6W")
    print("=" * 70)
    
    # Step 1: Process raw data to features
    df = process_raw_files()
    feature_cols = get_feature_columns(df, target_features=ASM_FEATURE_DIM)
    print(f"\nUsing {len(feature_cols)} features for ASM")
    
    # Step 2: Load ASM model
    print("\nLoading ASM model...")
    asm_loader = ASMModelV1Loader()
    
    # Step 3: Generate S4 London trades
    print("\nGenerating S4 London trades...")
    trades: List[S4Trade] = []
    skipped_nan = 0
    
    for idx in range(ASM_SEQ_LEN, len(df) - S4_CONFIG["max_bars_in_trade"]):
        row = df.iloc[idx]
        
        # Check S4 setup (London session only)
        is_setup, side = check_s4_setup(row, df, idx, session_filter="London")
        if not is_setup:
            continue
        
        # Calculate SL/TP
        sl, tp = calculate_sl_tp(row, side, S4_CONFIG["rr_target"])
        
        # Simulate trade
        outcome, pnl_r, exit_bar = simulate_trade(df, idx, side, sl, tp, S4_CONFIG["max_bars_in_trade"])
        
        # Get ASM context and predict
        context = get_context_window(df, idx, feature_cols, ASM_SEQ_LEN, ASM_FEATURE_DIM)
        if context is None or np.isnan(context).any():
            skipped_nan += 1
            continue
        
        asm_probs = asm_loader.predict_proba(context)
        
        trade = S4Trade(
            bar_index=idx,
            timestamp=str(row["timestamp"]),
            session=detect_session(row["timestamp"].hour),
            side=side,
            entry_price=row["close"],
            sl_price=sl,
            tp_price=tp,
            outcome=outcome,
            pnl_r=pnl_r,
            exit_bar=exit_bar,
            asm_p_up=asm_probs["p_up"],
            asm_p_down=asm_probs["p_down"],
            asm_p_neutral=asm_probs["p_neutral"],
            asm_p_shift=asm_probs["p_shift"],
        )
        trades.append(trade)
    
    print(f"  Generated {len(trades)} S4 London trades")
    if skipped_nan > 0:
        print(f"  Skipped {skipped_nan} trades due to NaN context")
    
    # Step 4: Calculate baseline stats
    baseline_stats = calculate_stats(trades)
    baseline_by_session = calculate_stats_by_session(trades)
    
    print(f"\n{'='*70}")
    print("BASELINE: S4_London NEW6W (no ASM filter)")
    print(f"{'='*70}")
    print(f"  Trades:     {baseline_stats['trades']}")
    print(f"  Winrate:    {baseline_stats['winrate']}%")
    print(f"  Expectancy: {baseline_stats['expectancy']}R")
    print(f"  MaxDD:      {baseline_stats['maxdd']}R")
    print(f"  Total R:    {baseline_stats['total_R']}R")
    
    # Step 5: Test ASM LowShift filters
    print(f"\n{'='*70}")
    print("ASM LowShift FILTER (p_shift <= threshold)")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Trades':<10} {'WR%':<10} {'Exp(R)':<12} {'MaxDD':<10} {'Retain%':<10}")
    print("-" * 64)
    
    filter_results = {}
    
    for threshold in P_SHIFT_THRESHOLDS:
        filtered = [t for t in trades if t.asm_p_shift <= threshold]
        stats = calculate_stats(filtered)
        retain_pct = round(100 * len(filtered) / len(trades), 1) if trades else 0
        
        print(f"{threshold:<12} {stats['trades']:<10} {stats['winrate']:<10} {stats['expectancy']:<12} {stats['maxdd']:<10} {retain_pct:<10}")
        
        filter_results[f"lowshift_{threshold}"] = {
            "threshold": threshold,
            **stats,
            "retain_pct": retain_pct,
        }
    
    # Primary filter: p_shift <= 0.2
    primary_filtered = [t for t in trades if t.asm_p_shift <= 0.2]
    primary_stats = calculate_stats(primary_filtered)
    primary_by_session = calculate_stats_by_session(primary_filtered)
    
    # Step 6: Build output JSON
    output = {
        "meta": {
            "strategy_id": "S4_LDN_ASM_LowShift_0.2_v1.0",
            "source_data": "data/raw/new_data",
            "data_period": "Apr 28 - Jun 02, 2025 (6W)",
            "asm_model": "ASM-GRU64-v1.0-C3.pt",
            "asm_seq_len": ASM_SEQ_LEN,
            "asm_feature_dim": ASM_FEATURE_DIM,
            "rr_target": S4_CONFIG["rr_target"],
            "session": "London",
            "total_bars": len(df),
            "date_generated": datetime.now().isoformat(),
            "comment": "Extended validation on new 6W data (out-of-sample)"
        },
        "baseline_s4_london": {
            **baseline_stats,
            "by_session": baseline_by_session
        },
        "asm_lowshift_0_2": {
            **primary_stats,
            "retain_pct": round(100 * len(primary_filtered) / len(trades), 1) if trades else 0,
            "by_session": primary_by_session
        },
        "filter_sweep": filter_results,
    }
    
    # Save JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    # Step 7: Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: S4_LDN + ASM LowShift 0.2 on NEW DATA 6W")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<15} {'Baseline S4_LDN':<20} {'+ ASM LowShift 0.2':<20} {'Change':<15}")
    print("-" * 70)
    
    def fmt_change(old, new, higher_better=True):
        if old == 0:
            return "N/A"
        pct = 100 * (new - old) / abs(old)
        sign = "+" if pct > 0 else ""
        good = (pct > 0) == higher_better
        return f"{sign}{pct:.1f}% {'✓' if good else '✗'}"
    
    print(f"{'Trades':<15} {baseline_stats['trades']:<20} {primary_stats['trades']:<20} {fmt_change(baseline_stats['trades'], primary_stats['trades'], False):<15}")
    print(f"{'Winrate':<15} {baseline_stats['winrate']:.1f}%{'':<16} {primary_stats['winrate']:.1f}%{'':<16} {fmt_change(baseline_stats['winrate'], primary_stats['winrate'], True):<15}")
    print(f"{'Expectancy':<15} {baseline_stats['expectancy']:.3f}R{'':<14} {primary_stats['expectancy']:.3f}R{'':<14} {fmt_change(baseline_stats['expectancy'], primary_stats['expectancy'], True):<15}")
    print(f"{'MaxDD':<15} {baseline_stats['maxdd']:.1f}R{'':<16} {primary_stats['maxdd']:.1f}R{'':<16} {fmt_change(baseline_stats['maxdd'], primary_stats['maxdd'], False):<15}")
    print(f"{'Total R':<15} {baseline_stats['total_R']:.1f}R{'':<16} {primary_stats['total_R']:.1f}R{'':<16} {fmt_change(baseline_stats['total_R'], primary_stats['total_R'], True):<15}")
    
    retain_pct = round(100 * len(primary_filtered) / len(trades), 1) if trades else 0
    print(f"\nTrade Retention: {retain_pct}% ({len(primary_filtered)}/{len(trades)})")
    
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    
    return output


if __name__ == "__main__":
    run_extended_validation()

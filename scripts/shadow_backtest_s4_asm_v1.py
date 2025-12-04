#!/usr/bin/env python3
"""
Shadow Backtest: S4_HighVol_FVG_London + ASM v1.1 Filter
========================================================
Evaluate ASM-GRU64-v1.0-C3 as auction filter for S4 baseline strategy.

v1.1 FIX: Uses unified ASM_FEATURE_COLS from src/asm_feature_spec.py
          Previous v1.0 had feature mismatch bug (missing "close" at position 0)

Usage:
    python scripts/shadow_backtest_s4_asm_v1.py
"""

import json
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.asm_feature_spec import ASM_FEATURE_COLS, ASM_SEQ_LEN, ASM_FEATURE_DIM

sys.path.insert(0, str(ROOT / "scripts"))
from asm_inference_v1 import ASMModelV1Loader

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
DATA_PATHS = [
    "data/processed_v2/*.csv",
    "output/new_data_features/smc_export_*.csv",
]

# Output
OUTPUT_PATH = Path("backtests/s4_highvol_fvg_london_asm_v1_shadow.json")

# S4 Rule parameters
S4_CONFIG = {
    "session": "London",  # 08:00 - 14:00 UTC
    "session_start_hour": 8,
    "session_end_hour": 14,
    "rr_target": 2.0,
    "max_bars_in_trade": 100,
}

# Note: ASM_SEQ_LEN, ASM_FEATURE_DIM, ASM_FEATURE_COLS imported from src.asm_feature_spec

# Threshold sweep - DIRECT FILTER (high p_shift)
T_SHIFT_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
T_DIR_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2]

# Threshold sweep - INVERSE FILTER (low p_shift)
T_SHIFT_MAX_VALUES = [0.2, 0.3, 0.4, 0.5]

# Threshold sweep - NEUTRAL FILTER (high p_neutral)
T_NEUTRAL_MIN_VALUES = [0.5, 0.6, 0.7, 0.8]


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
    # ASM scores (filled later)
    asm_p_up: float = 0.0
    asm_p_down: float = 0.0
    asm_p_neutral: float = 0.0
    asm_p_shift: float = 0.0
    asm_dir_score: float = 0.0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def load_feature_data(data_paths: List[str]) -> pd.DataFrame:
    """Load and combine all feature CSVs."""
    all_files = []
    for pattern in data_paths:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {data_paths}")
    
    print(f"Loading {len(all_files)} feature files...")
    
    dfs = []
    for f in sorted(all_files):
        df = pd.read_csv(f)
        df["_source_file"] = Path(f).name
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    print(f"  Total bars: {len(combined)}")
    return combined


def get_feature_columns(df: pd.DataFrame, max_features: int = 100) -> List[str]:
    """Get fixed feature columns matching training data order.
    
    Uses ASM_FEATURE_COLS from src.asm_feature_spec for consistency.
    """
    return ASM_FEATURE_COLS[:max_features]


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
    
    # Get rolling stats
    recent = df.iloc[max(0, idx - lookback):idx]
    range_q66 = recent["high_low_range"].quantile(0.66)
    avg_vol = recent["volume"].mean()
    
    # High vol conditions
    high_range = row["high_low_range"] > range_q66
    high_volume = row["volume"] > avg_vol * 2.0
    
    return high_range or high_volume


def check_s4_setup(row: pd.Series, df: pd.DataFrame, idx: int) -> Tuple[bool, int]:
    """
    Check if bar qualifies as S4 setup.
    
    Returns:
        (is_setup, side) where side is +1 (long) or -1 (short)
    """
    # Session filter
    hour = row["timestamp"].hour
    session = detect_session(hour)
    if session != S4_CONFIG["session"]:
        return False, 0
    
    # High vol regime
    if not is_high_vol_regime(row, df, idx):
        return False, 0
    
    # FVG + Trend conditions
    in_bull_fvg = row.get("in_bull_fvg", 0) == 1
    in_bear_fvg = row.get("in_bear_fvg", 0) == 1
    
    ext_trend = row.get("ext_trend_dir", 0)
    ext_bos_up = row.get("ext_bos_up", 0)
    ext_bos_down = row.get("ext_bos_down", 0)
    
    # Long setup: in bull FVG + uptrend
    if in_bull_fvg and ext_trend > 0:
        return True, 1
    
    # Short setup: in bear FVG + downtrend
    if in_bear_fvg and ext_trend < 0:
        return True, -1
    
    return False, 0


def calculate_sl_tp(row: pd.Series, side: int, rr: float = 2.0) -> Tuple[float, float]:
    """Calculate SL and TP for a trade."""
    atr_approx = row["high_low_range"] * 2
    
    if side == 1:  # Long
        sl = row["low"] - atr_approx * 0.5
        risk = row["close"] - sl
        tp = row["close"] + risk * rr
    else:  # Short
        sl = row["high"] + atr_approx * 0.5
        risk = sl - row["close"]
        tp = row["close"] - risk * rr
    
    return sl, tp


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: int, sl: float, tp: float, max_bars: int = 100) -> Tuple[str, float, int]:
    """Simulate trade outcome."""
    entry_price = df.iloc[entry_idx]["close"]
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        
        if side == 1:  # Long
            if low <= sl:
                return "loss", -1.0, i
            if high >= tp:
                return "win", 2.0, i
        else:  # Short
            if high >= sl:
                return "loss", -1.0, i
            if low <= tp:
                return "win", 2.0, i
    
    # Timeout
    return "timeout", 0.0, entry_idx + max_bars


def get_context_window(df: pd.DataFrame, idx: int, feature_cols: List[str], seq_len: int = 60) -> np.ndarray:
    """Get context window for ASM inference using fixed feature order.
    
    Uses ASM_FEATURE_COLS to ensure correct feature order matching training.
    Missing features are filled with 0.0.
    """
    if idx < seq_len:
        return None
    
    # Build context using fixed feature order
    context = []
    for i in range(idx - seq_len, idx):
        row = df.iloc[i]
        # Get features in exact order, fill missing with 0.0
        feature_row = [float(row.get(col, 0.0)) for col in feature_cols]
        context.append(feature_row)
    
    context = np.array(context, dtype=np.float32)
    
    # Safety check
    assert context.shape == (seq_len, len(feature_cols)), \
        f"ASM context shape mismatch: got {context.shape}, expected ({seq_len}, {len(feature_cols)})"
    
    # Handle NaN
    context = np.nan_to_num(context, nan=0.0)
    
    return context


def calculate_stats(trades: List[S4Trade]) -> Dict:
    """Calculate trading statistics."""
    if not trades:
        return {"trades": 0, "winrate": 0, "expectancy": 0, "maxdd": 0, "total_R": 0}
    
    pnls = [t.pnl_r for t in trades]
    wins = sum(1 for t in trades if t.outcome == "win")
    
    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    maxdd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    
    return {
        "trades": len(trades),
        "winrate": round(100 * wins / len(trades), 2),
        "expectancy": round(np.mean(pnls), 4),
        "maxdd": round(maxdd, 2),
        "total_R": round(sum(pnls), 2),
    }


# ==============================================================================
# MAIN BACKTEST
# ==============================================================================


def run_shadow_backtest():
    """Run shadow backtest with ASM filter."""
    print("=" * 60)
    print("Shadow Backtest: S4_HighVol_FVG_London + ASM v1.0")
    print("=" * 60)
    
    # Load data
    df = load_feature_data(DATA_PATHS)
    feature_cols = get_feature_columns(df, max_features=ASM_FEATURE_DIM)
    print(f"  Using {len(feature_cols)} features for ASM")
    
    # Load ASM model
    print("\nLoading ASM model...")
    asm_loader = ASMModelV1Loader()
    
    # Generate S4 trades
    print("\nGenerating S4 trades...")
    trades: List[S4Trade] = []
    
    for idx in range(ASM_SEQ_LEN, len(df) - S4_CONFIG["max_bars_in_trade"]):
        row = df.iloc[idx]
        
        is_setup, side = check_s4_setup(row, df, idx)
        if not is_setup:
            continue
        
        # Calculate SL/TP
        sl, tp = calculate_sl_tp(row, side, S4_CONFIG["rr_target"])
        
        # Simulate trade
        outcome, pnl_r, exit_bar = simulate_trade(df, idx, side, sl, tp, S4_CONFIG["max_bars_in_trade"])
        
        # Get ASM context and predict
        context = get_context_window(df, idx, feature_cols, ASM_SEQ_LEN)
        if context is None or np.isnan(context).any():
            continue
        
        asm_probs = asm_loader.predict_proba(context)
        
        # Calculate direction score
        if side == 1:  # Long
            dir_score = asm_probs["p_up"] - asm_probs["p_down"]
        else:  # Short
            dir_score = asm_probs["p_down"] - asm_probs["p_up"]
        
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
            asm_dir_score=dir_score,
        )
        trades.append(trade)
    
    print(f"  Generated {len(trades)} S4 trades")
    
    # Baseline stats
    baseline_stats = calculate_stats(trades)
    print(f"\n[BASELINE] S4_London (no ASM filter):")
    print(f"  Trades: {baseline_stats['trades']}, WR: {baseline_stats['winrate']}%, Exp: {baseline_stats['expectancy']}R, MaxDD: {baseline_stats['maxdd']}R")
    
    # Sweep thresholds
    print("\n" + "=" * 60)
    print("ASM Filter Sweep:")
    print("=" * 60)
    print(f"{'T_shift':<10} {'T_dir':<10} {'Trades':<10} {'WR%':<10} {'Exp(R)':<10} {'MaxDD':<10} {'Retain%':<10}")
    print("-" * 70)
    
    results = []
    
    for t_shift in T_SHIFT_VALUES:
        for t_dir in T_DIR_VALUES:
            # Filter trades
            filtered = [
                t for t in trades
                if t.asm_p_shift >= t_shift and t.asm_dir_score >= t_dir
            ]
            
            stats = calculate_stats(filtered)
            retain_pct = round(100 * len(filtered) / len(trades), 1) if trades else 0
            
            print(f"{t_shift:<10} {t_dir:<10} {stats['trades']:<10} {stats['winrate']:<10} {stats['expectancy']:<10} {stats['maxdd']:<10} {retain_pct:<10}")
            
            results.append({
                "T_shift": t_shift,
                "T_dir": t_dir,
                **stats,
                "retain_pct": retain_pct,
            })
    
    # =========================================================================
    # INVERSE FILTER - LOW p_shift (stable auction)
    # =========================================================================
    print("\n" + "=" * 60)
    print("INVERSE FILTER – LOW p_shift (stable auction)")
    print("=" * 60)
    print(f"{'T_shift_max':<12} {'Trades':<10} {'WR%':<10} {'Exp(R)':<10} {'MaxDD':<10} {'Retain%':<10}")
    print("-" * 62)
    
    inverse_results = []
    
    for t_shift_max in T_SHIFT_MAX_VALUES:
        # Filter: keep trades with LOW p_shift
        filtered = [t for t in trades if t.asm_p_shift <= t_shift_max]
        
        stats = calculate_stats(filtered)
        retain_pct = round(100 * len(filtered) / len(trades), 1) if trades else 0
        
        print(f"{t_shift_max:<12} {stats['trades']:<10} {stats['winrate']:<10} {stats['expectancy']:<10} {stats['maxdd']:<10} {retain_pct:<10}")
        
        inverse_results.append({
            "T_shift_max": t_shift_max,
            **stats,
            "retain_pct": retain_pct,
        })
    
    # =========================================================================
    # NEUTRAL FILTER - HIGH p_neutral
    # =========================================================================
    print("\n" + "=" * 60)
    print("NEUTRAL FILTER – HIGH p_neutral")
    print("=" * 60)
    print(f"{'T_neu_min':<12} {'Trades':<10} {'WR%':<10} {'Exp(R)':<10} {'MaxDD':<10} {'Retain%':<10}")
    print("-" * 62)
    
    neutral_results = []
    
    for t_neu_min in T_NEUTRAL_MIN_VALUES:
        # Filter: keep trades with HIGH p_neutral
        filtered = [t for t in trades if t.asm_p_neutral >= t_neu_min]
        
        stats = calculate_stats(filtered)
        retain_pct = round(100 * len(filtered) / len(trades), 1) if trades else 0
        
        print(f"{t_neu_min:<12} {stats['trades']:<10} {stats['winrate']:<10} {stats['expectancy']:<10} {stats['maxdd']:<10} {retain_pct:<10}")
        
        neutral_results.append({
            "T_neu_min": t_neu_min,
            **stats,
            "retain_pct": retain_pct,
        })
    
    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "meta": {
            "model": "ASM-GRU64-v1.0-C3",
            "date": datetime.now().isoformat(),
            "data_period": "Apr-Jun 2025 (6W)",
            "session": S4_CONFIG["session"],
            "rr_target": S4_CONFIG["rr_target"],
        },
        "baseline": baseline_stats,
        "direct_filter_results": results,
        "direct_filter_best": max(results, key=lambda x: x["expectancy"]) if results else None,
        "inverse_filter_results": inverse_results,
        "inverse_filter_best": max(inverse_results, key=lambda x: x["expectancy"]) if inverse_results else None,
        "neutral_filter_results": neutral_results,
        "neutral_filter_best": max(neutral_results, key=lambda x: x["expectancy"]) if neutral_results else None,
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n[BASELINE] S4_London:")
    print(f"  Trades: {baseline_stats['trades']}, WR: {baseline_stats['winrate']}%, Exp: {baseline_stats['expectancy']}R, MaxDD: {baseline_stats['maxdd']}R")
    
    if output["direct_filter_best"]:
        best = output["direct_filter_best"]
        print(f"\n[DIRECT FILTER BEST] T_shift={best['T_shift']}, T_dir={best['T_dir']}:")
        print(f"  Trades: {best['trades']}, WR: {best['winrate']}%, Exp: {best['expectancy']}R, MaxDD: {best['maxdd']}R, Retain: {best['retain_pct']}%")
    
    if output["inverse_filter_best"]:
        best = output["inverse_filter_best"]
        print(f"\n[INVERSE FILTER BEST] T_shift_max={best['T_shift_max']}:")
        print(f"  Trades: {best['trades']}, WR: {best['winrate']}%, Exp: {best['expectancy']}R, MaxDD: {best['maxdd']}R, Retain: {best['retain_pct']}%")
    
    if output["neutral_filter_best"]:
        best = output["neutral_filter_best"]
        print(f"\n[NEUTRAL FILTER BEST] T_neu_min={best['T_neu_min']}:")
        print(f"  Trades: {best['trades']}, WR: {best['winrate']}%, Exp: {best['expectancy']}R, MaxDD: {best['maxdd']}R, Retain: {best['retain_pct']}%")
    
    return output


if __name__ == "__main__":
    run_shadow_backtest()

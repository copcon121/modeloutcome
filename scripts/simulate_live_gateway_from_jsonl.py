#!/usr/bin/env python3
"""
Shadow Replay Script for Live Gateway
======================================
Replays historical JSONL data through the live gateway to validate
that it reproduces the same edge as offline backtest.

Usage:
    # In-process mode (faster, no HTTP):
    python scripts/simulate_live_gateway_from_jsonl.py
    
    # HTTP mode (test actual API):
    python scripts/simulate_live_gateway_from_jsonl.py --http --port 8000
"""

import argparse
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import requests

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
DATA_PATHS = [
    ROOT / "data/raw/new_data/*.jsonl",  # NEW 6W data
]

# Trade simulation
RR_TARGET = 2.0
MAX_BARS_IN_TRADE = 100


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SimulatedTrade:
    """Simulated trade from gateway signal"""
    bar_index: int
    timestamp: str
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    p_shift: float
    outcome: str = ""  # 'win', 'loss', 'timeout'
    pnl_r: float = 0.0
    exit_bar: int = 0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_raw_bars(data_paths: List[Path]) -> List[Dict]:
    """Load all raw bars from JSONL files, sorted by timestamp"""
    all_bars = []
    
    for pattern in data_paths:
        files = glob.glob(str(pattern))
        print(f"Found {len(files)} files matching {pattern}")
        
        for f in sorted(files):
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        bar = json.loads(line)
                        all_bars.append(bar)
                    except:
                        continue
    
    # Sort by timestamp
    all_bars.sort(key=lambda x: x['timestamp'])
    print(f"Total bars loaded: {len(all_bars)}")
    
    return all_bars


def simulate_trade_outcome(
    bars: List[Dict],
    entry_idx: int,
    side: str,
    sl: float,
    tp: float,
    max_bars: int = MAX_BARS_IN_TRADE
) -> Tuple[str, float, int]:
    """Simulate trade outcome using future bars"""
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(bars))):
        bar_data = bars[i]['bar']
        high = bar_data['h']
        low = bar_data['l']
        
        if side == "long":
            if low <= sl:
                return "loss", -1.0, i
            if high >= tp:
                return "win", 2.0, i
        else:  # short
            if high >= sl:
                return "loss", -1.0, i
            if low <= tp:
                return "win", 2.0, i
    
    return "timeout", 0.0, entry_idx + max_bars


def calculate_stats(trades: List[SimulatedTrade]) -> Dict:
    """Calculate trading statistics"""
    if not trades:
        return {"trades": 0, "winrate": 0, "expectancy": 0, "maxdd": 0, "total_R": 0}
    
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


# ==============================================================================
# IN-PROCESS REPLAY (No HTTP)
# ==============================================================================

def replay_inprocess(bars: List[Dict]) -> List[SimulatedTrade]:
    """Replay bars through gateway in-process (faster)"""
    from services.live_gateway.context_store import context_store, ContextStore
    from services.live_gateway.s4_engine import check_s4_setup, S4_CONFIG
    from services.live_gateway.asm_inference import asm_model
    from src.layer2_feature_engine_v2.schema import RawBar
    
    # Reset context store
    context_store.contexts = {}
    
    # Load ASM model
    if not asm_model.loaded:
        asm_model.load()
    
    trades: List[SimulatedTrade] = []
    signals_generated = 0
    
    print("\nReplaying bars through gateway...")
    progress_interval = max(1, len(bars) // 20)
    
    for idx, bar_json in enumerate(bars):
        if idx % progress_interval == 0:
            pct = 100 * idx / len(bars)
            print(f"  Progress: {pct:.0f}% ({idx}/{len(bars)})")
        
        # Skip if not enough future bars for trade simulation
        if idx >= len(bars) - MAX_BARS_IN_TRADE:
            continue
        
        # Parse bar
        ts = datetime.fromisoformat(bar_json['timestamp'].replace('Z', '+00:00').split('.')[0])
        bar_data = bar_json['bar']
        tick_data = bar_json.get('tick_features', {})
        
        # Create RawBar
        raw_bar = RawBar(
            symbol=bar_json.get('symbol', 'GC'),
            timeframe=bar_json.get('timeframe', 'M1'),
            timestamp=ts,
            bar_index=bar_json['bar_index'],
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
            vwap_daily=bar_data.get('vwap_daily', 0.0),
        )
        
        symbol = raw_bar.symbol
        timeframe = raw_bar.timeframe
        
        # Update context
        feature_bar, feature_dict = context_store.update(symbol, timeframe, raw_bar)
        
        # Check high vol
        is_high_vol = context_store.is_high_vol_regime(symbol, timeframe)
        
        # Check S4 setup
        hour = ts.hour
        s4_setup = check_s4_setup(
            feature_bar, feature_dict, is_high_vol, hour,
            session_filter=S4_CONFIG["session"]
        )
        
        if not s4_setup.is_valid:
            continue
        
        # Get ASM context
        asm_context = context_store.get_asm_context(symbol, timeframe)
        if asm_context is None:
            continue
        
        # Check for NaN
        if np.isnan(asm_context).any():
            continue
        
        # Run ASM inference
        asm_probs = asm_model.predict_proba(asm_context)
        
        # Apply LowShift filter
        if asm_probs["p_shift"] > 0.2:
            continue
        
        signals_generated += 1
        
        # Simulate trade outcome
        side = "long" if s4_setup.side == 1 else "short"
        outcome, pnl_r, exit_bar = simulate_trade_outcome(
            bars, idx, side, s4_setup.sl_price, s4_setup.tp_price
        )
        
        trade = SimulatedTrade(
            bar_index=idx,
            timestamp=str(ts),
            symbol=symbol,
            side=side,
            entry=s4_setup.entry_price,
            sl=s4_setup.sl_price,
            tp=s4_setup.tp_price,
            p_shift=asm_probs["p_shift"],
            outcome=outcome,
            pnl_r=pnl_r,
            exit_bar=exit_bar,
        )
        trades.append(trade)
    
    print(f"  Progress: 100% ({len(bars)}/{len(bars)})")
    print(f"\nSignals generated: {signals_generated}")
    print(f"Trades simulated: {len(trades)}")
    
    return trades


# ==============================================================================
# HTTP REPLAY
# ==============================================================================

def replay_http(bars: List[Dict], base_url: str = "http://localhost:8000") -> List[SimulatedTrade]:
    """Replay bars through gateway via HTTP"""
    trades: List[SimulatedTrade] = []
    signals_generated = 0
    
    print(f"\nReplaying bars via HTTP to {base_url}...")
    progress_interval = max(1, len(bars) // 20)
    
    for idx, bar_json in enumerate(bars):
        if idx % progress_interval == 0:
            pct = 100 * idx / len(bars)
            print(f"  Progress: {pct:.0f}% ({idx}/{len(bars)})")
        
        if idx >= len(bars) - MAX_BARS_IN_TRADE:
            continue
        
        # Send to gateway
        try:
            response = requests.post(
                f"{base_url}/live_bar",
                json=bar_json,
                timeout=5
            )
            result = response.json()
        except Exception as e:
            print(f"  Error at bar {idx}: {e}")
            continue
        
        if not result.get("has_signal"):
            continue
        
        signals_generated += 1
        
        # Simulate trade outcome
        side = result["side"]
        sl = result["sl"]
        tp = result["tp"]
        
        outcome, pnl_r, exit_bar = simulate_trade_outcome(bars, idx, side, sl, tp)
        
        trade = SimulatedTrade(
            bar_index=idx,
            timestamp=bar_json['timestamp'],
            symbol=result["symbol"],
            side=side,
            entry=result["entry"],
            sl=sl,
            tp=tp,
            p_shift=result.get("p_shift", 0),
            outcome=outcome,
            pnl_r=pnl_r,
            exit_bar=exit_bar,
        )
        trades.append(trade)
    
    print(f"  Progress: 100% ({len(bars)}/{len(bars)})")
    print(f"\nSignals generated: {signals_generated}")
    print(f"Trades simulated: {len(trades)}")
    
    return trades


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Shadow replay for live gateway")
    parser.add_argument("--http", action="store_true", help="Use HTTP mode instead of in-process")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--data", type=str, default=None, help="Custom data path pattern")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SHADOW REPLAY: Live Gateway Validation")
    print("=" * 70)
    
    # Load data
    data_paths = DATA_PATHS
    if args.data:
        data_paths = [Path(args.data)]
    
    bars = load_raw_bars(data_paths)
    
    if len(bars) == 0:
        print("ERROR: No bars loaded!")
        return
    
    # Replay
    if args.http:
        trades = replay_http(bars, f"http://localhost:{args.port}")
    else:
        trades = replay_inprocess(bars)
    
    # Calculate stats
    stats = calculate_stats(trades)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Live Gateway Shadow Replay")
    print("=" * 70)
    print(f"  Trades:     {stats['trades']}")
    print(f"  Winrate:    {stats['winrate']}%")
    print(f"  Expectancy: {stats['expectancy']}R")
    print(f"  MaxDD:      {stats['maxdd']}R")
    print(f"  Total R:    {stats['total_R']}R")
    
    # Compare with backtest
    print("\n" + "=" * 70)
    print("COMPARISON with Offline Backtest (NEW 6W)")
    print("=" * 70)
    print("  Backtest S4_LDN + ASM LowShift 0.2:")
    print("    Trades: 258, WR: 61.6%, Exp: +0.85R, MaxDD: 26R")
    print(f"\n  Gateway Replay:")
    print(f"    Trades: {stats['trades']}, WR: {stats['winrate']}%, Exp: {stats['expectancy']}R, MaxDD: {stats['maxdd']}R")
    
    # Save results
    output_path = ROOT / "backtests/live_gateway_replay_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "meta": {
                "date": datetime.now().isoformat(),
                "mode": "http" if args.http else "inprocess",
                "data_paths": [str(p) for p in data_paths],
                "total_bars": len(bars),
            },
            "stats": stats,
            "comparison": {
                "backtest_trades": 258,
                "backtest_winrate": 61.6,
                "backtest_expectancy": 0.85,
                "backtest_maxdd": 26.0,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

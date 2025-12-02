"""
Risk Parameter Sweep for Phase 3
Tests multiple ATR/RR configurations to find optimal baseline

This script is STANDALONE - it doesn't modify existing Phase 3 code.
It loads the already-labeled events and recomputes outcomes with different risk configs.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config as phase3_config


@dataclass
class RiskConfig:
    """Risk management configuration"""
    atr_mult_sl: float  # ATR multiplier for SL (e.g., 1.5)
    rr_target: float    # Risk-reward ratio for TP (e.g., 3.0)
    
    def __str__(self):
        return f"ATR={self.atr_mult_sl}, RR={self.rr_target}"


def compute_sl_tp(entry: float, is_long: bool, bar_range: float, cfg: RiskConfig) -> Tuple[float, float]:
    """
    Compute SL and TP prices
    
    Args:
        entry: Entry price
        is_long: True for long, False for short
        bar_range: Bar high-low range (ATR proxy)
        cfg: RiskConfig with atr_mult_sl and rr_target
        
    Returns:
        (sl_price, tp_price)
    """
    tick_size = 0.1  # GC
    buffer_ticks = 2
    buffer = buffer_ticks * tick_size
    
    atr_estimate = abs(bar_range) if bar_range > 0 else 5.0  # Fallback to 5.0 if missing
    
    if is_long:
        # LONG: SL below, TP above
        sl = entry - (atr_estimate * cfg.atr_mult_sl) - buffer
        risk = entry - sl
        tp = entry + (cfg.rr_target * risk)
    else:
        # SHORT: SL above, TP below
        sl = entry + (atr_estimate * cfg.atr_mult_sl) + buffer
        risk = sl - entry
        tp = entry - (cfg.rr_target * risk)
    
    # Safety check: ensure SL/TP are on correct sides
    if is_long:
        assert sl < entry < tp, f"Long SL/TP error: sl={sl:.2f},  entry={entry:.2f}, tp={tp:.2f}"
    else:
        assert tp < entry < sl, f"Short SL/TP error: tp={tp:.2f}, entry={entry:.2f}, sl={sl:.2f}"
    
    return sl, tp


def simulate_outcome(direction: str, entry: float, sl: float, tp: float, 
                     risk: float, future_bars: pd.DataFrame, max_hold: int = 100) -> Dict:
    """
    Simulate forward to determine outcome
    
    Returns dict with: hit, outcome_rr, hold_bars, exit_price
    """
    if len(future_bars) < 10:  # Min future bars
        return {'hit': 'none', 'outcome_rr': 0.0, 'hold_bars': 0, 'exit_price': entry}
    
    max_bars = min(max_hold, len(future_bars))
    
    for i in range(max_bars):
        bar = future_bars.iloc[i]
        
        if direction == "long":
            if bar['high'] >= tp:
                return {'hit': 'tp', 'outcome_rr': risk > 0 and (tp - entry) / risk or 0, 
                       'hold_bars': i + 1, 'exit_price': tp}
            if bar['low'] <= sl:
                return {'hit': 'sl', 'outcome_rr': -1.0, 
                       'hold_bars': i + 1, 'exit_price': sl}
        else:  # short
            if bar['low'] <= tp:
                return {'hit': 'tp', 'outcome_rr': risk > 0 and (entry - tp) / risk or 0, 
                       'hold_bars': i + 1, 'exit_price': tp}
            if bar['high'] >= sl:
                return {'hit': 'sl', 'outcome_rr': -1.0, 
                       'hold_bars': i + 1, 'exit_price': sl}
    
    # Max hold reached
    final_bar = future_bars.iloc[max_bars - 1]
    exit_price = final_bar['close']
    actual_pnl = exit_price - entry if direction == "long" else entry - exit_price
    outcome_rr = actual_pnl / risk if abs(risk) > 1e-6 else 0.0
    
    return {'hit': 'none', 'outcome_rr': outcome_rr, 
            'hold_bars': max_bars, 'exit_price': exit_price}


def load_data():
    """Load full bars and P2 labeled events"""
    print("\n[1/5] Loading data...")
    
    # Load full bar stream (for OHLCV)
    bars_path = Path("output/production_10weeks/features_with_ohlcv_10weeks.csv")
    print(f"  Loading bars: {bars_path}")
    df_bars = pd.read_csv(bars_path)
    df_bars['timestamp'] = pd.to_datetime(df_bars['timestamp'])
    print(f"  Loaded {len(df_bars):,} bars")
    
    # Load labeled events (already has signal_side from SMC rules)
    events_path = Path("output/phase3_labeled/events_p2_labeled_10weeks.jsonl")
    print(f"  Loading labeled events: {events_path}")
    
    events = []
    with open(events_path) as f:
        for line in f:
            events.append(json.loads(line))
    
    print(f"  Loaded {len(events):,} labeled events")
    
    # Filter to actionable only (long/short, not skip)
    actionable = [e for e in events if e['signal_side'] in ['long', 'short']]
    print(f"  Actionable signals: {len(actionable):,}")
    
    return df_bars, actionable


def recompute_outcomes_for_config(df_bars: pd.DataFrame, events: List[Dict], 
                                   cfg: RiskConfig) -> List[Dict]:
    """
    Recompute SL/TP and outcomes for all events using given RiskConfig
    """
    results = []
    
    for event in events:
        # Get event bar data
        bar_idx = event['global_bar_index']
        event_bar = df_bars.iloc[bar_idx]
        
        entry = float(event_bar['close'])
        bar_range = float(event_bar['high']) - float(event_bar['low'])
        is_long = event['signal_side'] == 'long'
        
        # Compute SL/TP with this config
        try:
            sl, tp = compute_sl_tp(entry, is_long, bar_range, cfg)
        except AssertionError as e:
            # Skip trades with invalid SL/TP
            continue
        
        risk = abs(entry - sl)
        
        # Get future bars
        future_start = bar_idx + 1
        future_end = min(future_start + 100, len(df_bars))
        future_bars = df_bars.iloc[future_start:future_end]
        
        # Simulate outcome
        outcome = simulate_outcome(
            direction=event['signal_side'],
            entry=entry,
            sl=sl,
            tp=tp,
            risk=risk,
            future_bars=future_bars
        )
        
        results.append({
            'event_id': event['event_id'],
            'signal_side': event['signal_side'],
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'risk': risk,
            **outcome
        })
    
    return results


def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute backtest metrics from trade results"""
    if not trades:
        return {}
    
    total = len(trades)
    winners = len([t for t in trades if t['hit'] == 'tp'])
    losers = len([t for t in trades if t['hit'] == 'sl'])
    none_hits = len([t for t in trades if t['hit'] == 'none'])
    
    winrate = winners / total * 100 if total > 0 else 0
    
    # R metrics
    rs = [t['outcome_rr'] for t in trades]
    total_r = sum(rs)
    avg_r = total_r / total if total > 0 else 0
    
    winner_rs = [t['outcome_rr'] for t in trades if t['hit'] == 'tp']
    loser_rs = [t['outcome_rr'] for t in trades if t['hit'] == 'sl']
    
    avg_winner = sum(winner_rs) / len(winner_rs) if winner_rs else 0
    avg_loser = sum(loser_rs) / len(loser_rs) if loser_rs else 0
    
    expectancy = (winrate/100 * avg_winner) + ((100-winrate)/100 * avg_loser) if losers else avg_winner
    
    # Profit factor
    total_wins = sum(winner_rs) if winner_rs else 0
    total_losses = abs(sum(loser_rs)) if loser_rs else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    # Drawdown
    equity = 0
    max_equity = 0
    max_dd = 0
    
    for t in trades:
        equity += t['outcome_rr']
        if equity > max_equity:
            max_equity = equity
        dd = max_equity - equity
        if dd > max_dd:
            max_dd = dd
    
    # By direction
    longs = [t for t in trades if t['signal_side'] == 'long']
    shorts = [t for t in trades if t['signal_side'] == 'short']
    
    long_winners = len([t for t in longs if t['hit'] == 'tp'])
    short_winners = len([t for t in shorts if t['hit'] == 'tp'])
    
    long_wr = long_winners / len(longs) * 100 if longs else 0
    short_wr = short_winners / len(shorts) * 100 if shorts else 0
    
    long_avg_r = sum(t['outcome_rr'] for t in longs) / len(longs) if longs else 0
    short_avg_r = sum(t['outcome_rr'] for t in shorts) / len(shorts) if shorts else 0
    
    return {
        'total_trades': total,
        'winners': winners,
        'losers': losers,
        'none_hits': none_hits,
        'winrate': winrate,
        'avg_r': avg_r,
        'expectancy': expectancy,
        'total_r': total_r,
        'profit_factor': profit_factor,
        'max_dd_r': max_dd,
        'long_trades': len(longs),
        'long_wr': long_wr,
        'long_avg_r': long_avg_r,
        'short_trades': len(shorts),
        'short_wr': short_wr,
        'short_avg_r': short_avg_r
    }


def run_sweep():
    """Main sweep function"""
    print("="*80)
    print("PHASE 3 RISK PARAMETER SWEEP")
    print("="*80)
    
    # Load data once
    df_bars, events = load_data()
    
    # Define grid
    ATR_MULTS = [0.5, 1.0, 1.5, 2.0]
    RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
    
    total_configs = len(ATR_MULTS) * len(RR_TARGETS)
    print(f"\n[2/5] Testing {total_configs} configurations...")
    print(f"  ATR multipliers: {ATR_MULTS}")
    print(f"  RR targets: {RR_TARGETS}")
    
    # Run sweep
    print("\n[3/5] Computing outcomes for each configuration...")
    
    results_table = []
    
    for atr_mult in ATR_MULTS:
        for rr_target in RR_TARGETS:
            cfg = RiskConfig(atr_mult_sl=atr_mult, rr_target=rr_target)
            
            # Recompute outcomes
            trades = recompute_outcomes_for_config(df_bars, events, cfg)
            
            # Compute metrics
            metrics = compute_metrics(trades)
            
            if metrics:
                row = {
                    'atr_mult_sl': atr_mult,
                    'rr_target': rr_target,
                    **metrics
                }
                results_table.append(row)
                
                print(f"  {cfg}: {metrics['total_trades']} trades, "
                      f"{metrics['winrate']:.1f}% WR, {metrics['expectancy']:+.2f}R exp")
    
    # Create DataFrame and sort
    print("\n[4/5] Analyzing results...")
    df_results = pd.DataFrame(results_table)
    
    # Sort by expectancy (descending), then by max_dd (ascending)
    df_results = df_results.sort_values(['expectancy', 'max_dd_r'], 
                                        ascending=[False, True])
    
    # Save results
    output_dir = Path("output/phase3_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "risk_sweep_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved CSV: {csv_path}")
    
    # Create pretty text report
    txt_path = output_dir / "risk_sweep_results.txt"
    with open(txt_path, 'w') as f:
        f.write("PHASE 3 RISK PARAMETER SWEEP RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total configurations tested: {len(df_results)}\n")
        f.write(f"Sorted by: Expectancy (desc), Max DD (asc)\n\n")
        f.write("="*80 + "\n")
        f.write(df_results.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    
    print(f"[OK] Saved report: {txt_path}")
    
    # Print top 5
    print("\n[5/5] TOP 5 CONFIGURATIONS BY EXPECTANCY:")
    print("="*80)
    
    for i, row in enumerate(df_results.head(5).itertuples(), 1):
        print(f"\n{i}. ATR={row.atr_mult_sl}, RR={row.rr_target}")
        print(f"   Expectancy: {row.expectancy:+.2f}R")
        print(f"   Winrate: {row.winrate:.1f}%")
        print(f"   Total R: {row.total_r:+.1f}R")
        print(f"   Max DD: {row.max_dd_r:.1f}R")
        print(f"   Profit Factor: {row.profit_factor:.2f}")
        print(f"   Trades: {row.total_trades} (Long: {row.long_trades}, Short: {row.short_trades})")
    
    print("\n" + "="*80)
    print("SWEEP COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")
    print("\nRecommendation: Choose a config with high expectancy and acceptable max DD")


if __name__ == '__main__':
    run_sweep()

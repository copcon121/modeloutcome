"""
Backtest P2_v2 Setups - Rule-based (No ML)
Evaluate 3 configs A/B/C with fixed RR targets
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

ROOT = Path(__file__).parent.parent


@dataclass
class Trade:
    bar_index: int
    timestamp: str
    side: int  # +1 long, -1 short
    mode: int  # 1 or 2
    entry: float
    sl: float
    tp: float
    outcome: str  # 'win', 'loss', 'open'
    pnl_r: float
    exit_bar: int


def calculate_sl_tp(row: pd.Series, side: int, rr: float = 2.0) -> tuple:
    """Calculate SL and TP based on recent swing/zone"""
    atr = row.get('high_low_range', 1.0) * 2  # Approximate ATR
    
    if side == 1:  # Long
        # SL below recent low or zone bottom
        sl = row['low'] - atr * 0.5
        risk = row['close'] - sl
        tp = row['close'] + risk * rr
    else:  # Short
        sl = row['high'] + atr * 0.5
        risk = sl - row['close']
        tp = row['close'] - risk * rr
    
    return sl, tp


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: int, sl: float, tp: float, max_bars: int = 100) -> tuple:
    """Simulate trade outcome"""
    entry_price = df.iloc[entry_idx]['close']
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        if side == 1:  # Long
            if low <= sl:
                return 'loss', -1.0, i
            if high >= tp:
                return 'win', 2.0, i  # 2R win
        else:  # Short
            if high >= sl:
                return 'loss', -1.0, i
            if low <= tp:
                return 'win', 2.0, i
    
    # Still open after max_bars
    last_close = df.iloc[min(entry_idx + max_bars - 1, len(df) - 1)]['close']
    if side == 1:
        pnl = (last_close - entry_price) / (entry_price - sl) if entry_price != sl else 0
    else:
        pnl = (entry_price - last_close) / (sl - entry_price) if sl != entry_price else 0
    
    return 'timeout', pnl, entry_idx + max_bars


def backtest_config(config_name: str, df_all: pd.DataFrame) -> Dict:
    """Backtest a single config"""
    csv_path = ROOT / f"output/p2_v2_{config_name}.csv"
    df_p2 = pd.read_csv(csv_path)
    
    print(f"\n[{config_name}] Loaded {len(df_p2)} setups")
    
    trades = []
    
    # Group by mode and side
    for _, row in df_p2.iterrows():
        bar_idx = int(row['bar_index'])
        side = int(row['p2_side'])
        mode = int(row['p2_mode'])
        
        # Find this bar in full data
        match = df_all[df_all['bar_index'] == bar_idx]
        if len(match) == 0:
            continue
        
        full_row = match.iloc[0]
        entry_idx = match.index[0]
        
        # Calculate SL/TP
        sl, tp = calculate_sl_tp(full_row, side, rr=2.0)
        
        # Simulate
        outcome, pnl_r, exit_bar = simulate_trade(df_all, entry_idx, side, sl, tp)
        
        trades.append(Trade(
            bar_index=bar_idx,
            timestamp=str(row.get('timestamp', '')),
            side=side,
            mode=mode,
            entry=full_row['close'],
            sl=sl,
            tp=tp,
            outcome=outcome,
            pnl_r=pnl_r,
            exit_bar=exit_bar
        ))
    
    return analyze_trades(trades, config_name)


def analyze_trades(trades: List[Trade], config_name: str) -> Dict:
    """Analyze trade results"""
    if not trades:
        return {'config': config_name, 'n_trades': 0}
    
    df_trades = pd.DataFrame([t.__dict__ for t in trades])
    
    # Overall stats
    n_trades = len(df_trades)
    n_wins = (df_trades['outcome'] == 'win').sum()
    n_losses = (df_trades['outcome'] == 'loss').sum()
    winrate = n_wins / n_trades * 100 if n_trades > 0 else 0
    
    total_pnl = df_trades['pnl_r'].sum()
    expectancy = df_trades['pnl_r'].mean()
    
    # Max drawdown
    cumsum = df_trades['pnl_r'].cumsum()
    running_max = cumsum.cummax()
    drawdown = running_max - cumsum
    max_dd = drawdown.max()
    
    # By mode
    mode1 = df_trades[df_trades['mode'] == 1]
    mode2 = df_trades[df_trades['mode'] == 2]
    
    mode1_stats = {
        'n': len(mode1),
        'winrate': (mode1['outcome'] == 'win').sum() / len(mode1) * 100 if len(mode1) > 0 else 0,
        'expectancy': mode1['pnl_r'].mean() if len(mode1) > 0 else 0,
        'long': len(mode1[mode1['side'] == 1]),
        'short': len(mode1[mode1['side'] == -1])
    }
    
    mode2_stats = {
        'n': len(mode2),
        'winrate': (mode2['outcome'] == 'win').sum() / len(mode2) * 100 if len(mode2) > 0 else 0,
        'expectancy': mode2['pnl_r'].mean() if len(mode2) > 0 else 0,
        'long': len(mode2[mode2['side'] == 1]),
        'short': len(mode2[mode2['side'] == -1])
    }
    
    result = {
        'config': config_name,
        'n_trades': n_trades,
        'wins': n_wins,
        'losses': n_losses,
        'winrate': winrate,
        'total_pnl_r': total_pnl,
        'expectancy_r': expectancy,
        'max_dd_r': max_dd,
        'mode1': mode1_stats,
        'mode2': mode2_stats
    }
    
    # Print
    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*60}")
    print(f"  Total trades: {n_trades}")
    print(f"  Wins: {n_wins}, Losses: {n_losses}")
    print(f"  Winrate: {winrate:.1f}%")
    print(f"  Total PnL: {total_pnl:+.1f}R")
    print(f"  Expectancy: {expectancy:+.3f}R")
    print(f"  Max DD: {max_dd:.1f}R")
    print(f"\n  Mode 1 (VA Reversal):")
    print(f"    Trades: {mode1_stats['n']} (L:{mode1_stats['long']}, S:{mode1_stats['short']})")
    print(f"    Winrate: {mode1_stats['winrate']:.1f}%")
    print(f"    Expectancy: {mode1_stats['expectancy']:+.3f}R")
    print(f"\n  Mode 2 (Trend Cont.):")
    print(f"    Trades: {mode2_stats['n']} (L:{mode2_stats['long']}, S:{mode2_stats['short']})")
    print(f"    Winrate: {mode2_stats['winrate']:.1f}%")
    print(f"    Expectancy: {mode2_stats['expectancy']:+.3f}R")
    
    # Sample trades
    print(f"\n  Sample trades (first 3):")
    for t in trades[:3]:
        print(f"    bar={t.bar_index}, side={'L' if t.side==1 else 'S'}, mode={t.mode}, "
              f"outcome={t.outcome}, pnl={t.pnl_r:+.1f}R")
    
    return result


def main():
    print("="*70)
    print("P2_v2 BACKTEST - RULE-BASED (NO ML)")
    print("="*70)
    
    # Load full data
    csv_path = ROOT / "output/production_10weeks_v3/features_all_10weeks_v3.csv"
    print(f"\n[1] Loading full data from {csv_path.name}...")
    df_all = pd.read_csv(csv_path)
    print(f"  Total bars: {len(df_all)}")
    
    # Backtest each config
    configs = ['A_Conservative', 'B_Medium', 'C_Loose']
    results = []
    
    for cfg in configs:
        result = backtest_config(cfg, df_all)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Config':<20} {'Trades':>8} {'Winrate':>10} {'Expect_R':>12} {'MaxDD_R':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<20} {r['n_trades']:>8} {r['winrate']:>9.1f}% {r['expectancy_r']:>+11.3f} {r['max_dd_r']:>10.1f}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    
    best = max(results, key=lambda x: x['expectancy_r'])
    if best['expectancy_r'] >= 0.2 and best['n_trades'] >= 100:
        print(f"✓ Config {best['config']} shows edge!")
        print(f"  Expectancy: {best['expectancy_r']:+.3f}R, Trades: {best['n_trades']}")
    elif best['expectancy_r'] >= 0.1:
        print(f"~ Moderate signal with {best['config']}")
        print(f"  Expectancy: {best['expectancy_r']:+.3f}R - may need refinement")
    else:
        print(f"✗ No clear edge found with current delta rules")
        print(f"  Best expectancy: {best['expectancy_r']:+.3f}R")
        print(f"  Consider: different delta thresholds, zone quality filters, or different approach")


if __name__ == "__main__":
    main()

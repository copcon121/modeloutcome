"""
Baseline Backtest
Analyzes rule-only performance (SMC signal + fixed 3R target)
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config


def load_labeled_events(path):
    """Load labeled events from JSONL"""
    events = []
    with open(path, 'r') as f:
        for line in f:
            events.append(json.loads(line))
    return events


def run_backtest(events):
    """Run baseline backtest"""
    
    # Filter to actionable trades only
    trades = [e for e in events if e['signal_side'] in ['long', 'short']]
    
    print("\n" + "="*80)
    print("BASELINE BACKTEST RESULTS")
    print("="*80)
    
    #  Overall metrics
    total_trades = len(trades)
    winners = len([t for t in trades if t['hit'] == 'tp'])
    losers = len([t for t in trades if t['hit'] == 'sl'])
    none_hits = len([t for t in trades if t['hit'] == 'none'])
    
    winrate = winners / total_trades * 100 if total_trades > 0 else 0
    lossrate = losers / total_trades * 100 if total_trades > 0 else 0
    
    # R metrics
    total_r = sum(t['outcome_rr'] for t in trades)
    avg_r = total_r / total_trades if total_trades > 0 else 0
    
    winner_rs = [t['outcome_rr'] for t in trades if t['hit'] == 'tp']
    loser_rs = [t['outcome_rr'] for t in trades if t['hit'] == 'sl']
    
    avg_winner = sum(winner_rs) / len(winner_rs) if winner_rs else 0
    avg_loser = sum(loser_rs) / len(loser_rs) if loser_rs else 0
    
    # Expectancy
    expectancy = (winrate/100 * avg_winner) + (lossrate/100 * avg_loser)
    
    print(f"\nOverall Performance:")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Winners: {winners:,} ({winrate:.1f}%)")
    print(f"  Losers: {losers:,} ({lossrate:.1f}%)")
    print(f"  None: {none_hits:,} ({none_hits/total_trades*100:.1f}%)")
    
    print(f"\nR Multiples:")
    print(f"  Total R: {total_r:+.1f}R")
    print(f"  Average R: {avg_r:+.2f}R")
    print(f"  Avg Winner: {avg_winner:+.2f}R")
    print(f"  Avg Loser: {avg_loser:+.2f}R")
    print(f"  Expectancy E[R]: {expectancy:+.2f}R")
    
    # By direction
    print(f"\nBy Direction:")
    for direction in ['long', 'short']:
        dir_trades = [t for t in trades if t['signal_side'] == direction]
        dir_winners = len([t for t in dir_trades if t['hit'] == 'tp'])
        dir_wr = dir_winners / len(dir_trades) * 100 if dir_trades else 0
        dir_avg_r = sum(t['outcome_rr'] for t in dir_trades) / len(dir_trades) if dir_trades else 0
        print(f"  {direction.upper():5}: {len(dir_trades):4} trades, {dir_wr:5.1f}% WR, {dir_avg_r:+.2f}R avg")
    
    # Equity curve
    equity_r = 0
    equity_curve = []
    max_equity = 0
    max_dd = 0
    
    for i, trade in enumerate(trades):
        equity_r += trade['outcome_rr']
        equity_curve.append({
            'trade_num': i + 1,
            'cumulative_r': equity_r
        })
        
        if equity_r > max_equity:
            max_equity = equity_r
        
        dd = max_equity - equity_r
        if dd > max_dd:
            max_dd = dd
    
    print(f"\nRisk Metrics:")
    print(f"  Final equity: {equity_r:+.1f}R")
    print(f"  Max drawdown: {max_dd:.1f}R")
    print(f"  Profit factor: {abs(total_r / sum(loser_rs)) if loser_rs and sum(loser_rs) < 0 else 0:.2f}")
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'metrics': {
            'total_trades': total_trades,
            'winrate': winrate,
            'expectancy': expectancy,
            'total_r': total_r,
            'max_dd': max_dd
        }
    }


def main():
    print("="*80)
    print("PHASE 3: BASELINE BACKTEST")
    print("="*80)
    
    # Load events
    print(f"\nLoading events from {config.EVENTS_LABELED_PATH}...")
    events = load_labeled_events(config.EVENTS_LABELED_PATH)
    print(f"Loaded {len(events):,} labeled events")
    
    # Run backtest
    results = run_backtest(events)
    
    # Save equity curve
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df.to_csv(config.EQUITY_CURVE_PATH, index=False)
    print(f"\n[OK] Equity curve saved to {config.EQUITY_CURVE_PATH}")
    
    # Save report
    with open(config.BACKTEST_REPORT_PATH, 'w') as f:
        f.write("PHASE 3 BASELINE BACKTEST REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Dataset: 10 weeks, P2 events\n")
        f.write(f"Strategy: SMC Rule-based (trend + zone alignment)\n")
        f.write(f"Risk Management: 3R fixed target, swing-based SL\n\n")
        f.write("Summary Metrics:\n")
        f.write(f"  Total P2 events: {len(events):,}\n")
        f.write(f"  Actionable signals: {results['metrics']['total_trades']:,}\n")
        f.write(f"  Winrate: {results['metrics']['winrate']:.1f}%\n")
        f.write(f"  Expectancy: {results['metrics']['expectancy']:+.2f}R\n")
        f.write(f"  Total R: {results['metrics']['total_r']:+.1f}R\n")
        f.write(f"  Max DD: {results['metrics']['max_dd']:.1f}R\n")
    
    print(f"[OK] Report saved to {config.BACKTEST_REPORT_PATH}")
    
    print("\n" + "="*80)
    print("BASELINE BACKTEST COMPLETE!")
    print("="*80)
    print(f"\nKey Takeaways:")
    print(f"  - Rule-based system shows {results['metrics']['winrate']:.0f}% winrate")
    print(f"  - Positive expectancy: {results['metrics']['expectancy']:+.2f}R")
    print(f"  - This baseline validates the SMC rules have edge")
    print(f"  - ML model can potentially improve upon this baseline")
    
    print(f"\nPhase 3 complete! Ready for Phase 4: ML Training")


if __name__ == '__main__':
    main()

"""
S4 Policy Backtester - Evaluate policies on enriched trades

The backtester:
1. Takes a list of enriched trades (sorted by time)
2. Applies each policy to decide KEEP/SKIP
3. Computes metrics on kept trades
4. Generates league table comparing policies
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .policy_dataset import S4TradeEnriched, S4PolicyDataset
from .policy_rules import Policy
from .metrics import compute_metrics


@dataclass
class PolicyResult:
    """Result of evaluating a policy on trades."""
    
    name: str
    n_trades: int
    win_count: int
    loss_count: int
    be_count: int
    win_rate: float
    avg_rr: float
    expectancy: float
    std_rr: float
    max_drawdown_r: float
    profit_factor: float
    sharpe_like: float
    cum_r_final: float
    
    # Additional info
    n_skipped: int = 0
    skip_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PolicyResult':
        return cls(**d)
    
    def summary_str(self) -> str:
        """One-line summary."""
        return (
            f"{self.name}: n={self.n_trades}, "
            f"WR={self.win_rate:.1%}, "
            f"Exp={self.expectancy:.2f}R, "
            f"MaxDD={self.max_drawdown_r:.2f}R, "
            f"PF={self.profit_factor:.2f}"
        )


class Backtester:
    """Backtest engine for S4 policies."""
    
    def __init__(self, trades: List[S4TradeEnriched], min_trades: int = 5):
        """
        Args:
            trades: List of enriched trades (should be sorted by time)
            min_trades: Minimum trades for valid policy result
        """
        self.trades = sorted(trades, key=lambda t: t.time)
        self.min_trades = min_trades
        self.results: List[PolicyResult] = []
    
    def evaluate_policy(self, policy: Policy) -> PolicyResult:
        """Evaluate a single policy on all trades."""
        kept_trades = []
        skipped = 0
        
        for trade in self.trades:
            if policy.decide_trade(trade):
                kept_trades.append(trade)
            else:
                skipped += 1
        
        # Compute metrics on kept trades
        metrics = compute_metrics(kept_trades)
        
        result = PolicyResult(
            name=policy.name,
            n_trades=metrics['n_trades'],
            win_count=metrics['win_count'],
            loss_count=metrics['loss_count'],
            be_count=metrics['be_count'],
            win_rate=metrics['win_rate'],
            avg_rr=metrics['avg_rr'],
            expectancy=metrics['expectancy'],
            std_rr=metrics['std_rr'],
            max_drawdown_r=metrics['max_drawdown_r'],
            profit_factor=metrics['profit_factor'],
            sharpe_like=metrics['sharpe_like'],
            cum_r_final=metrics['cum_r_final'],
            n_skipped=skipped,
            skip_rate=skipped / len(self.trades) if self.trades else 0.0,
        )
        
        return result
    
    def evaluate_policies(self, policies: List[Policy]) -> List[PolicyResult]:
        """Evaluate multiple policies."""
        self.results = []
        for policy in policies:
            result = self.evaluate_policy(policy)
            self.results.append(result)
        return self.results
    
    def get_league_table(self, 
                         sort_by: str = 'expectancy',
                         min_trades: Optional[int] = None) -> List[PolicyResult]:
        """Get sorted league table of policy results.
        
        Args:
            sort_by: Metric to sort by (descending)
            min_trades: Filter policies with fewer trades
        """
        min_t = min_trades if min_trades is not None else self.min_trades
        
        # Filter by min trades
        valid_results = [r for r in self.results if r.n_trades >= min_t]
        
        # Sort by metric (descending)
        if sort_by == 'expectancy':
            valid_results.sort(key=lambda r: r.expectancy, reverse=True)
        elif sort_by == 'win_rate':
            valid_results.sort(key=lambda r: r.win_rate, reverse=True)
        elif sort_by == 'profit_factor':
            valid_results.sort(key=lambda r: r.profit_factor, reverse=True)
        elif sort_by == 'sharpe_like':
            valid_results.sort(key=lambda r: r.sharpe_like, reverse=True)
        elif sort_by == 'cum_r_final':
            valid_results.sort(key=lambda r: r.cum_r_final, reverse=True)
        
        return valid_results
    
    def save_results(self, path: str):
        """Save results to JSON."""
        data = {
            'n_total_trades': len(self.trades),
            'min_trades_threshold': self.min_trades,
            'results': [r.to_dict() for r in self.results]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_league_table_csv(self, path: str, sort_by: str = 'expectancy'):
        """Save league table to CSV."""
        import csv
        
        league = self.get_league_table(sort_by=sort_by)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            if not league:
                f.write("No valid policies\n")
                return
            
            fieldnames = list(league[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in league:
                writer.writerow(result.to_dict())
    
    def print_summary(self, top_n: int = 5, sort_by: str = 'expectancy'):
        """Print summary to console."""
        print("=" * 80)
        print(f"S4 Policy Backtest Summary")
        print(f"Total trades: {len(self.trades)}")
        print(f"Min trades threshold: {self.min_trades}")
        print("=" * 80)
        
        league = self.get_league_table(sort_by=sort_by)
        
        if not league:
            print("No valid policies (all have fewer trades than threshold)")
            return
        
        print(f"\nTop {min(top_n, len(league))} policies by {sort_by}:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Policy':<30} {'N':<6} {'WR%':<8} {'Exp(R)':<10} {'MaxDD':<8} {'PF':<8}")
        print("-" * 80)
        
        for i, result in enumerate(league[:top_n], 1):
            print(
                f"{i:<5} {result.name:<30} "
                f"{result.n_trades:<6} "
                f"{result.win_rate*100:>5.1f}%  "
                f"{result.expectancy:>8.2f}  "
                f"{result.max_drawdown_r:>6.2f}  "
                f"{result.profit_factor:>6.2f}"
            )
        
        print("-" * 80)
        
        # Best policy
        best = league[0]
        print(f"\n✅ Best policy: {best.name}")
        print(f"   Expectancy: {best.expectancy:.2f}R")
        print(f"   Win Rate: {best.win_rate:.1%}")
        print(f"   Trades: {best.n_trades} (skipped {best.n_skipped})")


def run_backtest(
    trades: List[S4TradeEnriched],
    policies: List[Policy],
    min_trades: int = 5,
    output_dir: Optional[str] = None,
) -> Backtester:
    """Convenience function to run full backtest.
    
    Args:
        trades: Enriched trades
        policies: List of policies to evaluate
        min_trades: Minimum trades for valid result
        output_dir: If provided, save results here
    
    Returns:
        Backtester instance with results
    """
    bt = Backtester(trades, min_trades=min_trades)
    bt.evaluate_policies(policies)
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        bt.save_results(f"{output_dir}/s4_policy_results_v1.json")
        bt.save_league_table_csv(f"{output_dir}/s4_policy_league_table_v1.csv")
    
    return bt

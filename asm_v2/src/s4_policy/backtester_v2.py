"""
S4 Policy Backtester v2 - Full backtest with train/val/test splits

Enhanced backtester with:
- Time-based train/val/test splits
- ML meta-policy training
- Comprehensive league tables
- Per-split evaluation
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .policy_dataset import S4TradeEnriched, S4PolicyDataset
from .policy_rules import Policy, PolicyConfig, create_policy, create_policies_from_config
from .policy_model import S4MetaModel, train_meta_model
from .metrics import compute_metrics


@dataclass
class SplitResult:
    """Result for a single split (train/val/test)."""
    split_name: str
    n_trades: int
    win_count: int
    loss_count: int
    be_count: int
    win_rate: float
    expectancy: float
    std_rr: float
    max_drawdown_r: float
    profit_factor: float
    sharpe_like: float
    cum_r_final: float
    n_skipped: int
    skip_rate: float


@dataclass
class PolicyResultV2:
    """Full policy result with train/val/test metrics."""
    name: str
    train: Optional[SplitResult]
    val: Optional[SplitResult]
    test: Optional[SplitResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'train': asdict(self.train) if self.train else None,
            'val': asdict(self.val) if self.val else None,
            'test': asdict(self.test) if self.test else None,
        }
    
    def get_test_expectancy(self) -> float:
        return self.test.expectancy if self.test else 0.0
    
    def get_test_n_trades(self) -> int:
        return self.test.n_trades if self.test else 0


class BacktesterV2:
    """Enhanced backtester with train/val/test evaluation."""
    
    def __init__(
        self,
        trades: List[S4TradeEnriched],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        min_trades: int = 5,
    ):
        """
        Args:
            trades: All enriched trades (sorted by time)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            min_trades: Minimum trades for valid result
        """
        self.all_trades = sorted(trades, key=lambda t: t.time)
        self.min_trades = min_trades
        
        # Time-based split
        self.train_trades, self.val_trades, self.test_trades = self._time_split(
            train_ratio, val_ratio, test_ratio
        )
        
        self.results: List[PolicyResultV2] = []
        self.meta_model: Optional[S4MetaModel] = None
    
    def _time_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[List[S4TradeEnriched], List[S4TradeEnriched], List[S4TradeEnriched]]:
        """Split trades by date (time-based, no shuffle)."""
        dates = sorted(set(t.get_date() for t in self.all_trades))
        n_dates = len(dates)
        
        n_train = int(n_dates * train_ratio)
        n_val = int(n_dates * val_ratio)
        
        train_dates = set(dates[:n_train])
        val_dates = set(dates[n_train:n_train + n_val])
        test_dates = set(dates[n_train + n_val:])
        
        train = [t for t in self.all_trades if t.get_date() in train_dates]
        val = [t for t in self.all_trades if t.get_date() in val_dates]
        test = [t for t in self.all_trades if t.get_date() in test_dates]
        
        return train, val, test
    
    def _evaluate_on_split(
        self,
        trades: List[S4TradeEnriched],
        policy: Policy,
        split_name: str,
    ) -> SplitResult:
        """Evaluate policy on a single split."""
        kept = []
        skipped = 0
        
        for trade in trades:
            if policy.decide_trade(trade):
                kept.append(trade)
            else:
                skipped += 1
        
        metrics = compute_metrics(kept)
        
        return SplitResult(
            split_name=split_name,
            n_trades=metrics['n_trades'],
            win_count=metrics['win_count'],
            loss_count=metrics['loss_count'],
            be_count=metrics['be_count'],
            win_rate=metrics['win_rate'],
            expectancy=metrics['expectancy'],
            std_rr=metrics['std_rr'],
            max_drawdown_r=metrics['max_drawdown_r'],
            profit_factor=metrics['profit_factor'],
            sharpe_like=metrics['sharpe_like'],
            cum_r_final=metrics['cum_r_final'],
            n_skipped=skipped,
            skip_rate=skipped / len(trades) if trades else 0.0,
        )
    
    def evaluate_policy(self, policy: Policy) -> PolicyResultV2:
        """Evaluate policy on all splits."""
        train_result = self._evaluate_on_split(self.train_trades, policy, 'train')
        val_result = self._evaluate_on_split(self.val_trades, policy, 'val')
        test_result = self._evaluate_on_split(self.test_trades, policy, 'test')
        
        return PolicyResultV2(
            name=policy.name,
            train=train_result,
            val=val_result,
            test=test_result,
        )
    
    def train_meta_model(self) -> S4MetaModel:
        """Train ML meta-model on training data."""
        train_ds = S4PolicyDataset(self.train_trades)
        val_ds = S4PolicyDataset(self.val_trades)
        
        model, results = train_meta_model(train_ds, val_ds)
        self.meta_model = model
        
        print(f"Meta-model trained:")
        print(f"  Train accuracy: {results['train']['accuracy']:.3f}")
        if 'val' in results:
            print(f"  Val accuracy: {results['val']['accuracy']:.3f}")
        
        return model
    
    def evaluate_policies(
        self,
        policies: List[Policy],
        include_ml_policies: bool = True,
        ml_thresholds: List[float] = [0.4, 0.5, 0.6],
    ) -> List[PolicyResultV2]:
        """Evaluate all policies."""
        self.results = []
        
        # Evaluate rule-based policies
        for policy in policies:
            result = self.evaluate_policy(policy)
            self.results.append(result)
        
        # Train and evaluate ML policies
        if include_ml_policies and len(self.train_trades) >= 20:
            self.train_meta_model()
            
            if self.meta_model and self.meta_model.trained:
                from .policy_rules import MLPredictionPolicy
                
                for threshold in ml_thresholds:
                    ml_config = PolicyConfig(
                        name=f"P_ML_thresh_{threshold}",
                        policy_type="ml",
                        params={"threshold": threshold}
                    )
                    ml_policy = MLPredictionPolicy(ml_config)
                    ml_policy.set_model(self.meta_model)
                    
                    result = self.evaluate_policy(ml_policy)
                    self.results.append(result)
        
        return self.results
    
    def get_league_table(
        self,
        split: str = 'test',
        sort_by: str = 'expectancy',
        min_trades: Optional[int] = None,
    ) -> List[PolicyResultV2]:
        """Get sorted league table for a split."""
        min_t = min_trades if min_trades is not None else self.min_trades
        
        # Filter by min trades
        valid = []
        for r in self.results:
            split_result = getattr(r, split, None)
            if split_result and split_result.n_trades >= min_t:
                valid.append(r)
        
        # Sort
        if sort_by == 'expectancy':
            valid.sort(key=lambda r: getattr(r, split).expectancy, reverse=True)
        elif sort_by == 'win_rate':
            valid.sort(key=lambda r: getattr(r, split).win_rate, reverse=True)
        elif sort_by == 'profit_factor':
            valid.sort(key=lambda r: getattr(r, split).profit_factor, reverse=True)
        elif sort_by == 'sharpe_like':
            valid.sort(key=lambda r: getattr(r, split).sharpe_like, reverse=True)
        
        return valid
    
    def get_best_policy(self, split: str = 'test') -> Optional[PolicyResultV2]:
        """Get best policy by expectancy on split."""
        league = self.get_league_table(split=split, sort_by='expectancy')
        return league[0] if league else None
    
    def save_results(self, output_dir: str):
        """Save all results to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Full results JSON
        results_data = {
            'n_total_trades': len(self.all_trades),
            'n_train': len(self.train_trades),
            'n_val': len(self.val_trades),
            'n_test': len(self.test_trades),
            'results': [r.to_dict() for r in self.results],
        }
        with open(f"{output_dir}/s4_policy_league_gc_m1_v1.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # League table CSV
        self._save_league_csv(f"{output_dir}/s4_policy_league_gc_m1_v1.csv")
        
        # Best policy JSON
        best = self.get_best_policy('test')
        if best:
            best_data = {
                'policy_name': best.name,
                'train': asdict(best.train) if best.train else None,
                'val': asdict(best.val) if best.val else None,
                'test': asdict(best.test) if best.test else None,
            }
            with open(f"{output_dir}/s4_policy_best_gc_m1_v1.json", 'w') as f:
                json.dump(best_data, f, indent=2)
    
    def _save_league_csv(self, path: str):
        """Save league table to CSV."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'policy', 
                'train_n', 'train_wr', 'train_exp', 'train_maxdd', 'train_pf',
                'val_n', 'val_wr', 'val_exp', 'val_maxdd', 'val_pf',
                'test_n', 'test_wr', 'test_exp', 'test_maxdd', 'test_pf',
            ])
            
            for r in self.results:
                row = [r.name]
                for split in ['train', 'val', 'test']:
                    s = getattr(r, split, None)
                    if s:
                        row.extend([
                            s.n_trades,
                            f"{s.win_rate:.3f}",
                            f"{s.expectancy:.3f}",
                            f"{s.max_drawdown_r:.3f}",
                            f"{s.profit_factor:.3f}",
                        ])
                    else:
                        row.extend(['', '', '', '', ''])
                writer.writerow(row)
    
    def print_summary(self, top_n: int = 5):
        """Print summary to console."""
        print("=" * 100)
        print("S4_LDN Policy Backtest Summary (V2)")
        print("=" * 100)
        print(f"Total trades: {len(self.all_trades)}")
        print(f"  Train: {len(self.train_trades)} ({len(self.train_trades)/len(self.all_trades)*100:.1f}%)")
        print(f"  Val:   {len(self.val_trades)} ({len(self.val_trades)/len(self.all_trades)*100:.1f}%)")
        print(f"  Test:  {len(self.test_trades)} ({len(self.test_trades)/len(self.all_trades)*100:.1f}%)")
        
        # Print league for each split
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*100}")
            print(f"Top {top_n} policies on {split.upper()}:")
            print("-" * 100)
            print(f"{'Rank':<5} {'Policy':<30} {'N':<6} {'WR%':<8} {'Exp(R)':<10} {'MaxDD':<8} {'PF':<8}")
            print("-" * 100)
            
            league = self.get_league_table(split=split, sort_by='expectancy')
            
            for i, r in enumerate(league[:top_n], 1):
                s = getattr(r, split)
                pf_str = f"{s.profit_factor:.2f}" if s.profit_factor != float('inf') else "inf"
                print(
                    f"{i:<5} {r.name:<30} "
                    f"{s.n_trades:<6} "
                    f"{s.win_rate*100:>5.1f}%  "
                    f"{s.expectancy:>8.3f}  "
                    f"{s.max_drawdown_r:>6.2f}  "
                    f"{pf_str:>6}"
                )
        
        # Best policy recommendation
        best = self.get_best_policy('test')
        if best:
            print("\n" + "=" * 100)
            print(f"✅ BEST POLICY FOR SHADOW-RUN: {best.name}")
            print(f"   Test Expectancy: {best.test.expectancy:.3f}R")
            print(f"   Test Win Rate: {best.test.win_rate:.1%}")
            print(f"   Test Trades: {best.test.n_trades}")
            print(f"   Test MaxDD: {best.test.max_drawdown_r:.2f}R")
            print("=" * 100)

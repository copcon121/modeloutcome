"""
SMC Rule-Based Labeler
Determines signal side (long/short/skip) based on trend + zone alignment
"""

import sys
from pathlib import Path
import logging
from typing import Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class SignalLabel:
    """Signal label result"""
    side: str  # "long", "short", "skip"
    reason: str  # Why this signal was chosen
    confidence: float = 1.0  # Optional confidence score


class SMCLabeler:
    """
    SMC Rule-Based Signal Labeler
    
    Logic:
    - LONG: ext_trend_dir > 0 AND (in_bull_fvg OR in_bull_ob)
    - SHORT: ext_trend_dir < 0 AND (in_bear_fvg OR in_bear_ob)
    - SKIP: All other cases
    """
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'long': 0,
            'short':0,
            'skip': 0
        }
        
    def determine_signal(self, features: Dict[str, any]) -> SignalLabel:
        """
        Determine signal side from event features
        
        Args:
            features: Dict of feature values for the event bar
            
        Returns:
            SignalLabel with side and reason
        """
        self.stats['total'] += 1
        
        # Extract key features
        ext_trend = features.get('ext_trend_dir', 0)
        
        # Zone flags - FIXED: Use actual column names from CSV
        in_bull_fvg = features.get('in_bull_fvg', False)
        in_bear_fvg = features.get('in_bear_fvg', False)
        
        # OB flags: Check both int and ext (use either)
        int_in_bull_ob = features.get('int_in_bull_ob', False)
        int_in_bear_ob = features.get('int_in_bear_ob', False)
        ext_in_bull_ob = features.get('ext_in_bull_ob', False)
        ext_in_bear_ob = features.get('ext_in_bear_ob', False)
        
        in_bull_ob = int_in_bull_ob or ext_in_bull_ob
        in_bear_ob = int_in_bear_ob or ext_in_bear_ob
        
        # Near flags
        near_bull_fvg = features.get('near_bull_fvg', False)
        near_bear_fvg = features.get('near_bear_fvg', False)
        int_near_bull_ob = features.get('int_near_bull_ob', False)
        int_near_bear_ob = features.get('int_near_bear_ob', False)
        ext_near_bull_ob = features.get('ext_near_bull_ob', False)
        ext_near_bear_ob = features.get('ext_near_bear_ob', False)
        
        near_bull_ob = int_near_bull_ob or ext_near_bull_ob
        near_bear_ob = int_near_bear_ob or ext_near_bear_ob
        
        # Check zone alignment
        in_bull_zone = in_bull_fvg or in_bull_ob or near_bull_fvg or near_bull_ob
        in_bear_zone = in_bear_fvg or in_bear_ob or near_bear_fvg or near_bear_ob
        
        # LONG signal: Uptrend + Bullish zone
        if ext_trend > 0 and in_bull_zone:
            reason = "ext_trend_up+"
            
            if in_bull_fvg:
                reason += "in_bull_fvg"
            elif in_bull_ob:
                reason += "in_bull_ob"
            elif near_bull_fvg:
                reason += "near_bull_fvg"
            else:
                reason += "near_bull_ob"
            
            self.stats['long'] += 1
            return SignalLabel(side="long", reason=reason)
        
        # SHORT signal: Downtrend + Bearish zone
        if ext_trend < 0 and in_bear_zone:
            reason = "ext_trend_down+"
            
            if in_bear_fvg:
                reason += "in_bear_fvg"
            elif in_bear_ob:
                reason += "in_bear_ob"
            elif near_bear_fvg:
                reason += "near_bear_fvg"
            else:
                reason += "near_bear_ob"
            
            self.stats['short'] += 1
            return SignalLabel(side="short", reason=reason)
        
        # SKIP: All other cases
        reasons = []
        
        if ext_trend == 0:
            reasons.append("no_clear_trend")
        if ext_trend > 0 and in_bear_zone:
            reasons.append("trend_zone_conflict_up_bear")
        if ext_trend < 0 and in_bull_zone:
            reasons.append("trend_zone_conflict_down_bull")
        if not in_bull_zone and not in_bear_zone:
            reasons.append("no_zone")
        
        reason = "_".join(reasons) if reasons else "other"
        
        self.stats['skip'] += 1
        return SignalLabel(side="skip", reason=reason)
    
    def get_stats(self) -> Dict[str, any]:
        """Get labeling statistics"""
        total = self.stats['total']
        if total == 0:
            return self.stats
        
        return {
            'total': total,
            'long':self.stats['long'],
            'long_pct': self.stats['long'] / total * 100,
            'short': self.stats['short'],
            'short_pct': self.stats['short'] / total * 100,
            'skip': self.stats['skip'],
            'skip_pct': self.stats['skip'] / total * 100,
            'actionable': self.stats['long'] + self.stats['short'],
            'actionable_pct': (self.stats['long'] + self.stats['short']) / total * 100
        }
    
    def print_stats(self):
        """Print labeling statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("SIGNAL LABELING STATISTICS")
        print("="*60)
        print(f"Total events: {stats['total']:,}")
        print(f"\nSignal Distribution:")
        print(f"  LONG:  {stats['long']:6,} ({stats['long_pct']:5.1f}%)")
        print(f"  SHORT: {stats['short']:6,} ({stats['short_pct']:5.1f}%)")
        print(f"  SKIP:  {stats['skip']:6,} ({stats['skip_pct']:5.1f}%)")
        print(f"\nActionable: {stats['actionable']:,} ({stats['actionable_pct']:.1f}%)")
        print("="*60 + "\n")


def main():
    """Test labeler with sample features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SMC labeler')
    parser.add_argument('--test', action='store_true', help='Run test cases')
    args = parser.parse_args()
    
    labeler = SMCLabeler()
    
    if args.test:
        print("="*60)
        print("TESTING SMC LABELER")
        print("="*60 + "\n")
        
        test_cases = [
            {
                'name': "Uptrend + Bull FVG",
                'features': {
                    'ext_trend_dir': 1,
                    'in_bull_fvg': True,
                    'in_bear_fvg': False,
                    'in_bull_ob': False,
                    'in_bear_ob': False
                },
                'expected': 'long'
            },
            {
                'name': "Downtrend + Bear OB",
                'features': {
                    'ext_trend_dir': -1,
                    'in_bull_fvg': False,
                    'in_bear_fvg': False,
                    'in_bull_ob': False,
                    'in_bear_ob': True
                },
                'expected': 'short'
            },
            {
                'name': "No trend",
                'features': {
                    'ext_trend_dir': 0,
                    'in_bull_fvg': True,
                    'in_bear_fvg': False,
                    'in_bull_ob': False,
                    'in_bear_ob': False
                },
                'expected': 'skip'
            },
            {
                'name': "Trend/Zone conflict",
                'features': {
                    'ext_trend_dir': 1,  # Uptrend
                    'in_bull_fvg': False,
                    'in_bear_fvg': True,  # But in bear zone
                    'in_bull_ob': False,
                    'in_bear_ob': False
                },
                'expected': 'skip'
            },
        ]
        
        for i, test in enumerate(test_cases, 1):
            signal = labeler.determine_signal(test['features'])
            status = "[OK]" if signal.side == test['expected'] else "[FAIL]"
            print(f"{i}. {test['name']}")
            print(f"   Expected: {test['expected']}, Got: {signal.side} ({signal.reason}) {status}")
            print()
        
        labeler.print_stats()
    else:
        print("Use --test to run test cases")


if __name__ == '__main__':
    main()

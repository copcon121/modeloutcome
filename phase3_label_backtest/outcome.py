"""
Outcome Calculator
Computes entry, SL, TP prices and simulates outcomes for labeled signals
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config
from phase3_label_backtest.labeler import SignalLabel
from phase3_label_backtest.risk_config import RiskConfig, load_risk_config_from_env

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Trading outcome result"""
    entry_price: float
    sl_price: float
    tp_price: float
    risk_ticks: float
    rr_ratio: float
    hit: str  # "tp", "sl", "none"
    outcome_rr: float
    hold_bars: int
    exit_price: float
    max_runup_rr: float
    max_drawdown_rr: float


class OutcomeCalculator:
    """
    Calculate trade entry/SL/TP and simulate outcomes
    Now supports configurable risk parameters via RiskConfig
    """
    
    def __init__(self, risk_config: Optional[RiskConfig] = None, tick_size: float = 0.1):
        """
        Initialize outcome calculator
        
        Args:
            risk_config: Risk configuration (SL/TP parameters)
            tick_size: Instrument tick size
        """
        self.risk_config = risk_config or load_risk_config_from_env()
        self.tick_size = tick_size
        self.buffer_ticks = self.risk_config.buffer_ticks
        self.max_hold = 100  # Max bars to hold (configurable later)
        self.target_rr = self.risk_config.rr_target
        
        logger.info(f"OutcomeCalculator initialized: {self.risk_config}")
    
    def calculate_trade(self,
                       signal: SignalLabel,
                       features: Dict,
                       future_bars: pd.DataFrame) -> Optional[TradeOutcome]:
        """
        Calculate complete trade outcome
        
        Args:
            signal: Signal label from labeler
            features: Current bar features
            future_bars: Future bars for outcome simulation
            
        Returns:
            TradeOutcome or None if signal is "skip"
        """
        if signal.side == "skip":
            return None
        
        # Entry price
        entry = features.get('close', 0)
        
        # Calculate SL/TP based on direction
        if signal.side == "long":
            sl = self._calculate_long_sl(features)
            risk = entry - sl
            tp = entry + (risk * self.target_rr)
        else:  # short
            sl = self._calculate_short_sl(features)
            risk = sl - entry
            tp = entry - (risk * self.target_rr)
        
        # Simulate outcome
        hit, outcome_rr, hold_bars, exit_price, max_runup_rr, max_drawdown_rr = self._simulate_outcome(
            signal.side, entry, sl, tp, risk, future_bars
        )
        
        return TradeOutcome(
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            risk_ticks=risk / self.tick_size,
            rr_ratio=self.target_rr,
            hit=hit,
            outcome_rr=outcome_rr,
            hold_bars=hold_bars,
            exit_price=exit_price,
            max_runup_rr=max_runup_rr,
            max_drawdown_rr=max_drawdown_rr,
        )
    
    def _calculate_long_sl(self, features: Dict) -> float:
        """Calculate SL for long trade using configurable ATR multiplier"""
        close = features.get('close', 0)
        
        # For LONG: SL should be BELOW entry
        # Use bar range as ATR proxy (could be enhanced with real ATR later)
        atr_estimate = abs(features.get('high_low_range', 5.0))
        
        # SL = close - (atr_mult_sl × ATR) - buffer
        sl = close - (atr_estimate * self.risk_config.atr_mult_sl) - (self.buffer_ticks * self.tick_size)
        
        return sl
    
    def _calculate_short_sl(self, features: Dict) -> float:
        """Calculate SL for short trade using configurable ATR multiplier"""
        close = features.get('close', 0)
        
        # For SHORT: SL should be ABOVE entry
        # Use bar range as ATR proxy
        atr_estimate = abs(features.get('high_low_range', 5.0))
        
        # SL = close + (atr_mult_sl × ATR) + buffer
        sl = close + (atr_estimate * self.risk_config.atr_mult_sl) + (self.buffer_ticks * self.tick_size)
        
        return sl
    
    def _simulate_outcome(self, direction: str, entry: float, sl: float,
                         tp: float, risk: float, future_bars: pd.DataFrame):
        """
        Simulate forward to determine outcome
        
        FIXED: CSV doesn't have 'high'/'low' columns, use 'close' as proxy
        
        Returns:
            (hit_type, outcome_rr, hold_bars, exit_price)
        """
        if len(future_bars) < config.MIN_FUTURE_BARS:
            # Not enough future data
            return "none", 0.0, 0, entry
        
        max_bars = min(self.max_hold, len(future_bars))
        
        max_runup_rr = 0.0
        max_drawdown_rr = 0.0

        for i in range(max_bars):
            bar = future_bars.iloc[i]
            close_price = bar.get('close', entry)  # Use close as proxy

            # compute run-up / drawdown in R
            if direction == "long":
                pnl = close_price - entry
            else:
                pnl = entry - close_price
            if abs(risk) > 1e-6:
                rr_move = pnl / risk
                max_runup_rr = max(max_runup_rr, rr_move)
                max_drawdown_rr = min(max_drawdown_rr, rr_move)
            
            if direction == "long":
                # Check TP first (assume better fill)
                if close_price >= tp:
                    return ("tp", self.target_rr, i + 1, tp, max_runup_rr, max_drawdown_rr)
                
                # Check SL
                if close_price <= sl:
                    return ("sl", -1.0, i + 1, sl, max_runup_rr, max_drawdown_rr)
            
            else:  # short
                # Check TP first
                if close_price <= tp:
                    return ("tp", self.target_rr, i + 1, tp, max_runup_rr, max_drawdown_rr)
                
                # Check SL
                if close_price >= sl:
                    return ("sl", -1.0, i + 1, sl, max_runup_rr, max_drawdown_rr)
        
        # Max hold reached - force close at market
        final_bar = future_bars.iloc[max_bars - 1]
        exit_price = final_bar['close']
        
        # Calculate actual RR
        actual_pnl = exit_price - entry if direction == "long" else entry - exit_price
        outcome_rr = actual_pnl / risk if abs(risk) > 1e-6 else 0.0
        
        return ("none", outcome_rr, max_bars, exit_price, max_runup_rr, max_drawdown_rr)


def main():
    """Test outcome calculator"""
    print("="*60)
    print("TESTING OUTCOME CALCULATOR")
    print("="*60 + "\n")
    
    calc = OutcomeCalculator()
    
    # Test long trade
    from phase3_label_backtest.labeler import SignalLabel
    
    long_signal = SignalLabel(side="long", reason="test")
    long_features = {
        'close': 3500.0,
        'ext_swing_low_distance': 5.0,
        'int_swing_low_distance': 3.0
    }
    
    # Create fake future bars that hit TP
    future_data = {
        'high': [3500.5, 3505.0, 3510.0, 3515.0],
        'low': [3499.0, 3503.0, 3508.0, 3513.0],
        'close': [3500.2, 3504.5, 3509.5, 3514.5]
    }
    future_bars = pd.DataFrame(future_data)
    
    outcome = calc.calculate_trade(long_signal, long_features, future_bars)
    
    print(f"LONG Trade Test:")
    print(f"  Entry: {outcome.entry_price:.1f}")
    print(f"  SL: {outcome.sl_price:.1f}")
    print(f"  TP: {outcome.tp_price:.1f}")
    print(f"  Risk: {outcome.risk_ticks:.1f} ticks")
    print(f"  RR Ratio: {outcome.rr_ratio:.1f}")
    print(f"  Hit: {outcome.hit}")
    print(f"  Outcome RR: {outcome.outcome_rr:+.2f}R")
    print(f"  Hold: {outcome.hold_bars} bars")
    print(f"  Exit: {outcome.exit_price:.1f}")
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == '__main__':
    main()

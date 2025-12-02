
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque

@dataclass
class Point:
    price: float
    bar_index: int
    timestamp: int = 0

@dataclass
class SwingPoint:
    price: float
    bar_index: int
    timestamp: int
    type: int # 1 for High, -1 for Low
    crossed: bool = False

@dataclass
class OrderBlock:
    top: float
    bottom: float
    bar_index: int # Source bar index (the extreme candle)
    timestamp: int
    type: int # 1 for Bullish, -1 for Bearish
    mitigated: bool = False
    invalidated: bool = False
    active: bool = True

@dataclass
class FVG:
    top: float
    bottom: float
    bar_index: int
    timestamp: int
    type: int # 1 for Bullish, -1 for Bearish
    mitigated: bool = False
    invalidated: bool = False
    active: bool = True

@dataclass
class StructureState:
    # Swings
    swing_high: Optional[SwingPoint] = None
    swing_low: Optional[SwingPoint] = None
    internal_high: Optional[SwingPoint] = None
    internal_low: Optional[SwingPoint] = None
    
    # Previous swings (for HH/HL/LL/LH detection)
    prev_swing_high: Optional[SwingPoint] = None
    prev_swing_low: Optional[SwingPoint] = None
    prev_internal_high: Optional[SwingPoint] = None
    prev_internal_low: Optional[SwingPoint] = None
    
    # Swing Labels (HH/HL/LL/LH) - Pulses for current bar
    # Pine: p_ivot.currentLevel > p_ivot.lastLevel ? 'HH' : 'LH'
    # Pine: p_ivot.currentLevel < p_ivot.lastLevel ? 'LL' : 'HL'
    swing_hh: bool = False  # Higher High
    swing_lh: bool = False  # Lower High
    swing_hl: bool = False  # Higher Low
    swing_ll: bool = False  # Lower Low
    
    internal_hh: bool = False
    internal_lh: bool = False
    internal_hl: bool = False
    internal_ll: bool = False
    
    # Trends
    swing_trend: int = 0 # 1 Bull, -1 Bear
    internal_trend: int = 0
    
    # Trailing Extremes (for PD Zones)
    trailing_top: float = -float('inf')
    trailing_bottom: float = float('inf')
    
    # Events (Pulses for current bar)
    bos_bull: bool = False
    bos_bear: bool = False
    choch_bull: bool = False
    choch_bear: bool = False
    
    internal_bos_bull: bool = False
    internal_bos_bear: bool = False
    internal_choch_bull: bool = False
    internal_choch_bear: bool = False
    
    # Sweeps
    swept_prev_ext_high: bool = False
    swept_prev_ext_low: bool = False
    swept_prev_int_high: bool = False
    swept_prev_int_low: bool = False
    
    # OB Mitigation (Pulses for current bar)
    # Fires when price breaks through an OB
    ob_bull_int_mitigated: bool = False  # Bullish internal OB broken (low < ob.bottom)
    ob_bear_int_mitigated: bool = False  # Bearish internal OB broken (high > ob.top)
    ob_bull_ext_mitigated: bool = False  # Bullish swing OB broken
    ob_bear_ext_mitigated: bool = False  # Bearish swing OB broken

class LuxSMC:
    def __init__(self, 
                 swing_length: int = 50, 
                 internal_length: int = 5,
                 fvg_threshold: float = 0.0,
                 max_zone_age: int = 100): # Added max age
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.fvg_threshold = fvg_threshold
        self.max_zone_age = max_zone_age
        
        # Data buffers (needed for lookback)
        self.highs = deque()
        self.lows = deque()
        self.closes = deque()
        self.opens = deque()
        self.timestamps = deque()
        self.indices = deque()
        
        # Max buffer size (enough for swing detection + OB lookback)
        self.max_buffer = max(swing_length, internal_length) * 5 
        
        # State
        self.state = StructureState()
        
        # Active Zones
        self.swing_obs: List[OrderBlock] = []
        self.internal_obs: List[OrderBlock] = []
        self.fvgs: List[FVG] = []
        
        # Internal tracking
        self.last_swing_leg = 0 # 1 Bull, -1 Bear
        self.last_internal_leg = 0
        
        # BOS History (for counting recent BOS events)
        self.bos_history: List[Dict] = []  # [{bar_index, direction: 1/-1}]
        self.bos_history_maxlen = 50  # Keep last 50 BOS events
        
        # Rolling range tracking
        self.rolling_high = 0.0
        self.rolling_low = float('inf')
        self.rolling_range_bars = 20  # Look back 20 bars for range
        
    def update(self, open_: float, high: float, low: float, close: float, timestamp: int, bar_index: int) -> StructureState:
        # 1. Update Buffers
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.opens.append(open_)
        self.timestamps.append(timestamp)
        self.indices.append(bar_index)
        
        if len(self.highs) > self.max_buffer:
            self.highs.popleft()
            self.lows.popleft()
            self.closes.popleft()
            self.opens.popleft()
            self.timestamps.popleft()
            self.indices.popleft()
            
        # Reset pulses
        self.state.bos_bull = False
        self.state.bos_bear = False
        self.state.choch_bull = False
        self.state.choch_bear = False
        self.state.internal_bos_bull = False
        self.state.internal_bos_bear = False
        self.state.internal_choch_bull = False
        self.state.internal_choch_bear = False
        
        # Reset HH/HL/LL/LH pulses
        self.state.swing_hh = False
        self.state.swing_lh = False
        self.state.swing_hl = False
        self.state.swing_ll = False
        self.state.internal_hh = False
        self.state.internal_lh = False
        self.state.internal_hl = False
        self.state.internal_ll = False
        
        # Reset sweeps
        self.state.swept_prev_ext_high = False
        self.state.swept_prev_ext_low = False
        self.state.swept_prev_int_high = False
        self.state.swept_prev_int_low = False
        
        # Reset OB mitigation pulses
        self.state.ob_bull_int_mitigated = False
        self.state.ob_bear_int_mitigated = False
        self.state.ob_bull_ext_mitigated = False
        self.state.ob_bear_ext_mitigated = False
        
        # Need enough data
        if len(self.highs) <= self.swing_length + 1:
            return self.state

        # 2. Update Trailing Extremes (for PD Zones)
        self._update_trailing_extremes(high, low)
        
        # 3. Detect Swings & Structure
        self._process_structure(self.swing_length, is_internal=False, bar_index=bar_index)
        self._process_structure(self.internal_length, is_internal=True, bar_index=bar_index)
        
        # 4. Detect FVGs
        self._detect_fvgs(bar_index, timestamp)
        
        # 5. Maintain Zones (Mitigation & Age)
        self._maintain_zones(high, low, bar_index)
        
        # 6. Detect Sweeps (Liquidity)
        # Check if current High/Low exceeds the most recent confirmed swing points
        # Only trigger on the crossing bar (or re-crossing)
        
        prev_high = self.highs[-2] if len(self.highs) > 1 else -float('inf')
        prev_low = self.lows[-2] if len(self.lows) > 1 else float('inf')
        
        if self.state.swing_high and high > self.state.swing_high.price and prev_high <= self.state.swing_high.price:
            self.state.swept_prev_ext_high = True
        if self.state.swing_low and low < self.state.swing_low.price and prev_low >= self.state.swing_low.price:
            self.state.swept_prev_ext_low = True
            
        if self.state.internal_high and high > self.state.internal_high.price and prev_high <= self.state.internal_high.price:
            self.state.swept_prev_int_high = True
        if self.state.internal_low and low < self.state.internal_low.price and prev_low >= self.state.internal_low.price:
            self.state.swept_prev_int_low = True
        
        return self.state

    def _update_trailing_extremes(self, high, low):
        self.state.trailing_top = max(self.state.trailing_top, high)
        self.state.trailing_bottom = min(self.state.trailing_bottom, low)

    # ... (rest of methods) ...

    def _maintain_zones(self, high: float, low: float, current_index: int):
        # Filter active zones
        # We will iterate and keep only active ones
        
        # 1. Swing OBs
        active_swing_obs = []
        for ob in self.swing_obs:
            if not ob.active: continue
            
            # Age Check
            if current_index - ob.bar_index > self.max_zone_age:
                ob.active = False
                continue
                
            # Mitigation/Invalidation Check
            if ob.type == 1: # Bullish OB
                if low < ob.bottom: # Broken
                    ob.mitigated = True
                    ob.active = False # Remove from active
                    self.state.ob_bull_ext_mitigated = True  # Fire signal
                elif low <= ob.top: # Touched/Testing
                    # We can keep it active but mark as testing?
                    # For now, just keep active until broken.
                    pass
            else: # Bearish OB
                if high > ob.top: # Broken
                    ob.mitigated = True
                    ob.active = False
                    self.state.ob_bear_ext_mitigated = True  # Fire signal
                elif high >= ob.bottom:
                    pass
            
            if ob.active:
                active_swing_obs.append(ob)
        self.swing_obs = active_swing_obs
        
        # 2. Internal OBs
        active_int_obs = []
        for ob in self.internal_obs:
            if not ob.active: continue
            
            if current_index - ob.bar_index > self.max_zone_age:
                ob.active = False
                continue
                
            if ob.type == 1:  # Bullish OB
                if low < ob.bottom:
                    ob.mitigated = True
                    ob.active = False
                    self.state.ob_bull_int_mitigated = True  # Fire signal
            else:  # Bearish OB
                if high > ob.top:
                    ob.mitigated = True
                    ob.active = False
                    self.state.ob_bear_int_mitigated = True  # Fire signal
                    
            if ob.active:
                active_int_obs.append(ob)
        self.internal_obs = active_int_obs
                
        # 3. FVGs
        active_fvgs = []
        for fvg in self.fvgs:
            if not fvg.active: continue
            
            if current_index - fvg.bar_index > self.max_zone_age:
                fvg.active = False
                continue
                
            if fvg.type == 1: # Bullish FVG
                if low < fvg.bottom: # Completely Filled/Broken
                    fvg.mitigated = True
                    fvg.active = False
                # Note: Partial fill logic could be complex (shrinking the FVG).
                # For now, we keep it active until fully filled/broken.
            else: # Bearish FVG
                if high > fvg.top: # Completely Filled/Broken
                    fvg.mitigated = True
                    fvg.active = False
            
            if fvg.active:
                active_fvgs.append(fvg)
        self.fvgs = active_fvgs

    def _process_structure(self, length: int, is_internal: bool, bar_index: int = 0):
        # Pine Logic:
        # newLegHigh = high[length] > ta.highest(length)
        # newLegLow = low[length] < ta.lowest(length)
        # Note: ta.highest(length) in Pine is max of current 'length' bars (excluding the one at 'length' ago?)
        # Pine: high[length] is the bar 'length' bars ago.
        # ta.highest(length) is max(high[0], high[1], ... high[length-1])
        # So we compare bar at -(length+1) with max of bars [-(length) ... -1]
        
        if len(self.highs) < length + 1:
            return

        # Indices in deque: -1 is current, -(length+1) is the candidate
        candidate_idx = -(length + 1)
        
        candidate_high = self.highs[candidate_idx]
        candidate_low = self.lows[candidate_idx]
        
        # Range for confirmation: last 'length' bars
        # In python slice: [-length:]
        recent_highs = list(self.highs)[-length:]
        recent_lows = list(self.lows)[-length:]
        
        highest_recent = max(recent_highs)
        lowest_recent = min(recent_lows)
        
        new_leg_high = candidate_high > highest_recent
        new_leg_low = candidate_low < lowest_recent
        
        # Determine Leg
        prev_leg = self.last_internal_leg if is_internal else self.last_swing_leg
        leg = prev_leg
            
        if new_leg_high:
            leg = -1 # Bearish Leg (we just confirmed a High)
        elif new_leg_low:
            leg = 1 # Bullish Leg (we just confirmed a Low)
        
        # No new leg detected - but still need to check BOS/CHoCH
        if leg == 0:
            # Still check BOS/CHoCH even without new leg
            self._check_bos_choch(is_internal, bar_index)
            return
            
        # Check if leg changed (or first leg)
        leg_changed = (prev_leg != leg)
        
        # Update leg state
        if is_internal:
            self.last_internal_leg = leg
        else:
            self.last_swing_leg = leg
        
        # If no leg change, still check BOS/CHoCH
        if not leg_changed:
            self._check_bos_choch(is_internal, bar_index)
            return
            
        # Leg changed! New Pivot Confirmed
        # Pine: When leg changes to BEARISH (-1), we confirm a HIGH
        # When leg changes to BULLISH (1), we confirm a LOW
        if leg_changed:
            # New Pivot Confirmed!
            pivot_bar_idx = self.indices[candidate_idx]
            pivot_ts = self.timestamps[candidate_idx]
            
            if leg == 1: # New Bullish Leg -> We just confirmed a Swing Low
                new_pivot = SwingPoint(candidate_low, pivot_bar_idx, pivot_ts, -1)
                
                if is_internal:
                    # Pine: p_ivot.currentLevel < p_ivot.lastLevel ? 'LL' : 'HL'
                    # lastLevel = current swing_low (before update), not prev_swing_low
                    last_low = self.state.internal_low
                    if last_low:
                        if candidate_low < last_low.price:
                            self.state.internal_ll = True  # Lower Low
                        else:
                            self.state.internal_hl = True  # Higher Low
                    self.state.prev_internal_low = self.state.internal_low
                    self.state.internal_low = new_pivot
                else:
                    # lastLevel = current swing_low (before update)
                    last_low = self.state.swing_low
                    if last_low:
                        if candidate_low < last_low.price:
                            self.state.swing_ll = True  # Lower Low
                        else:
                            self.state.swing_hl = True  # Higher Low
                    self.state.prev_swing_low = self.state.swing_low
                    self.state.swing_low = new_pivot
                    # Reset trailing bottom for PD
                    self.state.trailing_bottom = candidate_low
                    
            else: # New Bearish Leg -> We just confirmed a Swing High
                new_pivot = SwingPoint(candidate_high, pivot_bar_idx, pivot_ts, 1)
                
                if is_internal:
                    # Pine: p_ivot.currentLevel > p_ivot.lastLevel ? 'HH' : 'LH'
                    # lastLevel = current swing_high (before update), not prev_swing_high
                    last_high = self.state.internal_high
                    if last_high:
                        if candidate_high > last_high.price:
                            self.state.internal_hh = True  # Higher High
                        else:
                            self.state.internal_lh = True  # Lower High
                    self.state.prev_internal_high = self.state.internal_high
                    self.state.internal_high = new_pivot
                else:
                    # lastLevel = current swing_high (before update)
                    last_high = self.state.swing_high
                    if last_high:
                        if candidate_high > last_high.price:
                            self.state.swing_hh = True  # Higher High
                        else:
                            self.state.swing_lh = True  # Lower High
                    self.state.prev_swing_high = self.state.swing_high
                    self.state.swing_high = new_pivot
                    # Reset trailing top for PD
                    self.state.trailing_top = candidate_high
        
        # Update Leg State
        if is_internal:
            self.last_internal_leg = leg
        else:
            self.last_swing_leg = leg
            
        # Check BOS/CHoCH after updating pivots
        self._check_bos_choch(is_internal, bar_index)
    
    def _check_bos_choch(self, is_internal: bool, bar_index: int):
        """Check BOS/CHoCH - called every bar regardless of leg change"""
        if len(self.closes) < 2:
            return
            
        # PINE SCRIPT: Uses ta.crossover(close, level) which requires:
        # - current_close > level AND prev_close <= level
        # This ensures signal fires only on the ACTUAL crossover bar
        current_close = self.closes[-1]
        prev_close = self.closes[-2]
        
        # Bullish Break (Crossover High)
        # Pine: if ta.crossover(close, p_ivot.currentLevel) and not p_ivot.crossed
        #       tag = t_rend.bias == BEARISH ? CHOCH : BOS
        # BEARISH = -1, so: CHOCH if bias == -1, else BOS
        active_high = self.state.internal_high if is_internal else self.state.swing_high
        if active_high and not active_high.crossed:
            crossover_high = (current_close > active_high.price) and (prev_close <= active_high.price)
            if crossover_high:
                active_high.crossed = True
                trend_bias = self.state.internal_trend if is_internal else self.state.swing_trend
                
                # Pine: CHOCH if bias == BEARISH (-1), else BOS
                is_choch = (trend_bias == -1)
                
                if is_internal:
                    self.state.internal_trend = 1
                    if is_choch:
                        self.state.internal_choch_bull = True
                    else:
                        self.state.internal_bos_bull = True
                else:
                    self.state.swing_trend = 1
                    if is_choch:
                        self.state.choch_bull = True
                    else:
                        self.state.bos_bull = True
                    # Track BOS history (external only)
                    self._add_bos_history(bar_index, 1)
                    
                # Create Order Block (Bullish)
                self._create_order_block(active_high, 1, is_internal)

        # Bearish Break (Crossunder Low)
        # Pine: tag = t_rend.bias == BULLISH ? CHOCH : BOS
        # BULLISH = 1, so: CHOCH if bias == 1, else BOS
        active_low = self.state.internal_low if is_internal else self.state.swing_low
        if active_low and not active_low.crossed:
            crossunder_low = (current_close < active_low.price) and (prev_close >= active_low.price)
            if crossunder_low:
                active_low.crossed = True
                trend_bias = self.state.internal_trend if is_internal else self.state.swing_trend
                
                # Pine: CHOCH if bias == BULLISH (1), else BOS
                is_choch = (trend_bias == 1)
                
                if is_internal:
                    self.state.internal_trend = -1
                    if is_choch:
                        self.state.internal_choch_bear = True
                    else:
                        self.state.internal_bos_bear = True
                else:
                    self.state.swing_trend = -1
                    if is_choch:
                        self.state.choch_bear = True
                    else:
                        self.state.bos_bear = True
                    # Track BOS history (external only)
                    self._add_bos_history(bar_index, -1)
                    
                # Create Order Block (Bearish)
                self._create_order_block(active_low, -1, is_internal)

    def _create_order_block(self, broken_pivot: SwingPoint, type: int, is_internal: bool):
        # Type 1 = Bullish OB (created by Bullish Break) -> Look for Lowest Low (Origin)
        # Type -1 = Bearish OB (created by Bearish Break) -> Look for Highest High (Origin)
        
        start_idx = broken_pivot.bar_index
        end_idx = self.indices[-1]
        
        # We need to find data in buffer corresponding to these indices
        # Since buffers are deques, we convert to list or iterate
        # Optimization: We know the indices are sequential.
        
        # Find buffer index for start_idx
        try:
            buffer_start_pos = list(self.indices).index(start_idx)
        except ValueError:
            return # Too old
            
        range_highs = list(self.highs)[buffer_start_pos:]
        range_lows = list(self.lows)[buffer_start_pos:]
        range_opens = list(self.opens)[buffer_start_pos:]
        range_closes = list(self.closes)[buffer_start_pos:]
        range_indices = list(self.indices)[buffer_start_pos:]
        range_timestamps = list(self.timestamps)[buffer_start_pos:]
        
        if not range_highs:
            return

        ob_candle_idx = -1
        
        if type == 1: # Bullish OB -> Find Lowest Low
            min_val = min(range_lows)
            local_idx = range_lows.index(min_val)
            ob_candle_idx = local_idx
        else: # Bearish OB -> Find Highest High
            max_val = max(range_highs)
            local_idx = range_highs.index(max_val)
            ob_candle_idx = local_idx
            
        # Create OB from that candle
        # Pine: obHi = OBUseFullCandle ? highs[i] : Math.Max(opens[i], closes[i])
        # We'll use Full Candle for now as per common SMC
        ob_high = range_highs[ob_candle_idx]
        ob_low = range_lows[ob_candle_idx]
        ob_idx = range_indices[ob_candle_idx]
        ob_ts = range_timestamps[ob_candle_idx]
        
        ob = OrderBlock(ob_high, ob_low, ob_idx, ob_ts, type)
        
        if is_internal:
            self.internal_obs.insert(0, ob) # Add to front
            if len(self.internal_obs) > 20: self.internal_obs.pop()
        else:
            self.swing_obs.insert(0, ob)
            if len(self.swing_obs) > 20: self.swing_obs.pop()

    def _detect_fvgs(self, bar_index: int, timestamp: int):
        """
        Detect Fair Value Gaps using Pine Script logic
        
        PINE SCRIPT:
        ```
        bullishFairValueGap = currentLow > last2High and lastClose > last2High and barDeltaPercent > threshold
        bearishFairValueGap = currentHigh < last2Low and lastClose < last2Low and -barDeltaPercent > threshold
        ```
        
        Where:
        - currentLow = low[0] (current bar)
        - last2High = high[2] (2 bars ago)
        - lastClose = close[1] (previous bar)
        """
        if len(self.highs) < 3:
            return
            
        # Bar indices: -1 = current, -2 = previous, -3 = 2 bars ago
        curr_low = self.lows[-1]      # low[0]
        curr_high = self.highs[-1]    # high[0]
        
        prev_close = self.closes[-2]  # close[1] (lastClose)
        prev_open = self.opens[-2]    # open[1]
        
        prev2_high = self.highs[-3]   # high[2] (last2High)
        prev2_low = self.lows[-3]     # low[2] (last2Low)
        
        # Calculate bar delta percent for threshold (optional)
        bar_delta_pct = 0.0
        if prev_open != 0:
            bar_delta_pct = (prev_close - prev_open) / (prev_open * 100.0)
        
        # Bullish FVG: currentLow > last2High AND lastClose > last2High
        if curr_low > prev2_high and prev_close > prev2_high:
            gap = curr_low - prev2_high
            if gap > 0:  # Could add threshold check here
                fvg = FVG(curr_low, prev2_high, bar_index-1, self.timestamps[-2], 1)
                self.fvgs.insert(0, fvg)
            
        # Bearish FVG: currentHigh < last2Low AND lastClose < last2Low
        if curr_high < prev2_low and prev_close < prev2_low:
            gap = prev2_low - curr_high
            if gap > 0:
                fvg = FVG(prev2_low, curr_high, bar_index-1, self.timestamps[-2], -1)
                self.fvgs.insert(0, fvg)
            
        if len(self.fvgs) > 50:
            self.fvgs.pop()




    def _add_bos_history(self, bar_index: int, direction: int):
        """Track BOS event for counting recent BOS"""
        self.bos_history.append({'bar_index': bar_index, 'direction': direction})
        if len(self.bos_history) > self.bos_history_maxlen:
            self.bos_history.pop(0)
    
    def get_recent_bos_counts(self, current_bar_index: int, lookback_bars: int = 20) -> Tuple[int, int]:
        """Count BOS up/down in recent N bars"""
        bos_up = 0
        bos_down = 0
        min_idx = current_bar_index - lookback_bars
        
        for bos in self.bos_history:
            if bos['bar_index'] >= min_idx:
                if bos['direction'] == 1:
                    bos_up += 1
                else:
                    bos_down += 1
        return bos_up, bos_down
    
    def get_ob_imbalance(self, current_price: float, atr: float) -> float:
        """
        Calculate OB imbalance: n_buy_OB - n_sell_OB near current price
        Returns value clipped to [-3, 3]
        """
        threshold = 2.0 * atr if atr > 0 else 10.0
        n_buy = 0
        n_sell = 0
        
        for ob in self.swing_obs:
            if ob.mitigated:
                continue
            mid = (ob.top + ob.bottom) / 2
            if abs(current_price - mid) <= threshold:
                if ob.type == 1:
                    n_buy += 1
                else:
                    n_sell += 1
        
        for ob in self.internal_obs:
            if ob.mitigated:
                continue
            mid = (ob.top + ob.bottom) / 2
            if abs(current_price - mid) <= threshold:
                if ob.type == 1:
                    n_buy += 1
                else:
                    n_sell += 1
        
        imbalance = n_buy - n_sell
        return max(-3, min(3, imbalance))
    
    def get_swing_phase(self) -> int:
        """
        Determine swing phase: 0=range, 1=impulse, 2=pullback
        Based on recent structure breaks
        """
        if self.state.bos_bull or self.state.bos_bear:
            return 1  # Impulse (just broke structure)
        
        if self.state.choch_bull or self.state.choch_bear:
            return 2  # Pullback (trend change)
        
        if self.last_swing_leg != 0:
            return 1 if abs(self.state.swing_trend) == 1 else 0
        
        return 0  # Range
    
    def get_price_position_in_range(self, current_price: float) -> float:
        """
        Calculate price position in rolling range [0, 1]
        0 = at low, 1 = at high
        """
        if len(self.highs) < self.rolling_range_bars:
            return 0.5
        
        recent_highs = list(self.highs)[-self.rolling_range_bars:]
        recent_lows = list(self.lows)[-self.rolling_range_bars:]
        
        range_high = max(recent_highs)
        range_low = min(recent_lows)
        
        if range_high == range_low:
            return 0.5
        
        pos = (current_price - range_low) / (range_high - range_low)
        return max(0.0, min(1.0, pos))

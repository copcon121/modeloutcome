"""
Phase 2 Feature Engine V2 - Context Manager
Orchestrates all SMC detectors and builds FeatureBar output
"""

from typing import Optional, Dict, List
import logging
import math
from collections import deque

from .schema import RawBar, FeatureBar
from .config import SMCConfig
from .smc_core.smc_lux import LuxSMC

logger = logging.getLogger(__name__)

class HTFResampler:
    """
    Resamples M1 bars into Higher Timeframe (HTF) bars.
    Calculates EMA and ATR on the CLOSED HTF bars.
    """
    def __init__(self, period_minutes: int, ema_period: int = 200, atr_period: int = 14):
        self.period_minutes = period_minutes
        self.ema_period = ema_period
        self.atr_period = atr_period
        
        # State for current building bar
        self.current_bar = None
        self.last_closed_bar = None
        
        # History for indicators
        self.closed_closes = deque(maxlen=ema_period + 10)
        self.closed_highs = deque(maxlen=atr_period + 10)
        self.closed_lows = deque(maxlen=atr_period + 10)
        self.closed_trs = deque(maxlen=atr_period + 10)
        
        # Current Indicator Values (updated on Close)
        self.ema_value = 0.0
        self.atr_value = 0.0
        
        # Trend State
        self.trend = 0 # 1 Bull, -1 Bear (based on EMA)

    def update(self, bar: RawBar) -> Optional[Dict]:
        """
        Update with new M1 bar.
        Returns the CLOSED HTF bar dict if a bar just closed, else None.
        """
        timestamp = bar.timestamp
        
        # Determine bucket index
        # Assuming timestamp is datetime object
        total_minutes = timestamp.hour * 60 + timestamp.minute
        bucket_index = total_minutes // self.period_minutes
        
        # Check if we moved to a new bucket
        is_new_bucket = False
        if self.current_bar:
            current_bucket = self.current_bar['bucket_index']
            if bucket_index != current_bucket or timestamp.day != self.current_bar['timestamp'].day:
                is_new_bucket = True
                
        if self.current_bar is None:
            # First bar ever
            self._start_new_bar(bar, bucket_index)
            return None
            
        closed_bar = None
        if is_new_bucket:
            # Close the previous bar
            closed_bar = self.current_bar.copy()
            self.last_closed_bar = closed_bar
            
            # Update Indicators with the CLOSED bar
            self._update_indicators(closed_bar)
            
            # Start new bar
            self._start_new_bar(bar, bucket_index)
        else:
            # Update current bar
            self._update_current_bar(bar)
            
        return closed_bar

    def _start_new_bar(self, bar, bucket_index):
        self.current_bar = {
            'open': bar.o,
            'high': bar.h,
            'low': bar.l,
            'close': bar.c,
            'volume': bar.volume,
            'timestamp': bar.timestamp, # Start time
            'bucket_index': bucket_index
        }

    def _update_current_bar(self, bar):
        b = self.current_bar
        b['high'] = max(b['high'], bar.h)
        b['low'] = min(b['low'], bar.l)
        b['close'] = bar.c
        b['volume'] += bar.volume

    def _update_indicators(self, bar):
        c = bar['close']
        h = bar['high']
        l = bar['low']
        
        # TR
        prev_close = self.closed_closes[-1] if self.closed_closes else c
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        
        self.closed_closes.append(c)
        self.closed_highs.append(h)
        self.closed_lows.append(l)
        self.closed_trs.append(tr)
        
        # EMA
        if len(self.closed_closes) >= self.ema_period:
            # Simple EMA calculation
            if self.ema_value == 0.0:
                self.ema_value = sum(list(self.closed_closes)[-self.ema_period:]) / self.ema_period
            else:
                k = 2 / (self.ema_period + 1)
                self.ema_value = (c * k) + (self.ema_value * (1 - k))
                
        # ATR
        if len(self.closed_trs) >= self.atr_period:
            if self.atr_value == 0.0:
                self.atr_value = sum(list(self.closed_trs)[-self.atr_period:]) / self.atr_period
            else:
                # RMA (Wilder's Smoothing) usually for ATR
                alpha = 1.0 / self.atr_period
                self.atr_value = (tr * alpha) + (self.atr_value * (1 - alpha))
                
        # Trend
        if self.ema_value > 0:
            self.trend = 1 if c > self.ema_value else -1


class SMCContextManager:
    """
    Manages stateful context for feature engineering.
    - LuxSMC (M1)
    - Volume Profile (Daily)
    - Wave Strength (M1)
    - Multi-Timeframe Context (H1, H4)
    """
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        
        # M1 SMC
        swing_len = getattr(config, 'swing_ext_window', 50)
        int_len = getattr(config, 'swing_int_window', 5)
        max_age = getattr(config, 'max_zone_age', 100)
        
        self.smc = LuxSMC(
            swing_length=swing_len,
            internal_length=int_len,
            max_zone_age=max_age
        )
        
        # HTF Config (M5 + H1 for M1 trading)
        self.htf_ema_period = getattr(config, 'htf_ema_period', 50)
        self.htf_m5_swing_length = getattr(config, 'htf_m5_swing_length', 20)
        self.htf_h1_swing_length = getattr(config, 'htf_h1_swing_length', 20)
        
        # Volume Profile
        from .volume_profile import VolumeProfileBuilder, GC_M1_VP_DAILY
        self.vp_builder = VolumeProfileBuilder(GC_M1_VP_DAILY)
        
        # Wave Strength
        from .wave_analyzer import WaveAnalyzer
        self.wave_analyzer = WaveAnalyzer(window=10)
        
        # HTF Resamplers (M5 + H1)
        self.m5_resampler = HTFResampler(period_minutes=5, ema_period=self.htf_ema_period)
        self.h1_resampler = HTFResampler(period_minutes=60, ema_period=self.htf_ema_period)
        
        # HTF SMC (M5 + H1)
        # M5: swing_length=20 (needs 20 M5 bars = ~1.7 hours before first swing)
        # H1: swing_length=20 (needs 20 H1 bars = 20 hours before first swing)
        self.m5_smc = LuxSMC(swing_length=self.htf_m5_swing_length, internal_length=5)
        self.h1_smc = LuxSMC(swing_length=self.htf_h1_swing_length, internal_length=5)
        
        # Counters
        self.bar_count = 0
        self.m5_bar_count = 0
        self.h1_bar_count = 0
        
        logger.info(f"SMCContextManager initialized with LuxSMC (swing={swing_len}, int={int_len})")
        logger.info(f"  HTF: M5 swing={self.htf_m5_swing_length}, H1 swing={self.htf_h1_swing_length}")

    def update(self, bar: RawBar) -> FeatureBar:
        self.bar_count += 1
        
        # 1. Update M1 SMC
        ts_int = int(bar.timestamp.timestamp()) if hasattr(bar.timestamp, 'timestamp') else 0
        
        smc_state = self.smc.update(
            open_=bar.o,
            high=bar.h,
            low=bar.l,
            close=bar.c,
            timestamp=ts_int,
            bar_index=bar.bar_index
        )
        
        # 2. Update Volume Profile
        vp_state = self.vp_builder.update(bar)
        
        # 3. Update Wave Strength
        wave_features = self.wave_analyzer.update(bar, smc_state, len(self.smc.fvgs))
        
        # 4. Update HTF Context (M5 & H1)
        # M5
        closed_m5 = self.m5_resampler.update(bar)
        if closed_m5:
            self.m5_bar_count += 1
            self.m5_smc.update(
                closed_m5['open'], closed_m5['high'], closed_m5['low'], closed_m5['close'],
                int(closed_m5['timestamp'].timestamp()), self.m5_bar_count
            )
            
        # H1
        closed_h1 = self.h1_resampler.update(bar)
        if closed_h1:
            self.h1_bar_count += 1
            self.h1_smc.update(
                closed_h1['open'], closed_h1['high'], closed_h1['low'], closed_h1['close'],
                int(closed_h1['timestamp'].timestamp()), self.h1_bar_count
            )
            
        # 5. Build Feature Bar
        fb = self._build_feature_bar(bar, smc_state, vp_state, wave_features)
        return fb

    def _build_feature_bar(self, bar: RawBar, state, vp, wave) -> FeatureBar:
        
        # Helper for HTF Features
        def get_htf_features(smc: LuxSMC, resampler: HTFResampler, prefix: str):
            # 1. Trend
            is_bull = False
            is_bear = False
            
            # Trend Logic (Structure > EMA)
            trend = 0
            
            # 1. Structure (Primary)
            if smc.last_swing_leg == 1: trend = 1
            elif smc.last_swing_leg == -1: trend = -1
            
            # 2. EMA (Secondary / Confluence)
            # If structure is undefined/neutral, use EMA
            if trend == 0 and resampler.ema_value > 0:
                if bar.c > resampler.ema_value: trend = 1
                elif bar.c < resampler.ema_value: trend = -1
            
            if trend == 1: is_bull = True
            elif trend == -1: is_bear = True
            
            # 2. Premium / Discount (FIXED: use trailing extremes as fallback)
            premium = 0.0
            discount = 0.0
            
            sh = smc.state.swing_high
            sl = smc.state.swing_low
            
            # Get reference high/low (swing points or trailing extremes)
            ref_high = None
            ref_low = None
            
            if sh:
                ref_high = sh.price
            elif smc.state.trailing_top > -float('inf'):
                ref_high = smc.state.trailing_top
                
            if sl:
                ref_low = sl.price
            elif smc.state.trailing_bottom < float('inf'):
                ref_low = smc.state.trailing_bottom
            
            # Calculate premium/discount if we have both references
            if ref_high is not None and ref_low is not None and ref_high > ref_low:
                mid = (ref_high + ref_low) / 2
                if bar.c > mid: 
                    premium = 1.0
                else: 
                    discount = 1.0
                
            # 3. Swing Distances (FIXED: use trailing extremes as fallback)
            dist_sh = 0.0
            dist_sl = 0.0
            atr = resampler.atr_value if resampler.atr_value > 0 else 1.0
            
            if ref_high is not None:
                dist_sh = abs(ref_high - bar.c) / atr
            if ref_low is not None:
                dist_sl = abs(bar.c - ref_low) / atr
                
            # 4. FVG Proximity
            near_fvg = 0.0
            threshold = 0.2 * atr if atr > 0 else 1.0
            
            for fvg in smc.fvgs:
                if not fvg.active: continue
                
                if fvg.bottom <= bar.c <= fvg.top:
                    near_fvg = 1.0
                    break
                
                d_top = abs(bar.c - fvg.top)
                d_bot = abs(bar.c - fvg.bottom)
                if min(d_top, d_bot) < threshold:
                    near_fvg = 1.0
                    break
                    
            return {
                f'{prefix}_trend_up': 1.0 if is_bull else 0.0,
                f'{prefix}_trend_down': 1.0 if is_bear else 0.0,
                f'{prefix}_premium': premium,
                f'{prefix}_discount': discount,
                f'dist_to_{prefix}_swing_high': dist_sh,
                f'dist_to_{prefix}_swing_low': dist_sl,
                f'near_{prefix}_fvg': near_fvg
            }

        m5_feats = get_htf_features(self.m5_smc, self.m5_resampler, 'm5')
        h1_feats = get_htf_features(self.h1_smc, self.h1_resampler, 'h1')
        
        # Enhanced macro features (10 new)
        def get_enhanced_htf_features(smc: LuxSMC, resampler: HTFResampler, bar_count: int, prefix: str):
            atr = resampler.atr_value if resampler.atr_value > 0 else 1.0
            
            # swing_phase: 0=range, 1=impulse, 2=pullback
            swing_phase = smc.get_swing_phase()
            
            # price_pos_in_range: 0-1
            price_pos = smc.get_price_position_in_range(bar.c)
            
            # BOS counts in recent bars (use HTF bar count for lookback)
            lookback = 10  # 10 HTF bars
            bos_up, bos_down = smc.get_recent_bos_counts(bar_count, lookback)
            
            # OB imbalance
            ob_imbalance = smc.get_ob_imbalance(bar.c, atr)
            
            return {
                f'{prefix}_swing_phase': float(swing_phase),
                f'{prefix}_price_pos_in_range': price_pos,
                f'{prefix}_bos_up_count_recent': float(bos_up),
                f'{prefix}_bos_down_count_recent': float(bos_down),
                f'{prefix}_ob_imbalance': ob_imbalance
            }
        
        m5_enhanced = get_enhanced_htf_features(self.m5_smc, self.m5_resampler, self.m5_bar_count, 'm5')
        h1_enhanced = get_enhanced_htf_features(self.h1_smc, self.h1_resampler, self.h1_bar_count, 'h1')
        
        # M1 Zone Helpers
        def is_in_zone(price, zones):
            for z in zones:
                if z.mitigated: continue
                if z.bottom <= price <= z.top: return True
            return False
            
        def is_near_zone(price, zones, threshold):
            for z in zones:
                if z.mitigated: continue
                dist = 0
                if price > z.top: dist = price - z.top
                elif price < z.bottom: dist = z.bottom - price
                if dist <= threshold: return True
            return False
            
        def get_nearest_dist(price, zones):
            min_dist = 999.0
            for z in zones:
                if z.mitigated: continue
                mid = (z.top + z.bottom) / 2
                dist = abs(price - mid)
                if dist < min_dist: min_dist = dist
            return min_dist

        # Active M1 Zones
        active_fvgs = [f for f in self.smc.fvgs if not f.mitigated]
        active_swing_obs = [ob for ob in self.smc.swing_obs if not ob.mitigated]
        active_int_obs = [ob for ob in self.smc.internal_obs if not ob.mitigated]
        
        bull_fvgs = [f for f in active_fvgs if f.type == 1]
        bear_fvgs = [f for f in active_fvgs if f.type == -1]
        
        bull_swing_obs = [ob for ob in active_swing_obs if ob.type == 1]
        bear_swing_obs = [ob for ob in active_swing_obs if ob.type == -1]
        
        bull_int_obs = [ob for ob in active_int_obs if ob.type == 1]
        bear_int_obs = [ob for ob in active_int_obs if ob.type == -1]
        
        near_threshold = 10.0 * self.tick_size
        
        # M1 Distances
        int_swing_high_dist = ((bar.c - state.internal_high.price) / self.tick_size) if state.internal_high else 0.0
        int_swing_low_dist = ((state.internal_low.price - bar.c) / self.tick_size) if state.internal_low else 0.0
        
        ext_swing_high_dist = ((bar.c - state.swing_high.price) / self.tick_size) if state.swing_high else 0.0
        ext_swing_low_dist = ((state.swing_low.price - bar.c) / self.tick_size) if state.swing_low else 0.0
        
        dist_to_nearest_fvg = get_nearest_dist(bar.c, active_fvgs)
        dist_to_nearest_ob = get_nearest_dist(bar.c, active_swing_obs + active_int_obs)
        
        # Nearest FVG Size
        nearest_fvg_size = 0.0
        min_dist = 999.0
        for z in active_fvgs:
            mid = (z.top + z.bottom) / 2
            dist = abs(bar.c - mid)
            if dist < min_dist:
                min_dist = dist
                nearest_fvg_size = abs(z.top - z.bottom)

        # Build FeatureBar
        return FeatureBar(
            # Price/OHLCV features
            close=bar.c,
            high_low_range=bar.h - bar.l,
            body=abs(bar.c - bar.o),
            upper_wick=bar.h - max(bar.o, bar.c),
            lower_wick=min(bar.o, bar.c) - bar.l,
            close_return=0.0,
            volume=bar.volume,
            volume_change=0.0,
            
            # Orderflow features
            delta=bar.delta,
            delta_over_volume=(bar.delta / bar.volume if bar.volume > 0 else 0.0),
            buy_volume=bar.buy_volume,
            sell_volume=bar.sell_volume,
            buy_sell_ratio=(bar.buy_volume / bar.sell_volume if bar.sell_volume > 0 else 0.0),
            
            tick_speed=bar.tick_speed,
            aggr_buy_speed=bar.aggr_buy_speed,
            aggr_sell_speed=bar.aggr_sell_speed,
            price_speed=bar.price_speed,
            
            # Internal SMC features
            int_trend_dir=state.internal_trend,
            int_bos_up=state.internal_bos_bull,
            int_bos_down=state.internal_bos_bear,
            int_choch_up=state.internal_choch_bull,
            int_choch_down=state.internal_choch_bear,
            int_swing_high_distance=int_swing_high_dist,
            int_swing_low_distance=int_swing_low_dist,
            bars_since_int_swing_high=(self.bar_count - state.internal_high.bar_index) if state.internal_high else 999,
            bars_since_int_swing_low=(self.bar_count - state.internal_low.bar_index) if state.internal_low else 999,
            swept_prev_int_high=state.swept_prev_int_high,
            swept_prev_int_low=state.swept_prev_int_low,
            int_bias_bullish=(state.internal_trend > 0),
            
            # External SMC features
            ext_trend_dir=state.swing_trend,
            ext_bos_up=state.bos_bull,
            ext_bos_down=state.bos_bear,
            ext_choch_up=state.choch_bull,
            ext_choch_down=state.choch_bear,
            ext_swing_high_distance=ext_swing_high_dist,
            ext_swing_low_distance=ext_swing_low_dist,
            bars_since_ext_swing_high=(self.bar_count - state.swing_high.bar_index) if state.swing_high else 999,
            bars_since_ext_swing_low=(self.bar_count - state.swing_low.bar_index) if state.swing_low else 999,
            swept_prev_ext_high=state.swept_prev_ext_high,
            swept_prev_ext_low=state.swept_prev_ext_low,
            ext_bias_bullish=(state.swing_trend > 0),
            
            # Zone features
            in_bull_fvg=is_in_zone(bar.c, bull_fvgs),
            in_bear_fvg=is_in_zone(bar.c, bear_fvgs),
            near_bull_fvg=is_near_zone(bar.c, bull_fvgs, near_threshold),
            near_bear_fvg=is_near_zone(bar.c, bear_fvgs, near_threshold),
            
            # Internal OB
            int_in_bull_ob=is_in_zone(bar.c, bull_int_obs),
            int_in_bear_ob=is_in_zone(bar.c, bear_int_obs),
            int_near_bull_ob=is_near_zone(bar.c, bull_int_obs, near_threshold),
            int_near_bear_ob=is_near_zone(bar.c, bear_int_obs, near_threshold),
            
            # External OB
            ext_in_bull_ob=is_in_zone(bar.c, bull_swing_obs),
            ext_in_bear_ob=is_in_zone(bar.c, bear_swing_obs),
            ext_near_bull_ob=is_near_zone(bar.c, bull_swing_obs, near_threshold),
            ext_near_bear_ob=is_near_zone(bar.c, bear_swing_obs, near_threshold),
            
            dist_to_nearest_fvg=dist_to_nearest_fvg,
            dist_to_nearest_ob=dist_to_nearest_ob,
            nearest_fvg_size=nearest_fvg_size,
            
            # Volume Profile (from VP builder)
            vp_poc_price=vp.poc_price,
            vp_val_price=vp.val_price,
            vp_vah_price=vp.vah_price,
            vp_in_value_area=vp.in_value_area,
            vp_above_value_area=vp.above_value_area,
            vp_below_value_area=vp.below_value_area,
            vp_dist_to_poc=vp.dist_to_poc,
            vp_dist_to_vah=vp.dist_to_vah,
            vp_dist_to_val=vp.dist_to_val,
            
            # Wave Strength (v2)
            impulse_strength=wave['impulse_strength'],
            pullback_strength=wave['pullback_strength'],
            cum_delta_5=wave['cum_delta_5'],
            cum_delta_10=wave['cum_delta_10'],
            cum_delta_20=wave['cum_delta_20'],
            
            # VWAP features
            vwap_daily=bar.vwap_daily,
            dist_to_vwap=((bar.c - bar.vwap_daily) / self.tick_size if bar.vwap_daily > 0 else 0.0),
            
            # M5 / H1 Context (replaced H4 with M5 for M1 trading)
            **m5_feats,
            **h1_feats,
            
            # Enhanced macro features (10 new)
            **m5_enhanced,
            **h1_enhanced
        )

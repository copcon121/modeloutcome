"""
Event Filtering for ML Dataset
Single-pass filtering with 3 phases: P1 (strict), P2 (moderate), P3 (loose)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from .schema import FeatureBar


@dataclass
class EventFlags:
    """Flags for event detection - calculated once per bar"""
    # Structure events
    has_bos_choch: bool
    has_major_bos_choch: bool  # External only
    
    # Zone events
    has_new_fvg_ob: bool  # Detected on this bar
    in_zone: bool  # Currently in FVG or OB
    near_zone: bool  # Near FVG or OB
    
    # Volatility events
    high_range: bool
    high_volume: bool
    high_delta: bool
    high_tick_speed: bool
    
    # Liquidity sweeps
    ls_event: bool
    
    # Level interactions
    vwap_interaction: bool
    vp_interaction: bool
    
    # Derived
    high_volatility: bool  # Any volatility flag
    
    def __post_init__(self):
        """Calculate derived flags"""
        self.high_volatility = (self.high_range or self.high_volume or 
                               self.high_delta or self.high_tick_speed)


class EventFilter:
    """
    Event-based filtering for ML datasets
    
    Usage:
        filter = EventFilter()
        flags_list = filter.compute_flags(feature_bars, context)
        mask_p2 = filter.apply_phase2_filter(flags_list)
        filtered_bars = [fb for fb, keep in zip(feature_bars, mask_p2) if keep]
    """
    
    def __init__(self):
        """Initialize filter"""
        pass
    
    def compute_flags(
        self,
        feature_bars: List[FeatureBar],
        context: Dict = None
    ) -> List[EventFlags]:
        """
        Single pass: compute all event flags
        
        Args:
            feature_bars: List of feature bars
            context: Optional context with averages
            
        Returns:
            List of EventFlags (one per bar)
        """
        if context is None:
            context = self._build_context(feature_bars)
        
        flags_list = []
        
        for i, fb in enumerate(feature_bars):
            # 1. Structure events
            has_bos_choch = (
                fb.int_bos_up or fb.int_bos_down or
                fb.int_choch_up or fb.int_choch_down or
                fb.ext_bos_up or fb.ext_bos_down or
                fb.ext_choch_up or fb.ext_choch_down
            )
            
            has_major_bos_choch = (
                fb.ext_bos_up or fb.ext_bos_down or
                fb.ext_choch_up or fb.ext_choch_down
            )
            
            # 2. Zone events
            # NEW FVG/OB: check if this bar created one
            # NOTE: We'd need to track zone creation in detectors
            # For now, use proxy: in_zone changed from previous bar
            has_new_fvg_ob = False  # TODO: implement zone creation tracking
            
            in_zone = (fb.in_bull_fvg or fb.in_bear_fvg or
                      fb.in_bull_ob or fb.in_bear_ob)
            
            near_zone = (fb.near_bull_fvg or fb.near_bear_fvg or
                        fb.near_bull_ob or fb.near_bear_ob or
                        fb.dist_to_nearest_fvg < 20 or  # Within 20 ticks
                        fb.dist_to_nearest_ob < 20)
            
            # 3. Volatility events (STRICTER THRESHOLDS)
            avg_range = context.get('avg_range', 2.0)
            avg_volume = context.get('avg_volume', 100)
            avg_delta = context.get('avg_delta', 20)
            avg_tick_speed = context.get('avg_tick_speed', 500)
            
            # Stricter thresholds to reduce P2 from 60% to 20-50%
            high_range = fb.high_low_range > avg_range * 2.5  # Was 2.0
            high_volume = fb.volume > avg_volume * 2.0  # Was 1.5
            high_delta = abs(fb.delta_over_volume) > 0.65  # Was 0.5 (50% → 65%)
            high_tick_speed = fb.tick_speed > avg_tick_speed * 2.0  # Was 1.5
            
            # 4. Liquidity sweeps
            ls_event = fb.swept_prev_int_high or fb.swept_prev_int_low
            
            # 5. VWAP interaction (STRICTER)
            vwap_interaction = abs(fb.dist_to_vwap) < 3  # Was 5 ticks → now 3
            
            # 6. VP interaction (STRICTER)
            vp_interaction = (
                fb.vp_in_value_area or
                abs(fb.vp_dist_to_poc) < 5 or  # Was 10 ticks → now 5
                fb.vp_above_value_area != (i > 0 and flags_list[i-1].vp_interaction)
            ) if fb.vp_poc_price > 0 else False
            
            flags = EventFlags(
                has_bos_choch=has_bos_choch,
                has_major_bos_choch=has_major_bos_choch,
                has_new_fvg_ob=has_new_fvg_ob,
                in_zone=in_zone,
                near_zone=near_zone,
                high_range=high_range,
                high_volume=high_volume,
                high_delta=high_delta,
                high_tick_speed=high_tick_speed,
                ls_event=ls_event,
                vwap_interaction=vwap_interaction,
                vp_interaction=vp_interaction,
                high_volatility=False  # Will be set in __post_init__
            )
            
            flags_list.append(flags)
        
        return flags_list
    
    def _build_context(self, feature_bars: List[FeatureBar]) -> Dict:
        """Calculate average values for thresholds"""
        ranges = [fb.high_low_range for fb in feature_bars]
        volumes = [fb.volume for fb in feature_bars if fb.volume > 0]
        deltas = [abs(fb.delta) for fb in feature_bars]
        tick_speeds = [fb.tick_speed for fb in feature_bars if fb.tick_speed > 0]
        
        return {
            'avg_range': np.mean(ranges) if ranges else 2.0,
            'avg_volume': np.mean(volumes) if volumes else 100,
            'avg_delta': np.mean(deltas) if deltas else 20,
            'avg_tick_speed': np.mean(tick_speeds) if tick_speeds else 500
        }
    
    def apply_phase1_filter(self, flags_list: List[EventFlags]) -> List[bool]:
        """
        Phase 1: STRICT - Reversal specialist
        Keep: BOS/CHoCH (major) + new FVG/OB creation
        """
        return [
            f.has_major_bos_choch or f.has_new_fvg_ob
            for f in flags_list
        ]
    
    def apply_phase2_filter(self, flags_list: List[EventFlags]) -> List[bool]:
        """
        Phase 2: MODERATE - Balanced (RECOMMENDED)
        Keep: P1 OR (structure + high volatility + zone interaction)
        """
        mask_p1 = self.apply_phase1_filter(flags_list)
        
        return [
            mask_p1[i] or (
                f.has_bos_choch or  # Any structure
                f.high_volatility or  # High volatility
                f.in_zone or  # In zone
                f.ls_event  # Liquidity sweep
            )
            for i, f in enumerate(flags_list)
        ]
    
    def apply_phase3_filter(self, flags_list: List[EventFlags]) -> List[bool]:
        """
        Phase 3: LOOSE - More context
        Keep: P2 OR (VWAP/VP interactions + near zones)
        """
        mask_p2 = self.apply_phase2_filter(flags_list)
        
        return [
            mask_p2[i] or (
                f.vwap_interaction or
                f.vp_interaction or
                f.near_zone
            )
            for i, f in enumerate(flags_list)
        ]
    
    def tag_bars_with_phase(
        self,
        flags_list: List[EventFlags]
    ) -> List[str]:
        """
        Tag each bar with minimum phase (P1/P2/P3/None)
        Bar in P1 → tagged as P1 (not P2/P3)
        """
        mask_p1 = self.apply_phase1_filter(flags_list)
        mask_p2 = self.apply_phase2_filter(flags_list)
        mask_p3 = self.apply_phase3_filter(flags_list)
        
        tags = []
        for i in range(len(flags_list)):
            if mask_p1[i]:
                tags.append('P1')
            elif mask_p2[i]:
                tags.append('P2')
            elif mask_p3[i]:
                tags.append('P3')
            else:
                tags.append('None')
        
        return tags
    
    def get_filter_stats(
        self,
        flags_list: List[EventFlags]
    ) -> Dict:
        """Get statistics for each filter phase"""
        total = len(flags_list)
        mask_p1 = self.apply_phase1_filter(flags_list)
        mask_p2 = self.apply_phase2_filter(flags_list)
        mask_p3 = self.apply_phase3_filter(flags_list)
        
        return {
            'total_bars': total,
            'p1_strict': {
                'count': sum(mask_p1),
                'pct': sum(mask_p1) / total * 100 if total > 0 else 0
            },
            'p2_moderate': {
                'count': sum(mask_p2),
                'pct': sum(mask_p2) / total * 100 if total > 0 else 0
            },
            'p3_loose': {
                'count': sum(mask_p3),
                'pct': sum(mask_p3) / total * 100 if total > 0 else 0
            }
        }

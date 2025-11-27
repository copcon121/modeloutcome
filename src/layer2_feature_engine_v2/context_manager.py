"""
Phase 2 Feature Engine V2 - Context Manager
Orchestrates all SMC detectors and builds FeatureBar output
"""

from typing import Optional
import logging

from .schema import RawBar, FeatureBar, InternalSwingState, ExternalSwingState
from .config import SMCConfig
from .smc_core.swing import InternalSwingDetector, ExternalSwingDetector
from .smc_core.structure import StructureDetector
from .smc_core.zones import PDZoneTracker, FVGDetector, OBDetector

logger = logging.getLogger(__name__)


class SMCContextManager:
    """
    SMC Context Manager
    
    Orchestrates all SMC detectors and builds unified FeatureBar output.
    
    Architecture:
    1. Internal/External Swing Detection (window 5/50)
    2. Structure Detection (BOS/CHoCH) for both layers
    3. PD Zone Tracking (uses External swings)
    4. FVG Detection (3-bar gap pattern)
    5. OB Detection (from External structure breaks)
    
    Usage:
        manager = SMCContextManager(config, tick_size=0.1)
        for raw_bar in bar_iterator:
            feature_bar = manager.update(raw_bar)
            # feature_bar contains all SMC features
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        """
        Initialize all SMC detectors
        
        Args:
            config: SMC configuration
            tick_size: Instrument tick size (e.g., 0.1 for GC)
        """
        self.config = config
        self.tick_size = tick_size
        
        # Swing Detectors
        self.int_swing = InternalSwingDetector(config, tick_size)
        self.ext_swing = ExternalSwingDetector(config, tick_size)
        
        # Structure Detector (handles both internal and external)
        self.structure = StructureDetector(config, tick_size)
        
        # Zone Detectors
        self.pd_tracker = PDZoneTracker(config)
        self.fvg_detector = FVGDetector(config, tick_size)
        self.ob_detector = OBDetector(config, tick_size)
        
        logger.info(f"SMCContextManager initialized with tick_size={tick_size}")
    
    def update(self, bar: RawBar) -> FeatureBar:
        """
        Update all detectors with new bar and build FeatureBar
        
        Processing order (CRITICAL):
        1. Update swing detectors (internal + external)
        2. Update structure detector (BOS/CHoCH for both layers)
        3. Update PD zones (uses external swings)
        4. Update FVG detector (3-bar pattern)
        5. Update OB detector (uses external structure)
        6. Assemble FeatureBar
        
        Args:
            bar: Raw market bar
            
        Returns:
            FeatureBar with all SMC features populated
        """
        # 1. Swing Detection
        int_swing_state = self.int_swing.update(bar)
        ext_swing_state = self.ext_swing.update(bar)
        
        # 2. Structure Detection
        self.structure.update_internal(bar, int_swing_state)
        self.structure.update_external(bar, ext_swing_state)
        
        int_struct = self.structure.get_internal_state()
        ext_struct = self.structure.get_external_state()
        
        # 3. PD Zones (uses external swings)
        pd_state = self.pd_tracker.update(bar, ext_swing_state)
        
        # 4. FVG Detection
        new_fvgs = self.fvg_detector.update(bar)
        
        # 5. OB Detection (uses EXTERNAL structure!)
        new_obs = self.ob_detector.update(bar, ext_struct)
        
        # 6. Build FeatureBar
        feature_bar = self._build_feature_bar(
            bar, int_swing_state, ext_swing_state,
            int_struct, ext_struct, pd_state
        )
        
        return feature_bar
    
    def _build_feature_bar(
        self,
        bar: RawBar,
        int_swing: InternalSwingState,
        ext_swing: ExternalSwingState,
        int_struct: dict,
        ext_struct: dict,
        pd_state
    ) -> FeatureBar:
        """
        Build FeatureBar from all detector states
        
        Maps to FeatureBar schema fields (feature names, not metadata)
        """
        # Get zone counts
        active_fvgs = self.fvg_detector.get_active_fvgs()
        all_fvgs = self.fvg_detector.get_all_fvgs()
        active_obs = self.ob_detector.get_active_obs()
        all_obs = self.ob_detector.get_all_obs()
        
        # Calculate derived features
        body = abs(bar.c - bar.o)
        hl_range = bar.h - bar.l
        upper_wick = bar.h - max(bar.o, bar.c)
        lower_wick = min(bar.o, bar.c) - bar.l
        
        # Distance to swings (in ticks)
        int_swing_high_dist = ((bar.c - int_swing.swing_high_price) / self.tick_size 
                               if int_swing.swing_high_price else 0.0)
        int_swing_low_dist = ((int_swing.swing_low_price - bar.c) / self.tick_size 
                              if int_swing.swing_low_price else 0.0)
        ext_swing_high_dist = ((bar.c - ext_swing.swing_high_price) / self.tick_size 
                               if ext_swing.swing_high_price else 0.0)
        ext_swing_low_dist = ((ext_swing.swing_low_price - bar.c) / self.tick_size 
                              if ext_swing.swing_low_price else 0.0)
        
        # Find nearest zones
        nearest_fvg_dist = min([abs(bar.c - fvg.midpoint) for fvg in active_fvgs], default=999.0)
        nearest_ob_dist = min([abs(bar.c - (ob.top + ob.bottom) / 2) for ob in active_obs], default=999.0)
        
        # Check if in zones
        in_bull_fvg = any(fvg.is_bullish and fvg.bottom <= bar.c <= fvg.top for fvg in active_fvgs)
        in_bear_fvg = any(not fvg.is_bullish and fvg.bottom <= bar.c <= fvg.top for fvg in active_fvgs)
        in_bull_ob = any(ob.is_bullish and ob.bottom <= bar.c <= ob.top for ob in active_obs)
        in_bear_ob = any(not ob.is_bullish and ob.bottom <= bar.c <= ob.top for ob in active_obs)
        
        return FeatureBar(
            # Price/OHLCV features
            close=bar.c,
            high_low_range=hl_range,
            body=body,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            close_return=0.0,  # TODO: need previous bar
            volume=bar.volume,
            volume_change=0.0,  # TODO: need previous bar
            
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
            int_trend_dir=int_struct['structure_dir'],
            int_bos_up=int_struct['bos_up'],
            int_bos_down=int_struct['bos_down'],
            int_choch_up=int_struct['choch_up'],
            int_choch_down=int_struct['choch_down'],
            int_swing_high_distance=int_swing_high_dist,
            int_swing_low_distance=int_swing_low_dist,
            bars_since_int_swing_high=999,  # TODO: calculate
            bars_since_int_swing_low=999,   # TODO: calculate
            swept_prev_int_high=False,  # TODO: implement sweep detection
            swept_prev_int_low=False,
            int_bias_bullish=(int_struct['structure_dir'] > 0),
            
            # External SMC features
            ext_trend_dir=ext_struct['structure_dir'],
            ext_bos_up=ext_struct['bos_up'],
            ext_bos_down=ext_struct['bos_down'],
            ext_choch_up=ext_struct['choch_up'],
            ext_choch_down=ext_struct['choch_down'],
            ext_swing_high_distance=ext_swing_high_dist,
            ext_swing_low_distance=ext_swing_low_dist,
            bars_since_ext_swing_high=999,  # TODO: calculate
            bars_since_ext_swing_low=999,   # TODO: calculate
            swept_prev_ext_high=False,  # TODO: implement
            swept_prev_ext_low=False,
            ext_bias_bullish=(ext_struct['structure_dir'] > 0),
            
            # Zone features
            in_bull_fvg=in_bull_fvg,
            in_bear_fvg=in_bear_fvg,
            near_bull_fvg=False,  # TODO: define "near"
            near_bear_fvg=False,
            in_bull_ob=in_bull_ob,
            in_bear_ob=in_bear_ob,
            near_bull_ob=False,
            near_bear_ob=False,
            dist_to_nearest_fvg=nearest_fvg_dist,
            dist_to_nearest_ob=nearest_ob_dist,
            
            # Volume Profile (placeholder)
            vp_poc_price=0.0,
            vp_val_price=0.0,
            vp_vah_price=0.0,
            vp_in_value_area=False,
            vp_above_value_area=False,
            vp_below_value_area=False,
            vp_dist_to_poc=0.0,
            vp_dist_to_vah=0.0,
            vp_dist_to_val=0.0,
            
            # VWAP features
            vwap_daily=bar.vwap_daily,
            dist_to_vwap=((bar.c - bar.vwap_daily) / self.tick_size if bar.vwap_daily > 0 else 0.0)
        )
    
    def get_swing_states(self):
        """Get current swing detector states (for debugging)"""
        return {
            'internal': self.int_swing.state,
            'external': self.ext_swing.state
        }
    
    def get_structure_state(self):
        """Get current structure detector state (for debugging)"""
        return {
            'internal': self.structure.get_internal_state(),
            'external': self.structure.get_external_state()
        }
    
    def get_zone_summary(self):
        """Get summary of all zones (for debugging)"""
        return {
            'pd': self.pd_tracker.get_state(),
            'fvgs': {
                'active': len(self.fvg_detector.get_active_fvgs()),
                'total': len(self.fvg_detector.get_all_fvgs())
            },
            'obs': {
                'active': len(self.ob_detector.get_active_obs()),
                'total': len(self.ob_detector.get_all_obs())
            }
        }

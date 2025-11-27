"""
Volume Profile Builder
Builds volume profile from RawBar data (volume-only, no delta)
"""

import math
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from .vp_config import VPConfig, SessionDef
from .vp_state import VolumeProfileState
from ..schema import RawBar

logger = logging.getLogger(__name__)


class VolumeProfileBuilder:
    """
    Volume Profile Builder
    
    Builds volume profile using only volume (no delta).
    Supports daily or session-based profiles.
    
    Design: Extensible for future delta profile without major refactoring.
    """
    
    def __init__(self, config: VPConfig):
        """
        Initialize VP builder
        
        Args:
            config: VP configuration
        """
        self.config = config
        
        # Current profile histogram: price_level -> volume
        self._hist: Dict[float, float] = {}
        
        # Current profile ID
        self._current_profile_id: Optional[str] = None
        
        # Cached VP values (recomputed when histogram changes)
        self._cached_poc: Optional[float] = None
        self._cached_val: Optional[float] = None
        self._cached_vah: Optional[float] = None
        
        logger.info(f"VolumeProfileBuilder initialized: mode={config.mode}")
    
    def _get_profile_id(self, bar: RawBar) -> str:
        """
        Get profile ID for a bar based on config mode
        
        Args:
            bar: Raw bar
            
        Returns:
            Profile ID string
        """
        timestamp = bar.timestamp
        date_str = timestamp.date().isoformat()
        
        if self.config.mode == "daily":
            return date_str
        
        # Session mode
        hour = timestamp.hour
        
        # Find matching session
        for session in self.config.sessions:
            if session.start_hour <= hour < session.end_hour:
                return f"{date_str}_{session.name}"
        
        # Fallback: Unknown session
        return f"{date_str}_Unknown"
    
    def _reset_profile(self, profile_id: str) -> None:
        """Reset histogram for new profile"""
        self._hist.clear()
        self._current_profile_id = profile_id
        self._cached_poc = None
        self._cached_val = None
        self._cached_vah = None
        logger.debug(f"Reset profile: {profile_id}")
    
    def _update_histogram(self, bar: RawBar) -> None:
        """
        Update histogram with bar volume
        
        Simple approximation: distribute volume evenly across price bins
        """
        tick = self.config.tick_size
        low = bar.l
        high = bar.h
        vol = bar.volume
        
        if high <= low or vol <= 0:
            return
        
        # Calculate number of bins
        n_bins = max(1, int(round((high - low) / tick)) + 1)
        vol_per_bin = vol / n_bins
        
        # Distribute volume
        price = math.floor(low / tick) * tick
        while price <= high:
            self._hist[price] = self._hist.get(price, 0.0) + vol_per_bin
            price += tick
    
    def _compute_vp_levels(self) -> Tuple[float, float, float]:
        """
        Compute POC, VAL, VAH from histogram
        
        Returns:
            (poc_price, val_price, vah_price)
        """
        if not self._hist:
            # Empty histogram - return zeros
            return (0.0, 0.0, 0.0)
        
        # 1. Compute POC (price with max volume)
        poc_price = max(self._hist.items(), key=lambda kv: kv[1])[0]
        
        # 2. Compute VAL/VAH using 70% value area
        total_vol = sum(self._hist.values())
        target_vol = total_vol * self.config.value_area_pct
        
        # Sort prices by distance from POC
        prices = sorted(self._hist.keys())
        
        # Expand from POC outward to capture target_vol
        # Strategy: Start from POC, add nearest prices alternately
        poc_idx = prices.index(poc_price)
        
        included = {poc_price}
        cumulative_vol = self._hist[poc_price]
        
        left = poc_idx - 1
        right = poc_idx + 1
        
        while cumulative_vol < target_vol and (left >= 0 or right < len(prices)):
            # Decide whether to expand left or right
            # Choose side with higher volume
            left_vol = self._hist.get(prices[left], 0.0) if left >= 0 else 0.0
            right_vol = self._hist.get(prices[right], 0.0) if right < len(prices) else 0.0
            
            if left >= 0 and (right >= len(prices) or left_vol >= right_vol):
                # Add left
                included.add(prices[left])
                cumulative_vol += left_vol
                left -= 1
            elif right < len(prices):
                # Add right
                included.add(prices[right])
                cumulative_vol += right_vol
                right += 1
            else:
                break
        
        # VAL/VAH are min/max of included prices
        val_price = min(included)
        vah_price = max(included)
        
        return (poc_price, val_price, vah_price)
    
    def update(self, bar: RawBar) -> VolumeProfileState:
        """
        Update VP with new bar
        
        Args:
            bar: Raw bar
            
        Returns:
            VolumeProfileState for this bar
        """
        # 1. Determine profile ID
        profile_id = self._get_profile_id(bar)
        
        # 2. Reset if new profile
        if profile_id != self._current_profile_id:
            self._reset_profile(profile_id)
        
        # 3. Update histogram
        self._update_histogram(bar)
        
        # 4. Compute VP levels
        poc_price, val_price, vah_price = self._compute_vp_levels()
        
        # 5. Compute position flags and distances
        c = bar.c
        
        in_value_area = (c >= val_price) and (c <= vah_price) if val_price and vah_price else False
        above_value_area = c > vah_price if vah_price else False
        below_value_area = c < val_price if val_price else False
        
        dist_to_poc = (c - poc_price) / self.config.tick_size if poc_price else 0.0
        dist_to_vah = (c - vah_price) / self.config.tick_size if vah_price else 0.0
        dist_to_val = (c - val_price) / self.config.tick_size if val_price else 0.0
        
        # 6. Return state
        return VolumeProfileState(
            profile_id=profile_id,
            poc_price=poc_price,
            val_price=val_price,
            vah_price=vah_price,
            in_value_area=in_value_area,
            above_value_area=above_value_area,
            below_value_area=below_value_area,
            dist_to_poc=dist_to_poc,
            dist_to_vah=dist_to_vah,
            dist_to_val=dist_to_val,
            total_volume=sum(self._hist.values()),
            num_price_levels=len(self._hist)
        )
    
    def get_current_histogram(self) -> Dict[float, float]:
        """Get current histogram (for debugging)"""
        return self._hist.copy()

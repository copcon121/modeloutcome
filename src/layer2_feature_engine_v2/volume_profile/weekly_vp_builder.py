"""
Weekly Volume Profile Builder
Builds volume profile at weekly timeframe for ASM v1.x
"""

import math
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from .vp_config import VPConfig
from .vp_state import VolumeProfileState
from ..schema import RawBar

logger = logging.getLogger(__name__)


class WeeklyVolumeProfileBuilder:
    """
    Weekly Volume Profile Builder
    
    Builds volume profile using weekly aggregation.
    Week is defined as Monday 00:00 to Sunday 23:59 (or Friday close for futures).
    """
    
    def __init__(self, config: VPConfig):
        """
        Initialize Weekly VP builder
        
        Args:
            config: VP configuration (uses tick_size and value_area_pct)
        """
        self.config = config
        
        # Current profile histogram: price_level -> volume
        self._hist: Dict[float, float] = {}
        
        # Current week ID (year-week format)
        self._current_week_id: Optional[str] = None
        
        # Cached VP values
        self._cached_poc: float = 0.0
        self._cached_val: float = 0.0
        self._cached_vah: float = 0.0
        
        logger.info("WeeklyVolumeProfileBuilder initialized")
    
    def _get_week_id(self, timestamp: datetime) -> str:
        """
        Get week ID for a timestamp.
        Uses ISO week number (Monday = start of week).
        
        Args:
            timestamp: Bar timestamp
            
        Returns:
            Week ID string (e.g., "2025-W48")
        """
        iso_cal = timestamp.isocalendar()
        return f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    
    def _reset_profile(self, week_id: str) -> None:
        """Reset histogram for new week"""
        self._hist.clear()
        self._current_week_id = week_id
        self._cached_poc = 0.0
        self._cached_val = 0.0
        self._cached_vah = 0.0
        logger.debug(f"Reset weekly profile: {week_id}")
    
    def _update_histogram(self, bar: RawBar) -> None:
        """
        Update histogram with bar volume.
        Distributes volume evenly across price bins.
        """
        tick = self.config.tick_size
        low = bar.l
        high = bar.h
        vol = bar.volume
        
        if high <= low or vol <= 0:
            return
        
        n_bins = max(1, int(round((high - low) / tick)) + 1)
        vol_per_bin = vol / n_bins
        
        price = math.floor(low / tick) * tick
        while price <= high:
            self._hist[price] = self._hist.get(price, 0.0) + vol_per_bin
            price += tick
    
    def _compute_vp_levels(self) -> Tuple[float, float, float]:
        """
        Compute POC, VAL, VAH from histogram.
        
        Returns:
            (poc_price, val_price, vah_price)
        """
        if not self._hist:
            return (0.0, 0.0, 0.0)
        
        # POC = price with max volume
        poc_price = max(self._hist.items(), key=lambda kv: kv[1])[0]
        
        # Value Area (70% of volume)
        total_vol = sum(self._hist.values())
        target_vol = total_vol * self.config.value_area_pct
        
        prices = sorted(self._hist.keys())
        poc_idx = prices.index(poc_price)
        
        included = {poc_price}
        cumulative_vol = self._hist[poc_price]
        
        left = poc_idx - 1
        right = poc_idx + 1
        
        while cumulative_vol < target_vol and (left >= 0 or right < len(prices)):
            left_vol = self._hist.get(prices[left], 0.0) if left >= 0 else 0.0
            right_vol = self._hist.get(prices[right], 0.0) if right < len(prices) else 0.0
            
            if left >= 0 and (right >= len(prices) or left_vol >= right_vol):
                included.add(prices[left])
                cumulative_vol += left_vol
                left -= 1
            elif right < len(prices):
                included.add(prices[right])
                cumulative_vol += right_vol
                right += 1
            else:
                break
        
        val_price = min(included)
        vah_price = max(included)
        
        return (poc_price, val_price, vah_price)
    
    def update(self, bar: RawBar) -> Dict[str, float]:
        """
        Update Weekly VP with new bar.
        
        Args:
            bar: Raw bar
            
        Returns:
            Dict with weekly VA features
        """
        # Determine week ID
        week_id = self._get_week_id(bar.timestamp)
        
        # Reset if new week
        if week_id != self._current_week_id:
            self._reset_profile(week_id)
        
        # Update histogram
        self._update_histogram(bar)
        
        # Compute VP levels
        poc, val, vah = self._compute_vp_levels()
        self._cached_poc = poc
        self._cached_val = val
        self._cached_vah = vah
        
        # Compute features
        c = bar.c
        tick = self.config.tick_size
        
        weekly_va_center = (vah + val) / 2 if vah > 0 and val > 0 else 0.0
        
        in_weekly_va = 1.0 if (val <= c <= vah and vah > 0) else 0.0
        dist_to_weekly_vah = (c - vah) / tick if vah > 0 else 0.0
        dist_to_weekly_val = (c - val) / tick if val > 0 else 0.0
        
        return {
            "weekly_vah": vah,
            "weekly_val": val,
            "weekly_va_center": weekly_va_center,
            "weekly_poc": poc,
            "in_weekly_va": in_weekly_va,
            "dist_to_weekly_vah": dist_to_weekly_vah,
            "dist_to_weekly_val": dist_to_weekly_val,
        }
    
    @property
    def current_vah(self) -> float:
        return self._cached_vah
    
    @property
    def current_val(self) -> float:
        return self._cached_val
    
    @property
    def current_poc(self) -> float:
        return self._cached_poc

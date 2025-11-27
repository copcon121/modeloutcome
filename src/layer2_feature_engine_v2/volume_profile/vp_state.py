"""
Volume Profile State
Represents current VP state for a bar
"""

from dataclasses import dataclass


@dataclass
class VolumeProfileState:
    """
    Volume Profile State
    
    Represents the VP state at a given bar.
    Contains POC, VAL, VAH and derived features.
    """
    # Profile identification
    profile_id: str  # e.g., "2025-11-17" (daily) or "2025-11-17_Asia" (session)
    
    # Core VP levels
    poc_price: float  # Point of Control (highest volume price)
    val_price: float  # Value Area Low
    vah_price: float  # Value Area High
    
    # Position flags
    in_value_area: bool      # Price is inside [VAL, VAH]
    above_value_area: bool   # Price is above VAH
    below_value_area: bool   # Price is below VAL
    
    # Distance features (in ticks)
    dist_to_poc: float  # (close - POC) / tick_size
    dist_to_vah: float  # (close - VAH) / tick_size
    dist_to_val: float  # (close - VAL) / tick_size
    
    # Stats (optional, for debugging)
    total_volume: float = 0.0
    num_price_levels: int = 0

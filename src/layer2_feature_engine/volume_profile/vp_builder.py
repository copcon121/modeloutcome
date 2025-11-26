"""
Volume Profile Builder
Computes VAH, VAL, POC, HVN, LVN for a given window of bars
"""

from typing import List, Dict
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar, VolumeProfileState


def build_volume_profile(bars: List[RawBar], price_bins: int = 50) -> VolumeProfileState:
    """
    Build Volume Profile for a window of bars

    Args:
        bars: List of RawBar objects
        price_bins: Number of price bins to divide the range

    Returns:
        VolumeProfileState object
    """
    if len(bars) < 10:
        # Not enough data, return default state
        return _default_vp_state()

    # Find price range
    all_prices = []
    for bar in bars:
        all_prices.extend([bar.high, bar.low])

    min_price = min(all_prices)
    max_price = max(all_prices)

    if max_price - min_price < 1e-8:
        return _default_vp_state()

    # Create price bins
    bin_edges = np.linspace(min_price, max_price, price_bins + 1)
    bin_volumes = np.zeros(price_bins)

    # Distribute volume to bins
    for bar in bars:
        # Simple approach: distribute volume proportionally across bar's range
        bar_range = bar.high - bar.low
        if bar_range < 1e-8:
            # Single price bar, put all volume in one bin
            bin_idx = np.searchsorted(bin_edges, bar.close) - 1
            bin_idx = max(0, min(bin_idx, price_bins - 1))
            bin_volumes[bin_idx] += bar.volume
        else:
            # Distribute across bins the bar touches
            start_bin = np.searchsorted(bin_edges, bar.low) - 1
            end_bin = np.searchsorted(bin_edges, bar.high) - 1
            start_bin = max(0, min(start_bin, price_bins - 1))
            end_bin = max(0, min(end_bin, price_bins - 1))

            num_bins = end_bin - start_bin + 1
            vol_per_bin = bar.volume / num_bins

            for i in range(start_bin, end_bin + 1):
                bin_volumes[i] += vol_per_bin

    # Compute POC (Point of Control) - price with highest volume
    poc_bin = np.argmax(bin_volumes)
    poc = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

    # Compute Value Area (70% of volume)
    total_volume = np.sum(bin_volumes)
    value_area_volume = total_volume * 0.70

    # Start from POC and expand to find value area
    value_bins = {poc_bin}
    current_volume = bin_volumes[poc_bin]

    while current_volume < value_area_volume:
        # Find adjacent bins
        candidates = []
        for bin_idx in list(value_bins):
            if bin_idx > 0 and (bin_idx - 1) not in value_bins:
                candidates.append((bin_idx - 1, bin_volumes[bin_idx - 1]))
            if bin_idx < price_bins - 1 and (bin_idx + 1) not in value_bins:
                candidates.append((bin_idx + 1, bin_volumes[bin_idx + 1]))

        if not candidates:
            break

        # Add bin with highest volume
        next_bin = max(candidates, key=lambda x: x[1])[0]
        value_bins.add(next_bin)
        current_volume += bin_volumes[next_bin]

    # VAH and VAL
    value_bins_list = sorted(value_bins)
    vah_bin = value_bins_list[-1]
    val_bin = value_bins_list[0]

    vah = bin_edges[vah_bin + 1]
    val = bin_edges[val_bin]

    # Identify HVN (High Volume Nodes) and LVN (Low Volume Nodes)
    mean_volume = np.mean(bin_volumes)
    std_volume = np.std(bin_volumes)

    hvn_threshold = mean_volume + 1.5 * std_volume
    lvn_threshold = mean_volume - 0.5 * std_volume

    hvn_levels = []
    lvn_levels = []

    for i, vol in enumerate(bin_volumes):
        price = (bin_edges[i] + bin_edges[i + 1]) / 2
        if vol > hvn_threshold:
            hvn_levels.append(price)
        elif vol < lvn_threshold and vol > 0:
            lvn_levels.append(price)

    return VolumeProfileState(
        vah=vah,
        val=val,
        poc=poc,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels
    )


def extract_bar_vp_features(i: int, vp_state: VolumeProfileState, bar: RawBar) -> Dict[str, float]:
    """
    Extract Volume Profile features for a specific bar

    Args:
        i: Bar index
        vp_state: VolumeProfileState object
        bar: RawBar object

    Returns:
        Dict of features
    """
    features = {
        # Distance to VAH, VAL, POC (normalized)
        'dist_to_vah': vp_state.distance_to_vah(bar.close),
        'dist_to_val': vp_state.distance_to_val(bar.close),
        'dist_to_poc': vp_state.distance_to_poc(bar.close),

        # Position within value area
        'in_value_area': 1.0 if vp_state.val <= bar.close <= vp_state.vah else 0.0,
        'above_value_area': 1.0 if bar.close > vp_state.vah else 0.0,
        'below_value_area': 1.0 if bar.close < vp_state.val else 0.0,

        # Near HVN or LVN
        'near_hvn': 1.0 if _is_near_level(bar.close, vp_state.hvn_levels, threshold=0.001) else 0.0,
        'near_lvn': 1.0 if _is_near_level(bar.close, vp_state.lvn_levels, threshold=0.001) else 0.0,
    }

    return features


def _is_near_level(price: float, levels: List[float], threshold: float = 0.001) -> bool:
    """
    Check if price is near any level

    Args:
        price: Current price
        levels: List of price levels
        threshold: Relative distance threshold

    Returns:
        True if near any level
    """
    for level in levels:
        if abs(price - level) / price < threshold:
            return True
    return False


def _default_vp_state() -> VolumeProfileState:
    """Return default VolumeProfileState when not enough data"""
    return VolumeProfileState(
        vah=0.0,
        val=0.0,
        poc=0.0,
        hvn_levels=[],
        lvn_levels=[]
    )

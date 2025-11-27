"""
Tick Analyzer - Extract features from tick-level data
Analyzes tick speed, aggressive buy/sell, and price speed patterns
"""

from typing import List, Dict
import numpy as np

from ..core.schema import RawBar


def extract_tick_features(
    i: int,
    bars_list: List[RawBar],
    lookback: int = 14
) -> Dict[str, float]:
    """
    Extract tick-level orderflow features for bar at index i

    Args:
        i: Current bar index
        bars_list: List of all RawBar objects
        lookback: Number of bars to use for moving averages

    Returns:
        Dictionary of tick features
    """
    bar = bars_list[i]
    features = {}

    # Get historical window for moving averages
    start_idx = max(0, i - lookback + 1)
    window_bars = bars_list[start_idx:i+1]

    # --- Basic Tick Features ---

    # Tick speed (raw and normalized)
    tick_speeds = [b.tick_speed for b in window_bars if b.tick_speed > 0]
    if tick_speeds:
        tick_speed_mean = np.mean(tick_speeds)
        tick_speed_std = np.std(tick_speeds) if len(tick_speeds) > 1 else 1.0

        # Normalized tick speed (z-score)
        features['tick_speed_raw'] = bar.tick_speed
        features['tick_speed_ma'] = tick_speed_mean
        features['tick_speed_norm'] = (
            (bar.tick_speed - tick_speed_mean) / (tick_speed_std + 1e-8)
        )
    else:
        features['tick_speed_raw'] = 0.0
        features['tick_speed_ma'] = 0.0
        features['tick_speed_norm'] = 0.0

    # Tick acceleration (change in tick speed)
    if i > 0:
        prev_tick_speed = bars_list[i-1].tick_speed
        features['tick_acceleration'] = bar.tick_speed - prev_tick_speed
    else:
        features['tick_acceleration'] = 0.0

    # --- Aggressive Buy/Sell Features ---

    # Buy/Sell ratio
    if bar.aggr_sell_speed > 0:
        features['buy_sell_ratio'] = bar.aggr_buy_speed / bar.aggr_sell_speed
    else:
        features['buy_sell_ratio'] = 2.0 if bar.aggr_buy_speed > 0 else 1.0

    # Aggressive buy/sell normalized
    aggr_buys = [b.aggr_buy_speed for b in window_bars]
    aggr_sells = [b.aggr_sell_speed for b in window_bars]

    if aggr_buys:
        aggr_buy_mean = np.mean(aggr_buys)
        aggr_buy_std = np.std(aggr_buys) if len(aggr_buys) > 1 else 1.0
        features['aggr_buy_ma'] = aggr_buy_mean
        features['aggr_buy_norm'] = (
            (bar.aggr_buy_speed - aggr_buy_mean) / (aggr_buy_std + 1e-8)
        )
    else:
        features['aggr_buy_ma'] = 0.0
        features['aggr_buy_norm'] = 0.0

    if aggr_sells:
        aggr_sell_mean = np.mean(aggr_sells)
        aggr_sell_std = np.std(aggr_sells) if len(aggr_sells) > 1 else 1.0
        features['aggr_sell_ma'] = aggr_sell_mean
        features['aggr_sell_norm'] = (
            (bar.aggr_sell_speed - aggr_sell_mean) / (aggr_sell_std + 1e-8)
        )
    else:
        features['aggr_sell_ma'] = 0.0
        features['aggr_sell_norm'] = 0.0

    # Net aggressive buying pressure (buy - sell)
    features['aggr_net'] = bar.aggr_buy_speed - bar.aggr_sell_speed

    # Aggressive buying percentage
    total_aggr = bar.aggr_buy_speed + bar.aggr_sell_speed
    if total_aggr > 0:
        features['aggr_buy_pct'] = bar.aggr_buy_speed / total_aggr
    else:
        features['aggr_buy_pct'] = 0.5

    # --- Price Speed Features ---

    # Price speed (intrabar volatility)
    price_speeds = [b.price_speed for b in window_bars if b.price_speed > 0]
    if price_speeds:
        price_speed_mean = np.mean(price_speeds)
        price_speed_std = np.std(price_speeds) if len(price_speeds) > 1 else 1.0

        features['price_speed_raw'] = bar.price_speed
        features['price_speed_ma'] = price_speed_mean
        features['price_speed_norm'] = (
            (bar.price_speed - price_speed_mean) / (price_speed_std + 1e-8)
        )
    else:
        features['price_speed_raw'] = 0.0
        features['price_speed_ma'] = 0.0
        features['price_speed_norm'] = 0.0

    # Price speed per tick (efficiency of price movement)
    if bar.tick_speed > 0:
        features['price_per_tick'] = bar.price_speed / bar.tick_speed
    else:
        features['price_per_tick'] = 0.0

    # --- Delta Features (from volume delta) ---

    # Delta normalized by volume
    if bar.volume > 0:
        features['delta_norm'] = bar.delta / bar.volume
    else:
        features['delta_norm'] = 0.0

    # Delta intensity (abs delta / volume)
    if bar.volume > 0:
        features['delta_intensity'] = abs(bar.delta) / bar.volume
    else:
        features['delta_intensity'] = 0.0

    # Cumulative delta over window
    deltas = [b.delta for b in window_bars]
    features['cumulative_delta'] = sum(deltas)

    # Delta acceleration
    if i > 0:
        prev_delta = bars_list[i-1].delta
        features['delta_acceleration'] = bar.delta - prev_delta
    else:
        features['delta_acceleration'] = 0.0

    # --- Volume-Weighted Features ---

    # Average volume in window
    volumes = [b.volume for b in window_bars if b.volume > 0]
    if volumes:
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes) if len(volumes) > 1 else 1.0

        features['volume_ma'] = volume_mean
        features['volume_norm'] = (bar.volume - volume_mean) / (volume_std + 1e-8)
    else:
        features['volume_ma'] = 0.0
        features['volume_norm'] = 0.0

    # Volume acceleration
    if i > 0:
        prev_volume = bars_list[i-1].volume
        features['volume_acceleration'] = bar.volume - prev_volume
    else:
        features['volume_acceleration'] = 0.0

    # --- Composite Features ---

    # Buying pressure index (combines delta, buy/sell ratio, and aggr buy)
    # Formula: (delta_norm + (buy_sell_ratio - 1) / 5 + aggr_buy_norm / 3) / 3
    buying_pressure = (
        features['delta_norm'] +
        (features['buy_sell_ratio'] - 1.0) / 5.0 +
        features['aggr_buy_norm'] / 3.0
    ) / 3.0
    features['buying_pressure_index'] = buying_pressure

    # Activity intensity (combines tick speed, volume, and price speed)
    activity = (
        features['tick_speed_norm'] +
        features['volume_norm'] +
        features['price_speed_norm']
    ) / 3.0
    features['activity_intensity'] = activity

    return features


def get_tick_feature_names() -> List[str]:
    """
    Get list of all tick feature names in consistent order

    Returns:
        List of feature names
    """
    return [
        # Basic tick features
        'tick_speed_raw',
        'tick_speed_ma',
        'tick_speed_norm',
        'tick_acceleration',

        # Aggressive buy/sell features
        'buy_sell_ratio',
        'aggr_buy_ma',
        'aggr_buy_norm',
        'aggr_sell_ma',
        'aggr_sell_norm',
        'aggr_net',
        'aggr_buy_pct',

        # Price speed features
        'price_speed_raw',
        'price_speed_ma',
        'price_speed_norm',
        'price_per_tick',

        # Delta features
        'delta_norm',
        'delta_intensity',
        'cumulative_delta',
        'delta_acceleration',

        # Volume features
        'volume_ma',
        'volume_norm',
        'volume_acceleration',

        # Composite features
        'buying_pressure_index',
        'activity_intensity',
    ]


def extract_tick_features_batch(
    bars_list: List[RawBar],
    lookback: int = 14
) -> List[Dict[str, float]]:
    """
    Extract tick features for all bars in batch

    Args:
        bars_list: List of all RawBar objects
        lookback: Number of bars to use for moving averages

    Returns:
        List of feature dictionaries (one per bar)
    """
    features_list = []

    for i in range(len(bars_list)):
        features = extract_tick_features(i, bars_list, lookback)
        features_list.append(features)

    return features_list


if __name__ == "__main__":
    # Test tick analyzer
    from ..core.data_loader import load_raw_bars

    print("Testing Tick Analyzer...")

    # Load sample data
    jsonl_path = "/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl"
    bars = load_raw_bars(jsonl_path)

    print(f"Loaded {len(bars)} bars\n")

    # Extract tick features for first 10 bars
    print("Extracting tick features for first 10 bars...\n")

    for i in range(min(10, len(bars))):
        features = extract_tick_features(i, bars, lookback=5)

        print(f"Bar {i+1} at {bars[i].ts}:")
        print(f"  Tick Speed:     {features['tick_speed_raw']:.0f} (MA: {features['tick_speed_ma']:.1f}, Norm: {features['tick_speed_norm']:.2f})")
        print(f"  Buy/Sell Ratio: {features['buy_sell_ratio']:.2f}")
        print(f"  Delta Norm:     {features['delta_norm']:.3f}")
        print(f"  Buying Pressure:{features['buying_pressure_index']:.3f}")
        print(f"  Activity:       {features['activity_intensity']:.3f}")
        print()

    # Get feature names
    feature_names = get_tick_feature_names()
    print(f"Total tick features: {len(feature_names)}")
    print(f"Feature names: {', '.join(feature_names[:10])}...")

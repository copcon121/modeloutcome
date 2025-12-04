"""
Feature specification for STATE-ENC v1

Định nghĩa danh sách features, thứ tự, và metadata cho encoding.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"


@dataclass
class FeatureInfo:
    """Metadata cho một feature"""
    name: str
    ftype: FeatureType
    default: float = 0.0
    num_categories: int = 0  # Chỉ dùng cho categorical
    description: str = ""


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Group 1: Core OHLCV & Shape
OHLCV_FEATURES: List[FeatureInfo] = [
    FeatureInfo("o", FeatureType.NUMERIC, 0.0, 0, "Open price (normalized)"),
    FeatureInfo("h", FeatureType.NUMERIC, 0.0, 0, "High price (normalized)"),
    FeatureInfo("l", FeatureType.NUMERIC, 0.0, 0, "Low price (normalized)"),
    FeatureInfo("c", FeatureType.NUMERIC, 0.0, 0, "Close price (normalized)"),
    FeatureInfo("hl_range", FeatureType.NUMERIC, 0.0, 0, "High - Low"),
    FeatureInfo("body", FeatureType.NUMERIC, 0.0, 0, "Close - Open"),
    FeatureInfo("upper_wick", FeatureType.NUMERIC, 0.0, 0, "High - max(O,C)"),
    FeatureInfo("lower_wick", FeatureType.NUMERIC, 0.0, 0, "min(O,C) - Low"),
    FeatureInfo("bar_type", FeatureType.CATEGORICAL, 0.0, 3, "0=doji, 1=bull, 2=bear"),
    FeatureInfo("volume", FeatureType.NUMERIC, 0.0, 0, "Volume"),
    FeatureInfo("volume_vs_session_avg", FeatureType.NUMERIC, 1.0, 0, "Volume / session avg"),
    FeatureInfo("volume_zscore", FeatureType.NUMERIC, 0.0, 0, "Volume z-score"),
    FeatureInfo("atr_m1_14", FeatureType.NUMERIC, 0.0, 0, "ATR 14 period M1"),
    FeatureInfo("true_range", FeatureType.NUMERIC, 0.0, 0, "True range"),
    FeatureInfo("range_vs_session_avg", FeatureType.NUMERIC, 1.0, 0, "Range / session avg"),
]

# Group 2: Delta & Tick Microstructure
DELTA_TICK_FEATURES: List[FeatureInfo] = [
    FeatureInfo("delta", FeatureType.NUMERIC, 0.0, 0, "Delta (buy - sell volume)"),
    FeatureInfo("delta_abs", FeatureType.NUMERIC, 0.0, 0, "Absolute delta"),
    FeatureInfo("delta_sign", FeatureType.CATEGORICAL, 0.0, 3, "-1, 0, +1"),
    FeatureInfo("delta_vs_volume", FeatureType.NUMERIC, 0.0, 0, "Delta / volume"),
    FeatureInfo("cum_delta_session", FeatureType.NUMERIC, 0.0, 0, "Cumulative delta in session"),
    FeatureInfo("delta_zscore_session", FeatureType.NUMERIC, 0.0, 0, "Delta z-score"),
    FeatureInfo("tick_count", FeatureType.NUMERIC, 0.0, 0, "Number of ticks"),
    FeatureInfo("tick_speed", FeatureType.NUMERIC, 0.0, 0, "Ticks per second"),
    FeatureInfo("buy_volume", FeatureType.NUMERIC, 0.0, 0, "Buy volume"),
    FeatureInfo("sell_volume", FeatureType.NUMERIC, 0.0, 0, "Sell volume"),
    FeatureInfo("buy_ratio", FeatureType.NUMERIC, 0.5, 0, "Buy / total volume"),
    FeatureInfo("sell_ratio", FeatureType.NUMERIC, 0.5, 0, "Sell / total volume"),
    FeatureInfo("imbalance_buy_sell", FeatureType.NUMERIC, 0.0, 0, "(Buy - Sell) / volume"),
]

# Group 3: SMC Structure
SMC_FEATURES: List[FeatureInfo] = [
    # Trend
    FeatureInfo("ext_trend_dir", FeatureType.CATEGORICAL, 0.0, 3, "-1, 0, +1"),
    FeatureInfo("int_trend_dir", FeatureType.CATEGORICAL, 0.0, 3, "-1, 0, +1"),
    # External BOS/CHoCH
    FeatureInfo("ext_bos_up", FeatureType.BINARY, 0.0, 2, "External BOS up"),
    FeatureInfo("ext_bos_down", FeatureType.BINARY, 0.0, 2, "External BOS down"),
    FeatureInfo("ext_choch_up", FeatureType.BINARY, 0.0, 2, "External CHoCH up"),
    FeatureInfo("ext_choch_down", FeatureType.BINARY, 0.0, 2, "External CHoCH down"),
    FeatureInfo("bars_since_last_ext_bos", FeatureType.NUMERIC, 999.0, 0, "Bars since ext BOS"),
    FeatureInfo("bars_since_last_ext_choch", FeatureType.NUMERIC, 999.0, 0, "Bars since ext CHoCH"),
    # Internal BOS/CHoCH
    FeatureInfo("int_bos_up", FeatureType.BINARY, 0.0, 2, "Internal BOS up"),
    FeatureInfo("int_bos_down", FeatureType.BINARY, 0.0, 2, "Internal BOS down"),
    FeatureInfo("int_choch_up", FeatureType.BINARY, 0.0, 2, "Internal CHoCH up"),
    FeatureInfo("int_choch_down", FeatureType.BINARY, 0.0, 2, "Internal CHoCH down"),
    FeatureInfo("bars_since_last_int_bos", FeatureType.NUMERIC, 999.0, 0, "Bars since int BOS"),
    FeatureInfo("bars_since_last_int_choch", FeatureType.NUMERIC, 999.0, 0, "Bars since int CHoCH"),
    # Swing & Premium/Discount
    FeatureInfo("swing_high", FeatureType.NUMERIC, 0.0, 0, "Current swing high"),
    FeatureInfo("swing_low", FeatureType.NUMERIC, 0.0, 0, "Current swing low"),
    FeatureInfo("price_vs_swing_mid", FeatureType.NUMERIC, 0.0, 0, "Price position in swing range"),
    FeatureInfo("premium_zone", FeatureType.BINARY, 0.0, 2, "In premium zone"),
    FeatureInfo("discount_zone", FeatureType.BINARY, 0.0, 2, "In discount zone"),
    FeatureInfo("distance_to_swing_high", FeatureType.NUMERIC, 0.0, 0, "Distance to swing high (ticks)"),
    FeatureInfo("distance_to_swing_low", FeatureType.NUMERIC, 0.0, 0, "Distance to swing low (ticks)"),
    FeatureInfo("distance_to_swing_high_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized dist to swing high"),
    FeatureInfo("distance_to_swing_low_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized dist to swing low"),
    # Liquidity & Sweep
    FeatureInfo("sweep_prev_high", FeatureType.BINARY, 0.0, 2, "Swept previous high"),
    FeatureInfo("sweep_prev_low", FeatureType.BINARY, 0.0, 2, "Swept previous low"),
    FeatureInfo("sweep_type", FeatureType.CATEGORICAL, 0.0, 4, "0=none, 1=high, 2=low, 3=both"),
    FeatureInfo("bars_since_last_sweep", FeatureType.NUMERIC, 999.0, 0, "Bars since last sweep"),
    # OB Proximity
    FeatureInfo("near_ob_m1_bull", FeatureType.BINARY, 0.0, 2, "Near M1 bullish OB"),
    FeatureInfo("near_ob_m1_bear", FeatureType.BINARY, 0.0, 2, "Near M1 bearish OB"),
    FeatureInfo("near_ob_m5_bull", FeatureType.BINARY, 0.0, 2, "Near M5 bullish OB"),
    FeatureInfo("near_ob_m5_bear", FeatureType.BINARY, 0.0, 2, "Near M5 bearish OB"),
    FeatureInfo("ob_age_bars", FeatureType.NUMERIC, 0.0, 0, "OB age in bars"),
    FeatureInfo("distance_to_nearest_ob", FeatureType.NUMERIC, 0.0, 0, "Distance to nearest OB (signed)"),
    # FVG Proximity
    FeatureInfo("near_fvg_m1_bull", FeatureType.BINARY, 0.0, 2, "Near M1 bullish FVG"),
    FeatureInfo("near_fvg_m1_bear", FeatureType.BINARY, 0.0, 2, "Near M1 bearish FVG"),
    FeatureInfo("near_fvg_m5_bull", FeatureType.BINARY, 0.0, 2, "Near M5 bullish FVG"),
    FeatureInfo("near_fvg_m5_bear", FeatureType.BINARY, 0.0, 2, "Near M5 bearish FVG"),
    FeatureInfo("fvg_age_bars", FeatureType.NUMERIC, 0.0, 0, "FVG age in bars"),
    FeatureInfo("distance_to_nearest_fvg", FeatureType.NUMERIC, 0.0, 0, "Distance to nearest FVG (signed)"),
]

# Group 4: VA / Auction Features
VA_FEATURES: List[FeatureInfo] = [
    FeatureInfo("vah", FeatureType.NUMERIC, 0.0, 0, "Value Area High"),
    FeatureInfo("val", FeatureType.NUMERIC, 0.0, 0, "Value Area Low"),
    FeatureInfo("poc", FeatureType.NUMERIC, 0.0, 0, "Point of Control"),
    FeatureInfo("dist_to_vah", FeatureType.NUMERIC, 0.0, 0, "Distance to VAH (ticks)"),
    FeatureInfo("dist_to_val", FeatureType.NUMERIC, 0.0, 0, "Distance to VAL (ticks)"),
    FeatureInfo("dist_to_poc", FeatureType.NUMERIC, 0.0, 0, "Distance to POC (ticks)"),
    FeatureInfo("inside_value", FeatureType.BINARY, 0.0, 2, "Inside value area"),
    FeatureInfo("above_value", FeatureType.BINARY, 0.0, 2, "Above value area"),
    FeatureInfo("below_value", FeatureType.BINARY, 0.0, 2, "Below value area"),
    FeatureInfo("session_high", FeatureType.NUMERIC, 0.0, 0, "Session high"),
    FeatureInfo("session_low", FeatureType.NUMERIC, 0.0, 0, "Session low"),
    FeatureInfo("pos_in_session_range", FeatureType.NUMERIC, 0.5, 0, "Position in session range"),
    FeatureInfo("dist_to_session_high_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized dist to session high"),
    FeatureInfo("dist_to_session_low_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized dist to session low"),
]

# Group 5: Session/Time Features
TIME_FEATURES: List[FeatureInfo] = [
    FeatureInfo("session_id", FeatureType.CATEGORICAL, 0.0, 3, "0=ASIA, 1=LDN, 2=NY"),
    FeatureInfo("bar_index_in_session", FeatureType.NUMERIC, 0.0, 0, "Bar index in session"),
    FeatureInfo("bar_index_in_session_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized bar index"),
    FeatureInfo("minute_of_day", FeatureType.NUMERIC, 0.0, 0, "Minute of day (0-1439)"),
    FeatureInfo("minute_of_day_norm", FeatureType.NUMERIC, 0.0, 0, "Normalized minute of day"),
    FeatureInfo("day_of_week", FeatureType.CATEGORICAL, 0.0, 7, "Day of week (0-6)"),
]

# Group 6: Regime Hint (ASM v1)
REGIME_FEATURES: List[FeatureInfo] = [
    FeatureInfo("asm_regime_hint", FeatureType.CATEGORICAL, 0.0, 6, 
                "0=unknown, 1=range, 2=trend_up, 3=trend_down, 4=opening_drive_up, 5=opening_drive_down"),
]

# =============================================================================
# COMBINED FEATURE SPEC
# =============================================================================

ALL_FEATURE_GROUPS = [
    ("ohlcv", OHLCV_FEATURES),
    ("delta_tick", DELTA_TICK_FEATURES),
    ("smc", SMC_FEATURES),
    ("va", VA_FEATURES),
    ("time", TIME_FEATURES),
    ("regime", REGIME_FEATURES),
]

# Flat list of all features in order
FEATURE_SPEC: List[FeatureInfo] = []
for group_name, features in ALL_FEATURE_GROUPS:
    FEATURE_SPEC.extend(features)

# Feature name to index mapping
FEATURE_NAME_TO_IDX: Dict[str, int] = {f.name: i for i, f in enumerate(FEATURE_SPEC)}

# Feature name to info mapping
FEATURE_NAME_TO_INFO: Dict[str, FeatureInfo] = {f.name: f for f in FEATURE_SPEC}

# Lists by type
NUMERIC_FEATURES: List[str] = [f.name for f in FEATURE_SPEC if f.ftype == FeatureType.NUMERIC]
CATEGORICAL_FEATURES: List[str] = [f.name for f in FEATURE_SPEC if f.ftype == FeatureType.CATEGORICAL]
BINARY_FEATURES: List[str] = [f.name for f in FEATURE_SPEC if f.ftype == FeatureType.BINARY]

# Total feature dimension
TOTAL_FEATURE_DIM: int = len(FEATURE_SPEC)


def get_feature_names() -> List[str]:
    """Return ordered list of feature names"""
    return [f.name for f in FEATURE_SPEC]


def get_feature_defaults() -> Dict[str, float]:
    """Return dict of feature name -> default value"""
    return {f.name: f.default for f in FEATURE_SPEC}


def get_feature_index(name: str) -> int:
    """Get index of feature by name"""
    return FEATURE_NAME_TO_IDX.get(name, -1)


def get_feature_info(name: str) -> FeatureInfo:
    """Get FeatureInfo by name"""
    return FEATURE_NAME_TO_INFO.get(name)


def validate_bar_features(bar: Dict) -> Tuple[bool, List[str]]:
    """
    Validate bar dict has required features.
    Returns (is_valid, list_of_missing_features)
    """
    missing = []
    for f in FEATURE_SPEC:
        if f.name not in bar:
            missing.append(f.name)
    return len(missing) == 0, missing


# Print summary when module loaded
if __name__ == "__main__":
    print(f"Total features: {TOTAL_FEATURE_DIM}")
    print(f"  Numeric: {len(NUMERIC_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  Binary: {len(BINARY_FEATURES)}")
    print("\nFeature groups:")
    for group_name, features in ALL_FEATURE_GROUPS:
        print(f"  {group_name}: {len(features)} features")

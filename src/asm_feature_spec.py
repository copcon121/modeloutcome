#!/usr/bin/env python3
"""
ASM Feature Specification v1.1
==============================
Single source of truth for ASM model feature order.

This module defines the exact 100 features used by ASM-GRU64-v1.0-C3 model.
All ASM-related scripts MUST import from here to ensure consistency.

Feature order extracted from: output/asm_dataset_v1/asm_dataset_v1_stats.json
Training data built by: scripts/build_asm_dataset_v1.py

IMPORTANT:
- Do NOT modify this list without retraining the model
- All scripts (training, inference, backtest, live) must use this exact order
"""

# ASM Model Configuration
ASM_SEQ_LEN = 60        # Context window length (bars)
ASM_FEATURE_DIM = 100   # Number of features per bar

# Fixed feature order - matches training data exactly
# Source: output/asm_dataset_v1/asm_dataset_v1_stats.json["features"]["feature_names"][:100]
ASM_FEATURE_COLS = [
    # Bar OHLCV features (0-16)
    "close",                    # 0 - IMPORTANT: close is included!
    "high_low_range",           # 1
    "body",                     # 2
    "upper_wick",               # 3
    "lower_wick",               # 4
    "close_return",             # 5
    "volume",                   # 6
    "volume_change",            # 7
    "delta",                    # 8
    "delta_over_volume",        # 9
    "buy_volume",               # 10
    "sell_volume",              # 11
    "buy_sell_ratio",           # 12
    "tick_speed",               # 13
    "aggr_buy_speed",           # 14
    "aggr_sell_speed",          # 15
    "price_speed",              # 16
    
    # Internal structure features (17-28)
    "int_trend_dir",            # 17
    "int_bos_up",               # 18
    "int_bos_down",             # 19
    "int_choch_up",             # 20
    "int_choch_down",           # 21
    "int_swing_high_distance",  # 22
    "int_swing_low_distance",   # 23
    "bars_since_int_swing_high",# 24
    "bars_since_int_swing_low", # 25
    "swept_prev_int_high",      # 26
    "swept_prev_int_low",       # 27
    "int_bias_bullish",         # 28
    
    # External structure features (29-40)
    "ext_trend_dir",            # 29
    "ext_bos_up",               # 30
    "ext_bos_down",             # 31
    "ext_choch_up",             # 32
    "ext_choch_down",           # 33
    "ext_swing_high_distance",  # 34
    "ext_swing_low_distance",   # 35
    "bars_since_ext_swing_high",# 36
    "bars_since_ext_swing_low", # 37
    "swept_prev_ext_high",      # 38
    "swept_prev_ext_low",       # 39
    "ext_bias_bullish",         # 40
    
    # FVG features (41-55)
    "in_bull_fvg",              # 41
    "in_bear_fvg",              # 42
    "near_bull_fvg",            # 43
    "near_bear_fvg",            # 44
    "int_in_bull_ob",           # 45
    "int_in_bear_ob",           # 46
    "int_near_bull_ob",         # 47
    "int_near_bear_ob",         # 48
    "ext_in_bull_ob",           # 49
    "ext_in_bear_ob",           # 50
    "ext_near_bull_ob",         # 51
    "ext_near_bear_ob",         # 52
    "dist_to_nearest_fvg",      # 53
    "dist_to_nearest_ob",       # 54
    "nearest_fvg_size",         # 55
    
    # Volume Profile features (56-65)
    "vp_poc_price",             # 56
    "vp_val_price",             # 57
    "vp_vah_price",             # 58
    "vp_in_value_area",         # 59
    "vp_above_value_area",      # 60
    "vp_below_value_area",      # 61
    "vp_dist_to_poc",           # 62
    "vp_dist_to_vah",           # 63
    "vp_dist_to_val",           # 64
    "impulse_strength",         # 65
    
    # Momentum features (66-71)
    "pullback_strength",        # 66
    "cum_delta_5",              # 67
    "cum_delta_10",             # 68
    "cum_delta_20",             # 69
    "vwap_daily",               # 70
    "dist_to_vwap",             # 71
    
    # M5 HTF features (72-78)
    "m5_trend_up",              # 72
    "m5_trend_down",            # 73
    "m5_premium",               # 74
    "m5_discount",              # 75
    "dist_to_m5_swing_high",    # 76
    "dist_to_m5_swing_low",     # 77
    "near_m5_fvg",              # 78
    
    # H1 HTF features (79-85)
    "h1_trend_up",              # 79
    "h1_trend_down",            # 80
    "h1_premium",               # 81
    "h1_discount",              # 82
    "dist_to_h1_swing_high",    # 83
    "dist_to_h1_swing_low",     # 84
    "near_h1_fvg",              # 85
    
    # Raw OHLC (86-88)
    "open",                     # 86
    "high",                     # 87
    "low",                      # 88
    
    # M5 derived features (89-93)
    "m5_swing_phase",           # 89
    "m5_price_pos_in_range",    # 90
    "m5_bos_up_count_recent",   # 91
    "m5_bos_down_count_recent", # 92
    "m5_ob_imbalance",          # 93
    
    # H1 derived features (94-98)
    "h1_swing_phase",           # 94
    "h1_price_pos_in_range",    # 95
    "h1_bos_up_count_recent",   # 96
    "h1_bos_down_count_recent", # 97
    "h1_ob_imbalance",          # 98
    
    # Global index (99)
    "global_bar_index",         # 99
]

# Validation
assert len(ASM_FEATURE_COLS) == ASM_FEATURE_DIM, \
    f"ASM_FEATURE_COLS has {len(ASM_FEATURE_COLS)} features, expected {ASM_FEATURE_DIM}"


def validate_feature_order(feature_dict: dict) -> bool:
    """
    Validate that a feature dict contains all required ASM features.
    
    Args:
        feature_dict: Dictionary with feature names as keys
        
    Returns:
        True if all features present, raises ValueError otherwise
    """
    missing = [f for f in ASM_FEATURE_COLS if f not in feature_dict]
    if missing:
        raise ValueError(f"Missing ASM features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return True


def build_feature_vector(feature_dict: dict) -> list:
    """
    Build ASM feature vector from a feature dictionary.
    
    Args:
        feature_dict: Dictionary with feature names as keys
        
    Returns:
        List of 100 feature values in correct order
    """
    return [float(feature_dict.get(col, 0.0)) for col in ASM_FEATURE_COLS]


if __name__ == "__main__":
    print(f"ASM Feature Spec v1.1")
    print(f"  Sequence length: {ASM_SEQ_LEN}")
    print(f"  Feature dimension: {ASM_FEATURE_DIM}")
    print(f"  Total features defined: {len(ASM_FEATURE_COLS)}")
    print(f"\nFirst 10 features:")
    for i, f in enumerate(ASM_FEATURE_COLS[:10]):
        print(f"  {i:3d}: {f}")
    print(f"\nLast 5 features:")
    for i, f in enumerate(ASM_FEATURE_COLS[-5:], ASM_FEATURE_DIM - 5):
        print(f"  {i:3d}: {f}")

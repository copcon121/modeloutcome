"""
P2_v2 Filter - FAST VECTORIZED VERSION
Mode 1: VA Reversal, Mode 2: Trend Continuation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent


@dataclass
class DeltaConfig:
    name: str
    delta_spike1: float
    vol_spike1: float
    delta_spike2: float
    delta_long_entry: float
    vol_spike2: float
    delta_pb_max: float


CONFIG_A = DeltaConfig("A_Conservative", 2.0, 2.0, 1.5, 1.0, 1.5, 1.0)
CONFIG_B = DeltaConfig("B_Medium", 1.5, 1.5, 1.0, 0.8, 1.2, 1.2)
CONFIG_C = DeltaConfig("C_Loose", 1.2, 1.2, 0.8, 0.5, 1.0, 1.5)


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta features - vectorized"""
    df = df.copy()
    N = 20
    
    # Zscore
    df['delta_zscore'] = (df['delta'] - df['delta'].rolling(N, min_periods=5).mean()) / \
                         (df['delta'].rolling(N, min_periods=5).std() + 1e-8)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(N, min_periods=5).mean()) / \
                          (df['volume'].rolling(N, min_periods=5).std() + 1e-8)
    
    # Candle type
    df['is_bull'] = (df['close'] > df['open']).astype(int)
    df['is_bear'] = (df['close'] < df['open']).astype(int)
    
    # VA cross
    df['cross_val_down'] = ((df['vp_in_value_area'].shift(1) == 1) & 
                            (df['vp_below_value_area'] == 1)).astype(int)
    df['cross_vah_up'] = ((df['vp_in_value_area'].shift(1) == 1) & 
                          (df['vp_above_value_area'] == 1)).astype(int)
    df['reenter_va_below'] = ((df['vp_below_value_area'].shift(1) == 1) & 
                              (df['vp_in_value_area'] == 1)).astype(int)
    df['reenter_va_above'] = ((df['vp_above_value_area'].shift(1) == 1) & 
                              (df['vp_in_value_area'] == 1)).astype(int)
    
    return df


def apply_p2v2_filter(df: pd.DataFrame, cfg: DeltaConfig) -> pd.DataFrame:
    """Apply P2_v2 filter - simplified but faster"""
    df = add_delta_features(df)
    n = len(df)
    
    # Pre-compute rolling conditions
    N_BREAK = 15
    N_ENTRY = 10
    N_TREND = 30
    
    # Hunt bar detection (rolling max of hunt conditions)
    hunt_long_cond = (
        ((df['cross_val_down'] == 1) | (df['swept_prev_int_low'] == 1) | (df['int_choch_down'] == 1)) &
        (df['volume_zscore'] >= cfg.vol_spike1) &
        (df['delta_zscore'] <= -cfg.delta_spike1)
    ).astype(int)
    
    hunt_short_cond = (
        ((df['cross_vah_up'] == 1) | (df['swept_prev_int_high'] == 1) | (df['int_choch_up'] == 1)) &
        (df['volume_zscore'] >= cfg.vol_spike1) &
        (df['delta_zscore'] >= cfg.delta_spike1)
    ).astype(int)
    
    # Rolling sum to detect if hunt happened in last N bars
    df['hunt_long_recent'] = hunt_long_cond.rolling(N_BREAK, min_periods=1).sum()
    df['hunt_short_recent'] = hunt_short_cond.rolling(N_BREAK, min_periods=1).sum()
    
    # Strong spike detection (for exhaustion check)
    df['strong_sell_spike'] = (df['delta_zscore'] <= -cfg.delta_spike2).astype(int)
    df['strong_buy_spike'] = (df['delta_zscore'] >= cfg.delta_spike2).astype(int)
    df['recent_sell_spike'] = df['strong_sell_spike'].rolling(N_ENTRY, min_periods=1).sum()
    df['recent_buy_spike'] = df['strong_buy_spike'].rolling(N_ENTRY, min_periods=1).sum()
    
    # Recent BOS detection
    df['recent_ext_bos_down'] = df['ext_bos_down'].rolling(N_TREND, min_periods=1).sum()
    df['recent_ext_bos_up'] = df['ext_bos_up'].rolling(N_TREND, min_periods=1).sum()
    
    # Pullback delta check (max negative/positive in last 5 bars)
    df['pb_delta_min'] = df['delta_zscore'].rolling(5, min_periods=1).min()
    df['pb_delta_max'] = df['delta_zscore'].rolling(5, min_periods=1).max()
    
    # === MODE 1 LONG ===
    m1_long = (
        (df['m5_trend_up'] == 1) &
        (df['hunt_long_recent'] >= 1) &
        (df['recent_sell_spike'] <= 1) &  # Exhaustion: no new strong spike
        (df['is_bull'] == 1) &
        (df['volume_zscore'] >= cfg.vol_spike2) &
        (df['delta_zscore'] >= cfg.delta_long_entry) &
        ((df['reenter_va_below'] == 1) | (df['vp_in_value_area'] == 1))
    )
    
    # === MODE 1 SHORT ===
    m1_short = (
        (df['m5_trend_down'] == 1) &
        (df['hunt_short_recent'] >= 1) &
        (df['recent_buy_spike'] <= 1) &
        (df['is_bear'] == 1) &
        (df['volume_zscore'] >= cfg.vol_spike2) &
        (df['delta_zscore'] <= -cfg.delta_long_entry) &
        ((df['reenter_va_above'] == 1) | (df['vp_in_value_area'] == 1))
    )
    
    # === MODE 2 LONG ===
    m2_long = (
        ((df['m5_trend_up'] == 1) | (df['ext_trend_dir'] > 0)) &
        (df['recent_ext_bos_down'] == 0) &
        ((df['in_bull_fvg'] == 1) | (df['int_in_bull_ob'] == 1) | (df['near_bull_fvg'] == 1)) &
        (df['pb_delta_min'] >= -cfg.delta_pb_max) &
        (df['is_bull'] == 1) &
        (df['delta_zscore'] >= cfg.delta_long_entry * 0.5)
    )
    
    # === MODE 2 SHORT ===
    m2_short = (
        ((df['m5_trend_down'] == 1) | (df['ext_trend_dir'] < 0)) &
        (df['recent_ext_bos_up'] == 0) &
        ((df['in_bear_fvg'] == 1) | (df['int_in_bear_ob'] == 1) | (df['near_bear_fvg'] == 1)) &
        (df['pb_delta_max'] <= cfg.delta_pb_max) &
        (df['is_bear'] == 1) &
        (df['delta_zscore'] <= -cfg.delta_long_entry * 0.5)
    )
    
    # Assign (priority: M1 > M2)
    df['p2_mask'] = 0
    df['p2_mode'] = 0
    df['p2_side'] = 0
    
    df.loc[m2_short, ['p2_mask', 'p2_mode', 'p2_side']] = [1, 2, -1]
    df.loc[m2_long, ['p2_mask', 'p2_mode', 'p2_side']] = [1, 2, 1]
    df.loc[m1_short, ['p2_mask', 'p2_mode', 'p2_side']] = [1, 1, -1]
    df.loc[m1_long, ['p2_mask', 'p2_mode', 'p2_side']] = [1, 1, 1]
    
    return df


def main():
    print("="*70)
    print("P2_v2 FILTER - FAST VERSION")
    print("="*70)
    
    csv_path = ROOT / "output/production_10weeks_v3/features_all_10weeks_v3.csv"
    print(f"\n[1] Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Total bars: {len(df)}")
    
    results = []
    
    for cfg in [CONFIG_A, CONFIG_B, CONFIG_C]:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg.name}")
        print(f"{'='*70}")
        
        df_out = apply_p2v2_filter(df.copy(), cfg)
        
        n_p2 = (df_out['p2_mask'] == 1).sum()
        n_m1_long = ((df_out['p2_mode'] == 1) & (df_out['p2_side'] == 1)).sum()
        n_m1_short = ((df_out['p2_mode'] == 1) & (df_out['p2_side'] == -1)).sum()
        n_m2_long = ((df_out['p2_mode'] == 2) & (df_out['p2_side'] == 1)).sum()
        n_m2_short = ((df_out['p2_mode'] == 2) & (df_out['p2_side'] == -1)).sum()
        
        print(f"\n[STATS]")
        print(f"  Total P2 setups: {n_p2} ({n_p2/len(df)*100:.2f}%)")
        print(f"  Mode 1 Long:  {n_m1_long}")
        print(f"  Mode 1 Short: {n_m1_short}")
        print(f"  Mode 2 Long:  {n_m2_long}")
        print(f"  Mode 2 Short: {n_m2_short}")
        
        results.append({
            'config': cfg.name,
            'total': n_p2,
            'm1_long': n_m1_long,
            'm1_short': n_m1_short,
            'm2_long': n_m2_long,
            'm2_short': n_m2_short
        })
        
        # Save filtered data
        out_path = ROOT / f"output/p2_v2_{cfg.name}.csv"
        df_p2 = df_out[df_out['p2_mask'] == 1].copy()
        df_p2.to_csv(out_path, index=False)
        print(f"  Saved {len(df_p2)} rows to {out_path.name}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r['config']}: {r['total']} setups (M1L:{r['m1_long']}, M1S:{r['m1_short']}, M2L:{r['m2_long']}, M2S:{r['m2_short']})")


if __name__ == "__main__":
    main()

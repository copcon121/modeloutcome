"""
P2_v2 Filter - 2 Modes with Delta Integration
Mode 1: VA Reversal (hunt + exhaustion + reversal)
Mode 2: Trend Continuation (FVG/OB pullback)

Output: p2_mask, p2_mode, p2_side
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class DeltaConfig:
    """Delta thresholds configuration"""
    name: str
    # Hunt bar thresholds
    delta_spike1: float      # Hunt bar delta zscore threshold
    vol_spike1: float        # Hunt bar volume zscore threshold
    # Exhaustion thresholds
    delta_spike2: float      # No new spike stronger than this
    # Entry thresholds
    delta_long_entry: float  # Entry bar delta zscore for long
    vol_spike2: float        # Entry bar volume zscore
    # Pullback thresholds (Mode 2)
    delta_pb_max: float      # Max negative delta in pullback


# 3 Configs to test
CONFIG_A = DeltaConfig(
    name="A_Conservative",
    delta_spike1=2.0, vol_spike1=2.0,
    delta_spike2=1.5,
    delta_long_entry=1.0, vol_spike2=1.5,
    delta_pb_max=1.0
)

CONFIG_B = DeltaConfig(
    name="B_Medium",
    delta_spike1=1.5, vol_spike1=1.5,
    delta_spike2=1.0,
    delta_long_entry=0.8, vol_spike2=1.2,
    delta_pb_max=1.2
)

CONFIG_C = DeltaConfig(
    name="C_Loose",
    delta_spike1=1.2, vol_spike1=1.2,
    delta_spike2=0.8,
    delta_long_entry=0.5, vol_spike2=1.0,
    delta_pb_max=1.5
)


class P2V2Filter:
    """P2_v2 Filter with 2 modes"""
    
    def __init__(self, config: DeltaConfig):
        self.config = config
        # Lookback windows
        self.N_ZSCORE = 20       # Rolling window for zscore
        self.N_BREAK = 15        # Lookback for hunt bar
        self.N_ENTRY = 10        # Max bars after hunt for entry
        self.N_EXHAUST = 4       # Bars to check exhaustion
        self.N_TREND = 30        # Lookback for trend check
        self.N_LEG = 20          # Lookback for impulse leg
        
    def add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delta-based features"""
        df = df.copy()
        
        # 1. Delta zscore (rolling 20 bars)
        delta_mean = df['delta'].rolling(self.N_ZSCORE, min_periods=5).mean()
        delta_std = df['delta'].rolling(self.N_ZSCORE, min_periods=5).std()
        df['delta_zscore'] = (df['delta'] - delta_mean) / (delta_std + 1e-8)
        
        # 2. Volume zscore
        vol_mean = df['volume'].rolling(self.N_ZSCORE, min_periods=5).mean()
        vol_std = df['volume'].rolling(self.N_ZSCORE, min_periods=5).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
        
        # 3. Delta slope (last 4 bars)
        df['delta_slope'] = df['delta'].rolling(4, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
        
        # 4. Cumulative delta (leg)
        df['cum_delta_5'] = df['delta'].rolling(5, min_periods=1).sum()
        df['cum_delta_10'] = df['delta'].rolling(10, min_periods=1).sum()
        
        # 5. Is bull candle
        df['is_bull_candle'] = (df['close'] > df['open']).astype(int)
        df['is_bear_candle'] = (df['close'] < df['open']).astype(int)
        
        # 6. VA cross detection
        df['cross_val_down'] = (
            (df['vp_in_value_area'].shift(1) == 1) & 
            (df['vp_below_value_area'] == 1)
        ).astype(int)
        
        df['cross_vah_up'] = (
            (df['vp_in_value_area'].shift(1) == 1) & 
            (df['vp_above_value_area'] == 1)
        ).astype(int)
        
        # 7. Re-enter VA
        df['reenter_va_from_below'] = (
            (df['vp_below_value_area'].shift(1) == 1) & 
            (df['vp_in_value_area'] == 1)
        ).astype(int)
        
        df['reenter_va_from_above'] = (
            (df['vp_above_value_area'].shift(1) == 1) & 
            (df['vp_in_value_area'] == 1)
        ).astype(int)
        
        return df
    
    def detect_mode1_long(self, df: pd.DataFrame, i: int) -> Tuple[bool, Optional[int]]:
        """
        Mode 1 Long: VA Reversal from below
        Returns: (is_valid, hunt_bar_index)
        """
        cfg = self.config
        
        # A) Regime filter: m5 trend up OR recent HL pattern
        m5_up = df.iloc[i].get('m5_trend_up', 0) == 1
        # Simplified: just check m5 trend
        if not m5_up:
            return False, None
        
        # B) Find hunt bar in last N_BREAK bars
        hunt_bar = None
        for j in range(max(0, i - self.N_BREAK), i):
            row_j = df.iloc[j]
            
            # Hunt conditions:
            # - Cross VAL from above (break down)
            # - Volume spike
            # - Delta negative spike (sell pressure)
            cross_val = row_j.get('cross_val_down', 0) == 1
            vol_spike = row_j.get('volume_zscore', 0) >= cfg.vol_spike1
            delta_spike = row_j.get('delta_zscore', 0) <= -cfg.delta_spike1
            
            # Optional: sweep or choch
            sweep = row_j.get('swept_prev_int_low', 0) == 1
            choch = row_j.get('int_choch_down', 0) == 1
            
            if cross_val and vol_spike and delta_spike:
                hunt_bar = j
                break
            elif (sweep or choch) and vol_spike and delta_spike:
                hunt_bar = j
                break
        
        if hunt_bar is None:
            return False, None
        
        # C) Exhaustion check: no stronger delta spike after hunt
        for k in range(hunt_bar + 1, min(hunt_bar + self.N_EXHAUST + 1, i)):
            if k >= len(df):
                break
            if df.iloc[k].get('delta_zscore', 0) <= -cfg.delta_spike2:
                # New strong sell spike = not exhausted
                return False, None
        
        # D) Entry bar conditions
        row_i = df.iloc[i]
        
        # Bull candle
        if row_i.get('is_bull_candle', 0) != 1:
            return False, None
        
        # Volume spike at entry
        if row_i.get('volume_zscore', 0) < cfg.vol_spike2:
            return False, None
        
        # Delta positive (buy pressure)
        if row_i.get('delta_zscore', 0) < cfg.delta_long_entry:
            return False, None
        
        # Re-enter VA or close > VAL
        reenter = row_i.get('reenter_va_from_below', 0) == 1
        in_va = row_i.get('vp_in_value_area', 0) == 1
        if not (reenter or in_va):
            return False, None
        
        # Not too far from hunt
        if i - hunt_bar > self.N_ENTRY:
            return False, None
        
        return True, hunt_bar
    
    def detect_mode1_short(self, df: pd.DataFrame, i: int) -> Tuple[bool, Optional[int]]:
        """
        Mode 1 Short: VA Reversal from above
        Mirror of long logic
        """
        cfg = self.config
        
        # A) Regime filter: m5 trend down
        m5_down = df.iloc[i].get('m5_trend_down', 0) == 1
        if not m5_down:
            return False, None
        
        # B) Find hunt bar
        hunt_bar = None
        for j in range(max(0, i - self.N_BREAK), i):
            row_j = df.iloc[j]
            
            cross_vah = row_j.get('cross_vah_up', 0) == 1
            vol_spike = row_j.get('volume_zscore', 0) >= cfg.vol_spike1
            delta_spike = row_j.get('delta_zscore', 0) >= cfg.delta_spike1  # Positive spike
            
            sweep = row_j.get('swept_prev_int_high', 0) == 1
            choch = row_j.get('int_choch_up', 0) == 1
            
            if cross_vah and vol_spike and delta_spike:
                hunt_bar = j
                break
            elif (sweep or choch) and vol_spike and delta_spike:
                hunt_bar = j
                break
        
        if hunt_bar is None:
            return False, None
        
        # C) Exhaustion: no stronger buy spike
        for k in range(hunt_bar + 1, min(hunt_bar + self.N_EXHAUST + 1, i)):
            if k >= len(df):
                break
            if df.iloc[k].get('delta_zscore', 0) >= cfg.delta_spike2:
                return False, None
        
        # D) Entry bar
        row_i = df.iloc[i]
        
        if row_i.get('is_bear_candle', 0) != 1:
            return False, None
        
        if row_i.get('volume_zscore', 0) < cfg.vol_spike2:
            return False, None
        
        if row_i.get('delta_zscore', 0) > -cfg.delta_long_entry:  # Negative delta
            return False, None
        
        reenter = row_i.get('reenter_va_from_above', 0) == 1
        in_va = row_i.get('vp_in_value_area', 0) == 1
        if not (reenter or in_va):
            return False, None
        
        if i - hunt_bar > self.N_ENTRY:
            return False, None
        
        return True, hunt_bar
    
    def detect_mode2_long(self, df: pd.DataFrame, i: int) -> Tuple[bool, Optional[dict]]:
        """
        Mode 2 Long: Trend continuation pullback to FVG/OB
        """
        cfg = self.config
        row_i = df.iloc[i]
        
        # A) Trend filter
        m5_up = row_i.get('m5_trend_up', 0) == 1
        ext_trend_up = row_i.get('ext_trend_dir', 0) > 0
        
        if not (m5_up or ext_trend_up):
            return False, None
        
        # No recent ext_bos_down
        for j in range(max(0, i - self.N_TREND), i):
            if df.iloc[j].get('ext_bos_down', 0) == 1:
                return False, None
        
        # B) Check if in/near bull FVG or OB
        in_bull_fvg = row_i.get('in_bull_fvg', 0) == 1
        near_bull_fvg = row_i.get('near_bull_fvg', 0) == 1
        in_bull_ob = row_i.get('int_in_bull_ob', 0) == 1 or row_i.get('ext_in_bull_ob', 0) == 1
        near_bull_ob = row_i.get('int_near_bull_ob', 0) == 1 or row_i.get('ext_near_bull_ob', 0) == 1
        
        in_zone = in_bull_fvg or in_bull_ob
        near_zone = near_bull_fvg or near_bull_ob
        
        if not (in_zone or near_zone):
            return False, None
        
        # C) Pullback was weak (delta not too negative)
        # Check last 5 bars for pullback
        pb_delta_ok = True
        for j in range(max(0, i - 5), i):
            if df.iloc[j].get('delta_zscore', 0) < -cfg.delta_pb_max:
                pb_delta_ok = False
                break
        
        if not pb_delta_ok:
            return False, None
        
        # D) Entry bar: bull candle with positive delta
        if row_i.get('is_bull_candle', 0) != 1:
            return False, None
        
        if row_i.get('delta_zscore', 0) < cfg.delta_long_entry * 0.5:  # Relaxed for mode2
            return False, None
        
        zone_info = {
            'in_fvg': in_bull_fvg,
            'in_ob': in_bull_ob,
            'near_fvg': near_bull_fvg,
            'near_ob': near_bull_ob
        }
        
        return True, zone_info
    
    def detect_mode2_short(self, df: pd.DataFrame, i: int) -> Tuple[bool, Optional[dict]]:
        """
        Mode 2 Short: Trend continuation pullback to bear FVG/OB
        """
        cfg = self.config
        row_i = df.iloc[i]
        
        # A) Trend filter
        m5_down = row_i.get('m5_trend_down', 0) == 1
        ext_trend_down = row_i.get('ext_trend_dir', 0) < 0
        
        if not (m5_down or ext_trend_down):
            return False, None
        
        # No recent ext_bos_up
        for j in range(max(0, i - self.N_TREND), i):
            if df.iloc[j].get('ext_bos_up', 0) == 1:
                return False, None
        
        # B) In/near bear zone
        in_bear_fvg = row_i.get('in_bear_fvg', 0) == 1
        near_bear_fvg = row_i.get('near_bear_fvg', 0) == 1
        in_bear_ob = row_i.get('int_in_bear_ob', 0) == 1 or row_i.get('ext_in_bear_ob', 0) == 1
        near_bear_ob = row_i.get('int_near_bear_ob', 0) == 1 or row_i.get('ext_near_bear_ob', 0) == 1
        
        in_zone = in_bear_fvg or in_bear_ob
        near_zone = near_bear_fvg or near_bear_ob
        
        if not (in_zone or near_zone):
            return False, None
        
        # C) Pullback weak (delta not too positive)
        pb_delta_ok = True
        for j in range(max(0, i - 5), i):
            if df.iloc[j].get('delta_zscore', 0) > cfg.delta_pb_max:
                pb_delta_ok = False
                break
        
        if not pb_delta_ok:
            return False, None
        
        # D) Entry bar
        if row_i.get('is_bear_candle', 0) != 1:
            return False, None
        
        if row_i.get('delta_zscore', 0) > -cfg.delta_long_entry * 0.5:
            return False, None
        
        zone_info = {
            'in_fvg': in_bear_fvg,
            'in_ob': in_bear_ob,
            'near_fvg': near_bear_fvg,
            'near_ob': near_bear_ob
        }
        
        return True, zone_info
    
    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply P2_v2 filter to dataframe - VECTORIZED VERSION"""
        # Add delta features
        df = self.add_delta_features(df)
        
        # Initialize output columns
        n = len(df)
        p2_mask = np.zeros(n, dtype=int)
        p2_mode = np.zeros(n, dtype=int)
        p2_side = np.zeros(n, dtype=int)
        p2_hunt_bar = np.full(n, -1, dtype=int)
        
        # Convert to numpy for speed
        delta_zscore = df['delta_zscore'].values
        volume_zscore = df['volume_zscore'].values
        cross_val_down = df['cross_val_down'].values
        cross_vah_up = df['cross_vah_up'].values
        reenter_va_below = df['reenter_va_from_below'].values
        reenter_va_above = df['reenter_va_from_above'].values
        vp_in_va = df['vp_in_value_area'].values
        is_bull = df['is_bull_candle'].values
        is_bear = df['is_bear_candle'].values
        m5_up = df['m5_trend_up'].values if 'm5_trend_up' in df.columns else np.zeros(n)
        m5_down = df['m5_trend_down'].values if 'm5_trend_down' in df.columns else np.zeros(n)
        ext_trend = df['ext_trend_dir'].values if 'ext_trend_dir' in df.columns else np.zeros(n)
        swept_low = df['swept_prev_int_low'].values
        swept_high = df['swept_prev_int_high'].values
        int_choch_down = df['int_choch_down'].values
        int_choch_up = df['int_choch_up'].values
        ext_bos_down = df['ext_bos_down'].values
        ext_bos_up = df['ext_bos_up'].values
        in_bull_fvg = df['in_bull_fvg'].values
        in_bear_fvg = df['in_bear_fvg'].values
        near_bull_fvg = df['near_bull_fvg'].values
        near_bear_fvg = df['near_bear_fvg'].values
        int_in_bull_ob = df['int_in_bull_ob'].values
        int_in_bear_ob = df['int_in_bear_ob'].values
        
        cfg = self.config
        
        for i in range(self.N_ZSCORE, n):
            if i % 10000 == 0:
                print(f"    Processing bar {i}/{n}...")
            
            # === MODE 1 LONG ===
            if m5_up[i] == 1:
                hunt_bar = -1
                for j in range(max(0, i - self.N_BREAK), i):
                    # Hunt conditions
                    is_hunt = (
                        (cross_val_down[j] == 1 or swept_low[j] == 1 or int_choch_down[j] == 1) and
                        volume_zscore[j] >= cfg.vol_spike1 and
                        delta_zscore[j] <= -cfg.delta_spike1
                    )
                    if is_hunt:
                        hunt_bar = j
                        break
                
                if hunt_bar >= 0:
                    # Exhaustion check
                    exhausted = True
                    for k in range(hunt_bar + 1, min(hunt_bar + self.N_EXHAUST + 1, i)):
                        if delta_zscore[k] <= -cfg.delta_spike2:
                            exhausted = False
                            break
                    
                    if exhausted and i - hunt_bar <= self.N_ENTRY:
                        # Entry conditions
                        if (is_bull[i] == 1 and 
                            volume_zscore[i] >= cfg.vol_spike2 and
                            delta_zscore[i] >= cfg.delta_long_entry and
                            (reenter_va_below[i] == 1 or vp_in_va[i] == 1)):
                            p2_mask[i] = 1
                            p2_mode[i] = 1
                            p2_side[i] = 1
                            p2_hunt_bar[i] = hunt_bar
                            continue
            
            # === MODE 1 SHORT ===
            if m5_down[i] == 1:
                hunt_bar = -1
                for j in range(max(0, i - self.N_BREAK), i):
                    is_hunt = (
                        (cross_vah_up[j] == 1 or swept_high[j] == 1 or int_choch_up[j] == 1) and
                        volume_zscore[j] >= cfg.vol_spike1 and
                        delta_zscore[j] >= cfg.delta_spike1
                    )
                    if is_hunt:
                        hunt_bar = j
                        break
                
                if hunt_bar >= 0:
                    exhausted = True
                    for k in range(hunt_bar + 1, min(hunt_bar + self.N_EXHAUST + 1, i)):
                        if delta_zscore[k] >= cfg.delta_spike2:
                            exhausted = False
                            break
                    
                    if exhausted and i - hunt_bar <= self.N_ENTRY:
                        if (is_bear[i] == 1 and
                            volume_zscore[i] >= cfg.vol_spike2 and
                            delta_zscore[i] <= -cfg.delta_long_entry and
                            (reenter_va_above[i] == 1 or vp_in_va[i] == 1)):
                            p2_mask[i] = 1
                            p2_mode[i] = 1
                            p2_side[i] = -1
                            p2_hunt_bar[i] = hunt_bar
                            continue
            
            # === MODE 2 LONG ===
            if m5_up[i] == 1 or ext_trend[i] > 0:
                # No recent ext_bos_down
                has_bos_down = False
                for j in range(max(0, i - self.N_TREND), i):
                    if ext_bos_down[j] == 1:
                        has_bos_down = True
                        break
                
                if not has_bos_down:
                    in_zone = in_bull_fvg[i] == 1 or int_in_bull_ob[i] == 1
                    near_zone = near_bull_fvg[i] == 1
                    
                    if in_zone or near_zone:
                        # Pullback weak
                        pb_ok = True
                        for j in range(max(0, i - 5), i):
                            if delta_zscore[j] < -cfg.delta_pb_max:
                                pb_ok = False
                                break
                        
                        if pb_ok and is_bull[i] == 1 and delta_zscore[i] >= cfg.delta_long_entry * 0.5:
                            p2_mask[i] = 1
                            p2_mode[i] = 2
                            p2_side[i] = 1
                            continue
            
            # === MODE 2 SHORT ===
            if m5_down[i] == 1 or ext_trend[i] < 0:
                has_bos_up = False
                for j in range(max(0, i - self.N_TREND), i):
                    if ext_bos_up[j] == 1:
                        has_bos_up = True
                        break
                
                if not has_bos_up:
                    in_zone = in_bear_fvg[i] == 1 or int_in_bear_ob[i] == 1
                    near_zone = near_bear_fvg[i] == 1
                    
                    if in_zone or near_zone:
                        pb_ok = True
                        for j in range(max(0, i - 5), i):
                            if delta_zscore[j] > cfg.delta_pb_max:
                                pb_ok = False
                                break
                        
                        if pb_ok and is_bear[i] == 1 and delta_zscore[i] <= -cfg.delta_long_entry * 0.5:
                            p2_mask[i] = 1
                            p2_mode[i] = 2
                            p2_side[i] = -1
                            continue
        
        df['p2_mask'] = p2_mask
        df['p2_mode'] = p2_mode
        df['p2_side'] = p2_side
        df['p2_hunt_bar'] = p2_hunt_bar
        
        return df


def main():
    print("="*70)
    print("P2_v2 FILTER - DELTA INTEGRATION")
    print("="*70)
    
    # Load data
    csv_path = ROOT / "output/production_10weeks_v3/features_all_10weeks_v3.csv"
    print(f"\n[1] Loading data from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Total bars: {len(df)}")
    
    # Test each config
    configs = [CONFIG_A, CONFIG_B, CONFIG_C]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg.name}")
        print(f"{'='*70}")
        
        filter = P2V2Filter(cfg)
        df_filtered = filter.apply_filter(df)
        
        # Stats
        n_p2 = (df_filtered['p2_mask'] == 1).sum()
        n_mode1 = (df_filtered['p2_mode'] == 1).sum()
        n_mode2 = (df_filtered['p2_mode'] == 2).sum()
        n_long = (df_filtered['p2_side'] == 1).sum()
        n_short = (df_filtered['p2_side'] == -1).sum()
        
        print(f"\n[STATS]")
        print(f"  Total P2 setups: {n_p2} ({n_p2/len(df)*100:.2f}%)")
        print(f"  Mode 1 (VA Reversal): {n_mode1}")
        print(f"  Mode 2 (Trend Cont.): {n_mode2}")
        print(f"  Long: {n_long}, Short: {n_short}")
        
        # Save
        out_path = ROOT / f"output/p2_v2_{cfg.name}.csv"
        df_p2 = df_filtered[df_filtered['p2_mask'] == 1].copy()
        df_p2.to_csv(out_path, index=False)
        print(f"\n  Saved to {out_path.name}")


if __name__ == "__main__":
    main()

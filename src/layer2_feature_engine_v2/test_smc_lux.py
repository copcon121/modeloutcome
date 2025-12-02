
import json
import pandas as pd
from layer2_feature_engine_v2.smc_core.smc_lux import LuxSMC

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_test():
    filepath = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
    print(f"Loading data from {filepath}...")
    data = load_data(filepath)
    print(f"Loaded {len(data)} bars.")
    
    smc = LuxSMC(swing_length=50, internal_length=5)
    
    output_features = []
    
    for row in data:
        # Parse timestamp string to int (simple hash or just use index for now if needed, but int required)
        # We'll use bar_index as timestamp for simplicity in this test, or parse if needed.
        # The class expects int timestamp.
        ts_str = row['timestamp']
        # Simple mock timestamp
        ts = row['bar_index'] 
        
        state = smc.update(
            open_=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            timestamp=ts,
            bar_index=row['bar_index']
        )
        
        # Collect features for ML
        features = {
            'bar_index': row['bar_index'],
            'swing_trend': state.swing_trend,
            'internal_trend': state.internal_trend,
            'bos_bull': int(state.bos_bull),
            'bos_bear': int(state.bos_bear),
            'choch_bull': int(state.choch_bull),
            'choch_bear': int(state.choch_bear),
            'int_bos_bull': int(state.internal_bos_bull),
            'int_bos_bear': int(state.internal_bos_bear),
            'int_choch_bull': int(state.internal_choch_bull),
            'int_choch_bear': int(state.internal_choch_bear),
            'swing_high': state.swing_high.price if state.swing_high else None,
            'swing_low': state.swing_low.price if state.swing_low else None,
            'int_high': state.internal_high.price if state.internal_high else None,
            'int_low': state.internal_low.price if state.internal_low else None,
            'pd_premium_high': state.trailing_top,
            'pd_discount_low': state.trailing_bottom,
            'active_swing_obs': len([ob for ob in smc.swing_obs if not ob.mitigated]),
            'active_int_obs': len([ob for ob in smc.internal_obs if not ob.mitigated]),
            'active_fvgs': len([fvg for fvg in smc.fvgs if not fvg.mitigated])
        }
        output_features.append(features)
        
        if state.bos_bull or state.bos_bear:
            print(f"Bar {row['bar_index']}: Swing BOS {'Bull' if state.bos_bull else 'Bear'}")
            
    # Convert to DataFrame to check
    df = pd.DataFrame(output_features)
    print("\nFeature Summary:")
    print(df.describe().transpose())
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    run_test()

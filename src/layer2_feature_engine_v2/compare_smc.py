
import json
import pandas as pd
import sys
import os
from dataclasses import dataclass

# Add current directory to path to import local modules
# Run from src directory: python -m layer2_feature_engine_v2.compare_smc

from layer2_feature_engine_v2.config import SMCConfig
from layer2_feature_engine_v2.smc_core.smc_lux import LuxSMC

@dataclass
class Bar:
    o: float
    h: float
    l: float
    c: float
    volume: float = 0
    bar_index: int = 0

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_nearest_ob(active_obs, current_close):
    if not active_obs:
        return None
    
    # Filter for valid OBs (not mitigated)
    valid_obs = [ob for ob in active_obs if not ob.mitigated]
    if not valid_obs:
        return None
        
    # Find OB with minimum distance to close
    best_ob = None
    min_dist = float('inf')
    
    for ob in valid_obs:
        dist = 0
        if current_close > ob.top:
            dist = current_close - ob.top
        elif current_close < ob.bottom:
            dist = ob.bottom - current_close
        else:
            dist = 0
            
        if dist < min_dist:
            min_dist = dist
            best_ob = ob
        elif dist == min_dist:
            # Tie-breaker: use most recent (highest source bar index)
            if best_ob and ob.bar_index > best_ob.bar_index:
                best_ob = ob
                
    return best_ob

def get_nearest_fvg(active_fvgs, current_close):
    if not active_fvgs:
        return None
        
    valid_fvgs = [f for f in active_fvgs if not f.mitigated]
    if not valid_fvgs:
        return None
        
    best_fvg = None
    min_dist = float('inf')
    
    for f in valid_fvgs:
        dist = 0
        if current_close > f.top:
            dist = current_close - f.top
        elif current_close < f.bottom:
            dist = f.bottom - current_close
        else:
            dist = 0
            
        if dist < min_dist:
            min_dist = dist
            best_fvg = f
        elif dist == min_dist:
             if best_fvg and f.bar_index > best_fvg.bar_index:
                 best_fvg = f
                 
    return best_fvg

def run_comparison():
    # Path to reference JSONL
    jsonl_path = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
    
    print(f"Loading data from {jsonl_path}...")
    json_data = load_data(jsonl_path)
    print(f"Loaded {len(json_data)} bars.")
    if len(json_data) > 0:
        print(f"First Bar: {json_data[0]['bar_index']}, Last Bar: {json_data[-1]['bar_index']}")

    # Initialize LuxSMC
    # Using config values where appropriate, but LuxSMC defaults are usually fine
    smc = LuxSMC(
        swing_length=50,
        internal_length=5
    )

    mismatches = {
        'ob_detected': 0,
        'fvg_detected': 0,
        'choch_detected': 0,
        'bos_detected': 0
    }
    
    print("Starting comparison...")
    
    # Clear debug log
    with open("debug_trace.log", "w") as f:
        f.write("Debug Trace Started\n")
    with open("bos_mismatch.log", "w") as f:
        f.write("")
    
    for i, row in enumerate(json_data):
        bar_index = row['bar_index']
        
        # Update LuxSMC
        # Note: LuxSMC expects timestamp as int, we can use bar_index or dummy
        state = smc.update(
            open_=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            timestamp=bar_index, # Using bar_index as timestamp for simplicity
            bar_index=bar_index
        )
        
        # Debug Trace for specific range
        target_debug_center = 4451
        debug_range = (target_debug_center - 50, target_debug_center + 5)
        in_debug = debug_range[0] <= bar_index <= debug_range[1]
        
        if in_debug:
             # File log (full)
             with open("debug_trace.log", "a") as f:
                 f.write(f"--- Bar {bar_index} ---\n")
                 f.write(f"  OHLC: O={row['open']}, H={row['high']}, L={row['low']}, C={row['close']}\n")
                 f.write(f"  CS Raw: OB_Idx={row.get('ob_bar_index')}, FVG_Idx={row.get('fvg_bar_index')}, FVG_Det={row.get('fvg_detected')}\n")
                 f.write(f"  PY Active: OBs={len(smc.swing_obs) + len(smc.internal_obs)}, FVGs={len(smc.fvgs)}\n")
                 f.write(f"  Structure Ext: BOS_Up={state.bos_bull}, BOS_Dn={state.bos_bear}, CH_Up={state.choch_bull}, CH_Dn={state.choch_bear}\n")
                 f.write(f"  Structure Int: BOS_Up={state.internal_bos_bull}, BOS_Dn={state.internal_bos_bear}, CH_Up={state.internal_choch_bull}, CH_Dn={state.internal_choch_bear}\n")
        
        # Check OB
        cs_ob_index = row.get('ob_bar_index', -1)
        
        all_active_obs = [ob for ob in smc.swing_obs if not ob.mitigated] + [ob for ob in smc.internal_obs if not ob.mitigated]
        
        py_nearest_ob = get_nearest_ob(all_active_obs, row['close'])
        py_ob_index = py_nearest_ob.bar_index if py_nearest_ob else -1
        
        match = False
        start_bar_index = json_data[0]['bar_index']
        
        if cs_ob_index > 0 and cs_ob_index < start_bar_index:
             match = True # Treat as match or ignore legacy
        elif cs_ob_index <= 0 and py_ob_index == -1:
            match = True
        elif cs_ob_index == py_ob_index:
            match = True
            
        if not match:
             mismatches['ob_detected'] += 1

        # Check FVG
        cs_fvg_index = row.get('fvg_bar_index', -1)
        
        py_nearest_fvg = get_nearest_fvg(smc.fvgs, row['close'])
        py_fvg_index = py_nearest_fvg.bar_index if py_nearest_fvg else -1
        
        match = False
        if cs_fvg_index > 0 and cs_fvg_index < start_bar_index:
            match = True # Ignore legacy
        elif cs_fvg_index <= 0 and py_fvg_index == -1:
            match = True
        elif cs_fvg_index == py_fvg_index or cs_fvg_index == py_fvg_index - 1:
            match = True
            
        if not match:
            mismatches['fvg_detected'] += 1

        # Check BOS (External)
        py_bos = state.bos_bull or state.bos_bear
        cs_bos = row.get('bos_detected', False)
        
        if py_bos != cs_bos:
             # print(f"[BOS Mismatch] Bar {bar_index}: Py={py_bos}, CS={cs_bos}")
             with open("bos_mismatch.log", "a") as f:
                 f.write(f"[BOS Mismatch] Bar {bar_index}: Py={py_bos}, CS={cs_bos}\n")
             mismatches['bos_detected'] += 1
             # break # Don't break, see all mismatches

    print("Comparison Complete.")
    print("Mismatches:", mismatches)
    print(f"Total Bars: {len(json_data)}")

if __name__ == "__main__":
    run_comparison()

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.smc_core.smc_lux import LuxSMC

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def plot_smc(data, output_file='smc_chart_full.png', start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(data)
        
    subset = data[start_idx:end_idx]
    
    # Prepare data for LuxSMC
    smc = LuxSMC(swing_length=50, internal_length=5)
    
    # Collect features for plotting
    processed_bars = []
    
    # Events to plot
    ext_swings = [] # (idx, price, type)
    int_swings = []
    ext_bos = [] # (idx, price, type)
    int_bos = []
    sweeps = [] # (idx, price, type)
    
    captured_obs = []
    captured_fvgs = []
    
    print("Running LuxSMC and collecting features...")
    
    for i, row in enumerate(data):
        bar_index = row['bar_index']
        state = smc.update(
            open_=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            timestamp=bar_index,
            bar_index=bar_index
        )
        
        # Capture Zones (Naive: if new zone added to front)
        if smc.swing_obs and smc.swing_obs[0] not in captured_obs:
            captured_obs.append(smc.swing_obs[0])
        if smc.internal_obs and smc.internal_obs[0] not in captured_obs:
            captured_obs.append(smc.internal_obs[0])
        if smc.fvgs and smc.fvgs[0] not in captured_fvgs:
            captured_fvgs.append(smc.fvgs[0])
            
        if i >= start_idx and i < end_idx:
            processed_bars.append(row)
            
            # Capture Structure Events
            if state.bos_bull: ext_bos.append((bar_index, row['high'], 'BOS Up'))
            if state.bos_bear: ext_bos.append((bar_index, row['low'], 'BOS Dn'))
            if state.choch_bull: ext_bos.append((bar_index, row['high'], 'CHoCH Up'))
            if state.choch_bear: ext_bos.append((bar_index, row['low'], 'CHoCH Dn'))
            
            if state.internal_bos_bull: int_bos.append((bar_index, row['high'], 'iBOS Up'))
            if state.internal_bos_bear: int_bos.append((bar_index, row['low'], 'iBOS Dn'))
            if state.internal_choch_bull: int_bos.append((bar_index, row['high'], 'iCHoCH Up'))
            if state.internal_choch_bear: int_bos.append((bar_index, row['low'], 'iCHoCH Dn'))
            
            # Capture Sweeps
            if state.swept_prev_ext_high: sweeps.append((bar_index, row['high'], 'Swp Ext Hi'))
            if state.swept_prev_ext_low: sweeps.append((bar_index, row['low'], 'Swp Ext Lo'))
            if state.swept_prev_int_high: sweeps.append((bar_index, row['high'], 'Swp Int Hi'))
            if state.swept_prev_int_low: sweeps.append((bar_index, row['low'], 'Swp Int Lo'))

    print(f"Captured OBs: {len(captured_obs)}")
    if captured_obs:
        print(f"Sample OB: {captured_obs[0]}")
    print(f"Captured FVGs: {len(captured_fvgs)}")

    # Determine absolute bar index range
    if processed_bars:
        min_bar_idx = processed_bars[0]['bar_index']
        max_bar_idx = processed_bars[-1]['bar_index']
    else:
        min_bar_idx = 0
        max_bar_idx = float('inf')
        
    print(f"Filtering range: {min_bar_idx} to {max_bar_idx}")

    # Filter for plot range
    # Debug filtering
    plot_obs = []
    for ob in captured_obs:
        if ob.bar_index >= min_bar_idx - 100 and ob.bar_index <= max_bar_idx:
            plot_obs.append(ob)
        else:
            # print(f"Excluded OB at {ob.bar_index}")
            pass
            
    plot_fvgs = [fvg for fvg in captured_fvgs if fvg.bar_index >= min_bar_idx - 100 and fvg.bar_index <= max_bar_idx]
    
    print(f"Plotting OBs: {len(plot_obs)}")
    print(f"Plotting FVGs: {len(plot_fvgs)}")
    
    print("Plotting...")
    # Wide chart for 1 day M1
    fig, ax = plt.subplots(figsize=(60, 15))
    
    # Plot Candles
    opens = [b['open'] for b in processed_bars]
    highs = [b['high'] for b in processed_bars]
    lows = [b['low'] for b in processed_bars]
    closes = [b['close'] for b in processed_bars]
    indices = [b['bar_index'] for b in processed_bars]
    
    colors = ['green' if c >= o else 'red' for c, o in zip(closes, opens)]
    
    ax.vlines(indices, lows, highs, color=colors, linewidth=1)
    for idx, o, c, color in zip(indices, opens, closes, colors):
        height = abs(c - o)
        bottom = min(c, o)
        rect = patches.Rectangle((idx - 0.3, bottom), 0.6, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        
    # Plot Zones
    for ob in plot_obs:
        # Determine Color/Style based on state
        if ob.mitigated:
            color = 'gray'
            alpha = 0.1
            label = "OB (Mit)"
        elif not ob.active: # Expired by Age
            color = 'orange'
            alpha = 0.1
            label = "OB (Exp)"
        else: # Active
            color = 'darkgreen' if ob.type == 1 else 'darkred'
            alpha = 0.5
            label = "OB"

        # Extend to end or some length
        width = 100 # Visual width
        rect = patches.Rectangle((ob.bar_index, ob.bottom), width, ob.top - ob.bottom, 
                                 linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        # Label
        # ax.text(ob.bar_index, ob.top, label, fontsize=8, color=color)
        
    for fvg in plot_fvgs:
        if fvg.mitigated:
            color = 'lightgray'
            alpha = 0.1
        elif not fvg.active:
            color = 'yellow'
            alpha = 0.1
        else:
            color = 'lime' if fvg.type == 1 else 'pink'
            alpha = 0.4
            
        width = 50
        rect = patches.Rectangle((fvg.bar_index, fvg.bottom), width, fvg.top - fvg.bottom, 
                                 linewidth=0, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
    # Plot Structure Events
    for idx, price, label in ext_bos:
        ax.plot(idx, price, 'bo', markersize=8)
        ax.text(idx, price, label, fontsize=9, color='blue', rotation=45)
        
    for idx, price, label in int_bos:
        ax.plot(idx, price, 'co', markersize=5)
        # ax.text(idx, price, label, fontsize=7, color='cyan', rotation=45) # Too cluttered?
        
    # Plot Sweeps
    for idx, price, label in sweeps:
        ax.plot(idx, price, 'rx', markersize=6)
        # ax.text(idx, price, label, fontsize=7, color='red', rotation=90)

    ax.set_xlim(indices[0], indices[-1])
    min_y = min(lows)
    max_y = max(highs)
    ax.set_ylim(min_y, max_y)
    
    plt.title("LuxAlgo SMC Full Logic (1 Day M1)")
    plt.xlabel("Bar Index")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(output_file, dpi=100) # Higher DPI for zoom
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    data_path = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
    data = load_data(data_path)
    # Plot all data
    plot_smc(data, start_idx=0, end_idx=len(data))

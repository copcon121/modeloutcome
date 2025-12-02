import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def verify_macro():
    data_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\labeled_outcome"
    files = sorted(glob.glob(os.path.join(data_dir, "*_labeled.csv")))
    
    if not files:
        print("No files found.")
        return

    # Load the last file (most recent data)
    f = files[-1]
    print(f"Loading {f}...")
    df = pd.read_csv(f)
    
    # 14 Macro Features
    h1_cols = [
        'h1_trend_up', 'h1_trend_down', 
        'h1_premium', 'h1_discount',
        'dist_to_h1_swing_high', 'dist_to_h1_swing_low',
        'near_h1_fvg'
    ]
    h4_cols = [c.replace('h1', 'h4') for c in h1_cols]
    
    macro_cols = h1_cols + h4_cols
    
    print("\n--- Statistics for 14 Macro Features ---")
    print(df[macro_cols].describe().T[['mean', 'std', 'min', 'max']])
    
    # Check for constant values
    for c in macro_cols:
        if df[c].nunique() <= 1:
            print(f"WARNING: Feature {c} is constant! Value: {df[c].iloc[0]}")
            
    # Visualization
    # Plot H1 Trend and Distances
    subset = df.iloc[-500:].copy().reset_index(drop=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # 1. Price & Trend
    ax = axes[0]
    ax.plot(subset.index, subset['close'], label='Close', color='black')
    
    # Highlight H1 Trend
    # Up Trend
    up_trend = subset[subset['h1_trend_up'] == 1]
    ax.scatter(up_trend.index, up_trend['close'], color='green', s=10, alpha=0.3, label='H1 Trend Up')
    
    # Down Trend
    down_trend = subset[subset['h1_trend_down'] == 1]
    ax.scatter(down_trend.index, down_trend['close'], color='red', s=10, alpha=0.3, label='H1 Trend Down')
    
    ax.set_title("Price & H1 Trend")
    ax.legend()
    
    # 2. H1 Swing Distances
    ax = axes[1]
    ax.plot(subset.index, subset['dist_to_h1_swing_high'], label='Dist to H1 High', color='green')
    ax.plot(subset.index, subset['dist_to_h1_swing_low'], label='Dist to H1 Low', color='red')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("H1 Swing Distances (in ATRs)")
    ax.legend()
    
    # 3. Premium/Discount
    ax = axes[2]
    ax.plot(subset.index, subset['h1_premium'], label='H1 Premium', color='orange')
    ax.plot(subset.index, subset['h1_discount'], label='H1 Discount', color='blue')
    ax.set_title("H1 Premium/Discount Zones")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("verify_macro_features.png")
    print("\nPlot saved to verify_macro_features.png")

if __name__ == "__main__":
    verify_macro()

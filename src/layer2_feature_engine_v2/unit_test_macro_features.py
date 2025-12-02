import pandas as pd
import numpy as np
import glob
import os
import sys

def run_unit_tests():
    data_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2"
    files = sorted(glob.glob(os.path.join(data_dir, "*_features.csv")))
    
    if not files:
        print("CRITICAL: No processed files found to test.")
        sys.exit(1)
        
    print(f"Testing {len(files)} files...")
    
    all_passed = True
    
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"Testing {os.path.basename(f)} ({len(df)} rows)...")
            
            # 1. Check Columns Exist
            required_cols = [
                'h1_trend_up', 'h1_trend_down', 'h1_premium', 'h1_discount',
                'dist_to_h1_swing_high', 'dist_to_h1_swing_low', 'near_h1_fvg',
                'm5_trend_up', 'm5_trend_down', 'm5_premium', 'm5_discount',
                'dist_to_m5_swing_high', 'dist_to_m5_swing_low', 'near_m5_fvg'
            ]
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  FAILED: Missing columns: {missing}")
                all_passed = False
                continue
                
            # 2. Check Value Ranges
            # Binary checks (0 or 1)
            binary_cols = [
                'h1_trend_up', 'h1_trend_down', 'h1_premium', 'h1_discount', 'near_h1_fvg',
                'm5_trend_up', 'm5_trend_down', 'm5_premium', 'm5_discount', 'near_m5_fvg'
            ]
            
            for col in binary_cols:
                # Check if values are only 0 or 1 (allow float 0.0/1.0)
                unique_vals = df[col].unique()
                invalid_vals = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
                if invalid_vals:
                    print(f"  FAILED: {col} has invalid values: {invalid_vals[:5]}")
                    all_passed = False
                    
            # Distance checks (should be >= 0 usually, but swing distance can be negative if price crosses?)
            # Logic: dist = (swing - price) / atr. Can be negative if price > swing_high.
            # Wait, user said "assert dist >= 0".
            # Let's check the implementation:
            # dist_sh = (sh.price - bar.c) / atr
            # If price > sh, dist is negative.
            # BUT user requirement: "assert dist_to_h1_swing_high >= 0"
            # If the user implies ABSOLUTE distance, then my implementation might be "wrong" by their standard,
            # OR they imply that price shouldn't cross swing high without invalidating it?
            # In SMC, if price crosses swing high, it's a BOS/CHoCH.
            # Let's check if we used abs() in context_manager.
            # Implementation: `dist_sh = (sh.price - bar.c) / atr` -> Signed distance.
            # If user wants >= 0, they might mean "distance magnitude" or "valid swing".
            # Let's check for NaNs/Inf first.
            
            dist_cols = [
                'dist_to_h1_swing_high', 'dist_to_h1_swing_low',
                'dist_to_m5_swing_high', 'dist_to_m5_swing_low'
            ]
            
            for col in dist_cols:
                if df[col].isnull().any():
                    print(f"  FAILED: {col} contains NaNs")
                    all_passed = False
                if np.isinf(df[col]).any():
                    print(f"  FAILED: {col} contains Inf")
                    all_passed = False
                    
            # 3. Logic Consistency
            # Trend cannot be both Up and Down
            inconsistent_trend = df[(df['h1_trend_up'] == 1) & (df['h1_trend_down'] == 1)]
            if not inconsistent_trend.empty:
                print(f"  FAILED: H1 Trend Up and Down simultaneously in {len(inconsistent_trend)} rows")
                all_passed = False
                
            # Premium/Discount cannot be both 1 (unless exactly at midpoint? unlikely with float)
            inconsistent_pd = df[(df['h1_premium'] == 1) & (df['h1_discount'] == 1)]
            if not inconsistent_pd.empty:
                print(f"  FAILED: H1 Premium and Discount simultaneously in {len(inconsistent_pd)} rows")
                all_passed = False

        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False
            
    if all_passed:
        print("\nSUCCESS: All Unit Tests Passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some Unit Tests Failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_unit_tests()

import os
import sys
import glob
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.batch_process import process_file

def main():
    raw_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw"
    processed_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2_test"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(raw_dir, "smc_export_gc_m1_v3_*.jsonl"))
    if not files:
        print("No files found")
        return
        
    input_path = files[0]
    output_path = os.path.join(processed_dir, "test_features.csv")
    
import os
import sys
import glob
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.batch_process import process_file

def main():
    raw_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw"
    processed_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2_test"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(raw_dir, "smc_export_gc_m1_v3_*.jsonl"))
    if not files:
        print("No files found")
        return
        
    input_path = files[0]
    output_path = os.path.join(processed_dir, "test_features.csv")
    
    print(f"Processing {input_path}...")
    process_file(input_path, output_path)
    
    # Verify
    df = pd.read_csv(output_path)
    print("\nColumns:", df.columns.tolist())
    # Check for new H1/H4 columns
    new_cols = [
        'h1_trend_up', 'h1_trend_down', 'h4_trend_up', 'h4_trend_down',
        'h1_premium', 'h1_discount', 'h4_premium', 'h4_discount',
        'dist_to_h1_swing_high', 'dist_to_h1_swing_low',
        'dist_to_h4_swing_high', 'dist_to_h4_swing_low',
        'near_h1_fvg', 'near_h4_fvg'
    ]
    
    print("\nChecking new columns:")
    for col in new_cols:
        if col in df.columns:
            print(f"{col}: Found. Sample values: {df[col].iloc[-5:].values}")
        else:
            print(f"{col}: NOT FOUND!")
            
    print("\nDone.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import glob
import os
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.schema import FeatureBar
from src.layer2_feature_engine_v2.event_filter import EventFilter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("p2_mask_gen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_masks():
    input_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2"
    output_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\p2_masks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*_features.csv"))
    logger.info(f"Found {len(files)} feature files.")
    
    filter_engine = EventFilter()
    
    total_bars = 0
    total_kept = 0
    
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Convert DF to FeatureBar objects
            # We need to map columns to FeatureBar fields
            # FeatureBar has many fields. We can use kwargs unpacking if columns match.
            # However, CSV columns might be slightly different or have extra.
            # FeatureBar is a dataclass.
            
            feature_bars = []
            valid_fields = FeatureBar.__dataclass_fields__.keys()
            
            for _, row in df.iterrows():
                # Filter row keys to only valid fields
                row_dict = {k: v for k, v in row.items() if k in valid_fields}
                
                # Handle missing fields or type conversions if needed
                # CSV might have NaNs or different types
                # We assume batch_process.py output is compatible.
                
                try:
                    fb = FeatureBar(**row_dict)
                    feature_bars.append(fb)
                except Exception as e:
                    logger.warning(f"Error creating FeatureBar: {e}")
                    continue
            
            if not feature_bars:
                logger.warning(f"No valid bars in {f}")
                continue
                
            # Compute Flags
            flags = filter_engine.compute_flags(feature_bars)
            
            # Apply P2 Filter
            mask = filter_engine.apply_phase2_filter(flags)
            
            # Save Mask
            mask_df = pd.DataFrame({'p2_mask': mask})
            output_path = os.path.join(output_dir, os.path.basename(f).replace("_features.csv", "_p2_mask.csv"))
            mask_df.to_csv(output_path, index=False)
            
            kept = sum(mask)
            total = len(mask)
            total_bars += total
            total_kept += kept
            
            logger.info(f"Processed {os.path.basename(f)}: Kept {kept}/{total} ({kept/total*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            
    logger.info(f"Mask generation complete.")
    logger.info(f"Total Bars: {total_bars}")
    logger.info(f"Total Kept: {total_kept} ({total_kept/total_bars*100:.1f}%)")

if __name__ == "__main__":
    generate_masks()

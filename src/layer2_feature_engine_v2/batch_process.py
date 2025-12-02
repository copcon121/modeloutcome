import os
import sys
import glob
import logging
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_raw_bar(data):
    """Convert JSON dict to RawBar"""
    # Parse timestamp (format: "2025-10-06T00:00:00.0000000")
    # Python 3.11+ supports ISO format directly, but for safety with .0000000:
    ts_str = data['timestamp']
    # Truncate sub-seconds if needed or use fromisoformat
    ts = datetime.fromisoformat(ts_str)
    
    bar_data = data['bar']
    tick_data = data.get('tick_features', {})
    
    return RawBar(
        symbol=data.get('symbol', 'GC'),
        timeframe=data.get('timeframe', 'M1'),
        timestamp=ts,
        bar_index=data['bar_index'],
        o=bar_data['o'],
        h=bar_data['h'],
        l=bar_data['l'],
        c=bar_data['c'],
        volume=bar_data['volume'],
        delta=bar_data.get('delta', 0),
        buy_volume=bar_data.get('buy_volume', 0),
        sell_volume=bar_data.get('sell_volume', 0),
        best_bid=bar_data.get('best_bid', bar_data['c']),
        best_ask=bar_data.get('best_ask', bar_data['c']),
        tick_speed=tick_data.get('tick_speed', 0),
        aggr_buy_speed=tick_data.get('aggr_buy_speed', 0),
        aggr_sell_speed=tick_data.get('aggr_sell_speed', 0),
        price_speed=tick_data.get('price_speed', bar_data['h'] - bar_data['l']),
        vwap_daily=bar_data.get('vwap_daily', 0.0)
    )

def process_file(input_path, output_path):
    logger.info(f"Processing {input_path}...")
    
    # Initialize Manager
    manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    processed_bars = []
    
    try:
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    raw_bar = load_raw_bar(data)
                    feature_bar = manager.update(raw_bar)
                    
                    # Convert to dict and add timestamp/index for reference
                    fb_dict = feature_bar.to_dict()
                    fb_dict['timestamp'] = raw_bar.timestamp.isoformat()
                    fb_dict['bar_index'] = raw_bar.bar_index
                    # Add raw OHLC for labeling/visualization
                    fb_dict['open'] = raw_bar.o
                    fb_dict['high'] = raw_bar.h
                    fb_dict['low'] = raw_bar.l
                    processed_bars.append(fb_dict)
                    
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    continue
                    
        # Save to CSV
        df = pd.DataFrame(processed_bars)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} bars to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process file {input_path}: {e}")
        return False

def main():
    raw_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw"
    processed_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Pattern for 10 weeks data
    pattern = os.path.join(raw_dir, "smc_export_gc_m1_v3_*.jsonl")
    files = glob.glob(pattern)
    
    logger.info(f"Found {len(files)} files to process.")
    
    for input_path in files:
        filename = os.path.basename(input_path)
        output_filename = filename.replace(".jsonl", "_features.csv")
        output_path = os.path.join(processed_dir, output_filename)
        
        process_file(input_path, output_path)
        
    logger.info("Batch processing complete.")

if __name__ == "__main__":
    main()

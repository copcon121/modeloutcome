import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # Load sample data
    data_path = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
    logger.info(f"Loading data from {data_path}")
    raw_data = load_jsonl(data_path)
    
    # Initialize Context Manager
    manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    logger.info("Processing bars...")
    
    for i, item in enumerate(raw_data):
        # Extract bar data
        vol_stats = item.get('volume_stats', {})
        volume = vol_stats.get('total_volume', 0)
        delta = vol_stats.get('delta_close', 0)
        buy_pct = vol_stats.get('buy_percent', 50)
        buy_vol = volume * (buy_pct / 100.0)
        sell_vol = volume - buy_vol
        
        # Create RawBar
        raw_bar = RawBar(
            symbol=item['symbol'],
            timeframe=item['tf'],
            timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
            bar_index=item['bar_index'],
            o=item['open'],
            h=item['high'],
            l=item['low'],
            c=item['close'],
            volume=volume,
            delta=delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            best_bid=0.0, # Not in this export
            best_ask=0.0, # Not in this export
            tick_speed=0.0, # Not in this export
            aggr_buy_speed=0.0,
            aggr_sell_speed=0.0,
            price_speed=0.0,
            vwap_daily=0.0 # Not in this export
        )
        
        # Update Manager
        feature_bar = manager.update(raw_bar)
        
        # Log events
        if feature_bar.ext_bos_up:
            logger.info(f"[{raw_bar.bar_index}] EXT BOS UP detected! Price: {raw_bar.c}")
        if feature_bar.ext_bos_down:
            logger.info(f"[{raw_bar.bar_index}] EXT BOS DOWN detected! Price: {raw_bar.c}")
            
        if feature_bar.swept_prev_ext_high:
            logger.info(f"[{raw_bar.bar_index}] EXT SWEEP HIGH! Price: {raw_bar.h}")
        if feature_bar.swept_prev_ext_low:
            logger.info(f"[{raw_bar.bar_index}] EXT SWEEP LOW! Price: {raw_bar.l}")
            
        if feature_bar.in_bull_fvg:
            logger.info(f"[{raw_bar.bar_index}] IN BULL FVG! Price: {raw_bar.c}")
            
        if feature_bar.ext_in_bull_ob:
            logger.info(f"[{raw_bar.bar_index}] IN EXT BULL OB! Price: {raw_bar.c}")
        if feature_bar.int_in_bull_ob:
            logger.info(f"[{raw_bar.bar_index}] IN INT BULL OB! Price: {raw_bar.c}")
            
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()

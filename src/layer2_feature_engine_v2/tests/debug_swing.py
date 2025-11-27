"""Debug swing detection"""
import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.loaders import iter_raw_bars
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)

for i, bar in enumerate(iter_raw_bars("data/raw/smc_export_gc_m1_v3.jsonl")):
    if i >= 20:  # Just first 20 bars
        break
    
    print(f"\nBar {i} (bar_index={bar.bar_index}): H={bar.h}, L={bar.l}")
    print(f"  Deque size: {len(detector.highs)}")
    
    state = detector.update(bar)
    
    if state.swing_high_price:
        print(f"  -> Swing HIGH: {state.swing_high_price} at bar {state.swing_high_bar_index}")
    if state.swing_low_price:
        print(f"  -> Swing LOW: {state.swing_low_price} at bar {state.swing_low_bar_index}")

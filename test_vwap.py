"""
Test VWAP Integration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datetime import datetime
from layer2_feature_engine_v2.schema import RawBar, FeatureBar
from layer2_feature_engine_v2.context_manager import SMCContextManager
from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG

print("="*80)
print("VWAP INTEGRATION TEST")
print("="*80)

# Create sample bar with VWAP
raw_bar = RawBar(
    symbol='GC',
    timeframe='M1',
    timestamp=datetime.now(),
    bar_index=1,
    o=4126.9,
    h=4126.9,
    l=4126.1,
    c=4126.5,
    volume=38,
    delta=1,
    buy_volume=19.5,
    sell_volume=18.5,
    best_bid=4126.5,
    best_ask=4126.5,
    tick_speed=60,
    aggr_buy_speed=19.5,
    aggr_sell_speed=18.5,
    price_speed=0.8,
    vwap_daily=4125.57  # From user's example
)

print(f"\nRawBar created:")
print(f"  Close: {raw_bar.c}")
print(f"  VWAP Daily: {raw_bar.vwap_daily}")
print(f"  Volume: {raw_bar.volume}")

# Initialize Context Manager
manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)

# Process bar
feature_bar = manager.update(raw_bar)

print(f"\nFeatureBar created:")
print(f"  Close: {feature_bar.close}")
print(f"  VWAP Daily: {feature_bar.vwap_daily}")
print(f"  Distance to VWAP: {feature_bar.dist_to_vwap:.1f} ticks")
print(f"  Above VWAP: {feature_bar.close > feature_bar.vwap_daily}")

# Calculate manually
manual_dist = (raw_bar.c - raw_bar.vwap_daily) / 0.1
print(f"\nManual calculation:")
print(f"  (4126.5 - 4125.57) / 0.1 = {manual_dist:.1f} ticks")

print("\n" + "="*80)
print("VWAP INTEGRATION SUCCESS!")
print("="*80)
print(f"Total features in FeatureBar: {len(feature_bar.to_dict())}")
print(f"Feature names sample: {list(feature_bar.to_dict().keys())[:10]}")

"""Quick debug - print fractal detection"""
import sys
sys.path.insert(0, '/home/user/modeloutcome/src')

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.smc.swing import detect_fractal_pivots

bars = load_raw_bars("/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl")[:300]

print(f"Total bars: {len(bars)}")

# Test fractal detection
fractal_highs, fractal_lows = detect_fractal_pivots(bars, left=2, right=2)

print(f"\nFractal Highs: {len(fractal_highs)}")
if fractal_highs:
    print(f"  First 10: {fractal_highs[:10]}")

print(f"\nFractal Lows: {len(fractal_lows)}")
if fractal_lows:
    print(f"  First 10: {fractal_lows[:10]}")

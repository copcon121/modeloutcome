"""
5-Day Export Validation Script
Validates Phase 2 Feature Engine with production data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

import json
from datetime import datetime

from layer2_feature_engine_v2.dataset_builder import DatasetBuilder
from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.event_filter import EventFilter

print("="*80)
print("5-DAY EXPORT VALIDATION")
print("="*80)

# Find JSONL file
data_dir = Path("data/raw")
jsonl_files = list(data_dir.glob("*.jsonl"))

if not jsonl_files:
    print("\n❌ ERROR: No JSONL files found in data/raw/")
    print("Please export data from NinjaTrader first")
    sys.exit(1)

# Use most recent file
jsonl_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
print(f"\nFile: {jsonl_file.name}")
print(f"Size: {jsonl_file.stat().st_size / 1024 / 1024:.1f} MB")

# Initialize builder
print("\n" + "="*80)
print("STEP 1: LOADING BARS")
print("="*80)

builder = DatasetBuilder(GC_M1_SMC_CONFIG, tick_size=0.1)
raw_bars = builder.load_jsonl(str(jsonl_file))

print(f"\n[OK] Loaded: {len(raw_bars)} bars")
print(f"  Expected: 7,000-7,500 bars for 5 days")
print(f"  Status: {'PASS' if 7000 <= len(raw_bars) <= 7500 else 'WARNING'}")

# Build features
print("\n" + "="*80)
print("STEP 2: BUILDING FEATURES")
print("="*80)

feature_bars = builder.build_features(raw_bars)

print(f"\n✓ Built: {len(feature_bars)} feature bars")
print(f"  Features per bar: {len(feature_bars[0].to_dict())}")

# Sample feature bar
sample = feature_bars[100]
print(f"\n  Sample bar #100:")
print(f"    Close: {sample.close:.1f}")
print(f"    Volume: {sample.volume:.0f}")
print(f"    VWAP: {sample.vwap_daily:.1f}")
print(f"    VP POC: {sample.vp_poc_price:.1f}")
print(f"    Delta: {sample.delta:.1f}")

# Apply event filtering
print("\n" + "="*80)
print("STEP 3: EVENT FILTERING")
print("="*80)

event_filter = EventFilter()
flags = event_filter.compute_flags(feature_bars)
stats = event_filter.get_filter_stats(flags)

print(f"\nFilter Statistics:")
print(f"  Total bars: {stats['total_bars']}")
print(f"\n  Phase 1 (Strict):")
print(f"    Count: {stats['p1_strict']['count']:>5} bars")
print(f"    Pct:   {stats['p1_strict']['pct']:>5.1f}%")
print(f"    Target: 0.5-2%")
print(f"    Status: {'✓ PASS' if 0.5 <= stats['p1_strict']['pct'] <= 2.0 else '⚠ CHECK'}")

print(f"\n  Phase 2 (Moderate) - RECOMMENDED:")
print(f"    Count: {stats['p2_moderate']['count']:>5} bars")
print(f"    Pct:   {stats['p2_moderate']['pct']:>5.1f}%")
print(f"    Target: 20-50%")
print(f"    Status: {'✓ PASS' if 20 <= stats['p2_moderate']['pct'] <= 50 else '⚠ CHECK'}")

print(f"\n  Phase 3 (Loose):")
print(f"    Count: {stats['p3_loose']['count']:>5} bars")
print(f"    Pct:   {stats['p3_loose']['pct']:>5.1f}%")
print(f"    Target: 80-95%")
print(f"    Status: {'✓ PASS' if 80 <= stats['p3_loose']['pct'] <= 95 else '⚠ CHECK'}")

# Build sequences with P2 filter
print("\n" + "="*80)
print("STEP 4: BUILD SEQUENCES (P2 FILTERED)")
print("="*80)

mask_p2 = event_filter.apply_phase2_filter(flags)
filtered_bars_p2 = [fb for fb, keep in zip(feature_bars, mask_p2) if keep]

print(f"\nP2 Filtered bars: {len(filtered_bars_p2)}")

# Build sequences from filtered
sequences, indices = builder.build_sequences(filtered_bars_p2, window_size=60, stride=1)

print(f"\nSequences built:")
print(f"  Shape: {sequences.shape}")
print(f"  Expected: [~{len(filtered_bars_p2) - 60}, 60, {len(feature_bars[0].to_dict())}]")
print(f"  Status: {'✓ PASS' if sequences.shape[1] == 60 else '❌ FAIL'}")

# Export outputs
print("\n" + "="*80)
print("STEP 5: EXPORT OUTPUTS")
print("="*80)

output_dir = Path("output/5day_validation")
output_dir.mkdir(parents=True, exist_ok=True)

# Export CSV (all bars)
csv_path = output_dir / "features_all.csv"
builder.export_csv(feature_bars, str(csv_path))
print(f"\n✓ All features: {csv_path}")
print(f"  Size: {csv_path.stat().st_size / 1024:.0f} KB")

# Export CSV (P2 filtered)
csv_p2_path = output_dir / "features_p2_filtered.csv"
builder.export_csv(filtered_bars_p2, str(csv_p2_path))
print(f"\n✓ P2 filtered features: {csv_p2_path}")
print(f"  Size: {csv_p2_path.stat().st_size / 1024:.0f} KB")

# Export NPY (sequences)
builder.export_npy(sequences, indices, str(output_dir), prefix="sequences_p2")
print(f"\n✓ Sequences: {output_dir}/sequences_p2_sequences.npy")
print(f"  Shape: {sequences.shape}")

# Validation summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

checks = {
    "Bar count (7000-7500)": 7000 <= len(raw_bars) <= 7500,
    "P1 filter (0.5-2%)": 0.5 <= stats['p1_strict']['pct'] <= 2.0,
    "P2 filter (20-50%)": 20 <= stats['p2_moderate']['pct'] <= 50,
    "P3 filter (80-95%)": 80 <= stats['p3_loose']['pct'] <= 95,
    "VWAP present": sample.vwap_daily > 0,
    "VP POC present": sample.vp_poc_price > 0,
    "Sequences shape": sequences.shape[1] == 60,
}

print()
all_pass = True
for check, passed in checks.items():
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"  {check:.<40} {status}")
    if not passed:
        all_pass = False

print(f"\n{'='*80}")
if all_pass:
    print("🎉 ALL CHECKS PASSED - READY FOR 30-DAY EXPORT!")
else:
    print("⚠️  SOME CHECKS FAILED - REVIEW RESULTS")
print(f"{'='*80}")

# Show sample P2 filtered bars
print("\n" + "="*80)
print("SAMPLE P2 FILTERED BARS (First 5)")
print("="*80)

p2_indices = [i for i, keep in enumerate(mask_p2) if keep][:5]
for idx in p2_indices:
    fb = feature_bars[idx]
    fl = flags[idx]
    print(f"\nBar {idx}:")
    print(f"  Close: {fb.close:.1f}, Vol: {fb.volume:.0f}, VWAP: {fb.vwap_daily:.1f}")
    print(f"  Flags: BOS={fl.has_bos_choch}, Zone={fl.in_zone}, HighVol={fl.high_volatility}")
    
    events = []
    if fb.int_bos_up: events.append("INT_BOS_UP")
    if fb.int_bos_down: events.append("INT_BOS_DOWN")
    if fb.ext_bos_up: events.append("EXT_BOS_UP")
    if fb.ext_bos_down: events.append("EXT_BOS_DOWN")
    if fb.in_bull_fvg: events.append("IN_BULL_FVG")
    if fb.in_bear_fvg: events.append("IN_BEAR_FVG")
    
    if events:
        print(f"  Events: {', '.join(events)}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {output_dir}")
print("\nNext steps:")
print("  1. Review features_all.csv manually")
print("  2. Check P2 filtered bars make sense")
print("  3. If all good → Export 30 days from NinjaTrader")
print("  4. Run same validation with 30-day data")
print("  5. Proceed to Phase 3: Labeling")

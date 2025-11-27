import sys
sys.path.insert(0, 'src')

from layer2_feature_engine_v2.dataset_builder import DatasetBuilder
from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.event_filter import EventFilter

builder = DatasetBuilder(GC_M1_SMC_CONFIG, 0.1)
bars = builder.load_jsonl('data/raw/smc_export_gc_m1_v3.jsonl')
fbs = builder.build_features(bars)
flt = EventFilter()
flags = flt.compute_flags(fbs)
stats = flt.get_filter_stats(flags)

print("Filter Results:")
print(f"  P1 (Strict):   {stats['p1_strict']['pct']:.1f}% ({stats['p1_strict']['count']} bars)")
print(f"  P2 (Moderate): {stats['p2_moderate']['pct']:.1f}% ({stats['p2_moderate']['count']} bars)")  
print(f"  P3 (Loose):    {stats['p3_loose']['pct']:.1f}% ({stats['p3_loose']['count']} bars)")

# Check VP
sample = fbs[100]
print(f"\nVP Status:")
print(f"  POC: {sample.vp_poc_price:.1f}")
print(f"  VAL: {sample.vp_val_price:.1f}")
print(f"  VAH: {sample.vp_vah_price:.1f}")

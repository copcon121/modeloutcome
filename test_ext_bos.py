"""Quick test for EXTERNAL BOS/CHoCH - 12/11 data"""
import json
from datetime import datetime
from pathlib import Path
import sys

src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.smc_core.swing import ExternalSwingDetector
from layer2_feature_engine_v2.smc_core.structure import StructureDetector

ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251112.jsonl"  # 12/11 data!
max_bars = 500

ext_swing = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
structure = StructureDetector(GC_M1_SMC_CONFIG, tick_size=0.1)

ext_signals = []

with open(ninja_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_bars:
            break
        if not line.strip():
            continue
        
        data = json.loads(line)
        timestamp_str = data.get('timestamp', '')
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        raw_bar = RawBar(
            symbol='GC', timeframe='M1', timestamp=timestamp,
            bar_index=data.get('bar_index', i),
            o=data.get('open', 0), h=data.get('high', 0),
            l=data.get('low', 0), c=data.get('close', 0),
            volume=0, delta=0, buy_volume=0, sell_volume=0,
            best_bid=0, best_ask=0, tick_speed=0,
            aggr_buy_speed=0, aggr_sell_speed=0, price_speed=0
        )
        
        ext_state = ext_swing.update(raw_bar)
        structure.update_external(raw_bar, ext_state)
        ext_struct = structure.get_external_state()
        
        # Track any external signal
        if ext_struct['bos_up']:
            ext_signals.append(('BOS_UP', data.get('bar_index'), timestamp.strftime('%Y-%m-%d %H:%M:%S')))
        if ext_struct['bos_down']:
            ext_signals.append(('BOS_DOWN', data.get('bar_index'), timestamp.strftime('%Y-%m-%d %H:%M:%S')))
        if ext_struct['choch_up']:
            ext_signals.append(('CHOCH_UP', data.get('bar_index'), timestamp.strftime('%Y-%m-%d %H:%M:%S')))
        if ext_struct['choch_down']:
            ext_signals.append(('CHOCH_DOWN', data.get('bar_index'), timestamp.strftime('%Y-%m-%d %H:%M:%S')))

print(f"EXTERNAL BOS/CHoCH in {max_bars} bars (12/11/2025):")
print(f"Total signals: {len(ext_signals)}")
print()

# Count by type
from collections import Counter
counts = Counter(sig[0] for sig in ext_signals)
print("By type:")
for sig_type in ['BOS_UP', 'BOS_DOWN', 'CHOCH_UP', 'CHOCH_DOWN']:
    print(f"  {sig_type:12} = {counts[sig_type]:3}")
print()

# Show all signals
print("All EXTERNAL signals:")
for sig_type, bar, time in ext_signals:
    print(f"  {sig_type:12} at bar {bar:>4} | {time}")

# Check for BOS_DOWN specifically
bos_down_signals = [sig for sig in ext_signals if sig[0] == 'BOS_DOWN']
if bos_down_signals:
    print(f"\n✅ BOS_DOWN logic WORKING! Found {len(bos_down_signals)} signals")
    for sig_type, bar, time in bos_down_signals:
        print(f"   Bar {bar} at {time}")
else:
    print(f"\n⚠️  No BOS_DOWN in first {max_bars} bars")

import json

with open(r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

print(f"Total bars loaded: {len(bars)}")

# Check swing_pattern field
swings_with_pattern = [b for b in bars[:200] if b.get('bar', {}).get('swing_pattern') != 0]
print(f"\nBars with swing_pattern != 0 (first 200): {len(swings_with_pattern)}")

if swings_with_pattern:
    print("\nFirst 10 swing patterns:")
    for b in swings_with_pattern[:10]:
        pattern = b['bar']['swing_pattern']
        pattern_str = {1: "HH", 2: "HL", -1: "LL", -2: "LH", 0: "none"}.get(pattern, str(pattern))
        print(f"  Bar {b['bar_index']}: pattern={pattern} ({pattern_str})")

# Check is_swing_high / is_swing_low fields  
swing_highs = [b for b in bars[:200] if b.get('is_swing_high')]
swing_lows = [b for b in bars[:200] if b.get('is_swing_low')]

print(f"\n`is_swing_high=true` (first 200): {len(swing_highs)}")
print(f"`is_swing_low=true` (first 200): {len(swing_lows)}")

if swing_highs:
    print("\nFirst 5 swing highs:")
    for b in swing_highs[:5]:
        print(f"  Bar {b['bar_index']} @ {b['timestamp']}: high={b['high']}")

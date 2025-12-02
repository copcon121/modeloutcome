import json
from datetime import datetime

filepath = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
data = []
with open(filepath, 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total Bars: {len(data)}")
print(f"Start: {data[0]['timestamp']}")
print(f"End: {data[-1]['timestamp']}")

# Find a 1-day chunk (e.g., first 1440 bars or a specific day)
# Let's just print the first and last timestamp to decide.

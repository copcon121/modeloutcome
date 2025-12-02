
import json
import os

def normalize_ts(ts):
    # Take first 19 chars: YYYY-MM-DDTHH:MM:SS
    return ts[:19]

def load_file_a(filepath):
    data = {}
    print(f"Loading File A: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            row = json.loads(line)
            ts = normalize_ts(row['timestamp'])
            # File A is flat
            ohlc = {
                'o': row['open'],
                'h': row['high'],
                'l': row['low'],
                'c': row['close']
            }
            data[ts] = ohlc
    print(f"File A loaded: {len(data)} bars")
    return data

def load_file_b(filepath):
    data = {}
    print(f"Loading File B: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            row = json.loads(line)
            ts = normalize_ts(row['timestamp'])
            # File B has nested bar object
            bar = row['bar']
            ohlc = {
                'o': bar['o'],
                'h': bar['h'],
                'l': bar['l'],
                'c': bar['c']
            }
            data[ts] = ohlc
    print(f"File B loaded: {len(data)} bars")
    return data

def compare_data(data_a, data_b):
    print("Comparing data...")
    mismatches = 0
    missing_in_b = 0
    missing_in_a = 0
    
    # Check A against B
    for ts, ohlc_a in data_a.items():
        if ts not in data_b:
            missing_in_b += 1
            if missing_in_b <= 5:
                print(f"[Missing in B] Timestamp: {ts}")
            continue
            
        ohlc_b = data_b[ts]
        
        if (ohlc_a['o'] != ohlc_b['o'] or
            ohlc_a['h'] != ohlc_b['h'] or
            ohlc_a['l'] != ohlc_b['l'] or
            ohlc_a['c'] != ohlc_b['c']):
            
            mismatches += 1
            if mismatches <= 10:
                print(f"[Mismatch] Timestamp: {ts}")
                print(f"  A: {ohlc_a}")
                print(f"  B: {ohlc_b}")
                
    # Check B against A (for missing)
    for ts in data_b:
        if ts not in data_a:
            missing_in_a += 1
            
    print("-" * 30)
    print(f"Comparison Result:")
    print(f"Total Bars in A: {len(data_a)}")
    print(f"Total Bars in B: {len(data_b)}")
    print(f"Missing in B: {missing_in_b}")
    print(f"Missing in A: {missing_in_a}")
    print(f"OHLC Mismatches: {mismatches}")
    
    if mismatches == 0 and missing_in_b == 0 and missing_in_a == 0:
        print("SUCCESS: 100% Match!")
    else:
        print("FAILURE: Data discrepancies found.")

if __name__ == "__main__":
    file_a = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\deepseek_enhanced_GC 12-25_M1_20251006.jsonl"
    file_b = r"c:\Users\Administrator\Desktop\modeloutcome\data\raw\smc_export_gc_m1_v3_20251006.jsonl"
    
    if not os.path.exists(file_a):
        print(f"Error: File A not found: {file_a}")
    elif not os.path.exists(file_b):
        print(f"Error: File B not found: {file_b}")
    else:
        data_a = load_file_a(file_a)
        data_b = load_file_b(file_b)
        compare_data(data_a, data_b)

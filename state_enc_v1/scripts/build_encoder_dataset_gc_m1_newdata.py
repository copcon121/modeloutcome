#!/usr/bin/env python3
"""
Build Encoder Dataset for GC M1 NEW DATA (OOS Phase 3)

This script builds encoder dataset from bars_enhanced (SMC features).
Uses SAME feature spec as v1.2 - NO retraining.

Prerequisites:
    python scripts/build_gc_m1_features_newdata.py

Usage:
    python state_enc_v1/scripts/build_encoder_dataset_gc_m1_newdata.py
"""

import json
import glob
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def load_bars_enhanced(bars_enhanced_path: str) -> list:
    """Load bars_enhanced JSONL file (with SMC features)."""
    all_bars = []
    
    print(f"Loading bars_enhanced from: {bars_enhanced_path}")
    
    if not Path(bars_enhanced_path).exists():
        print(f"  WARNING: bars_enhanced not found, falling back to raw data...")
        return None
    
    with open(bars_enhanced_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                bar = json.loads(line)
                all_bars.append(bar)
            except:
                continue
    
    # Sort by time
    all_bars.sort(key=lambda b: b.get('time', b.get('bar_time', '')))
    print(f"  Loaded {len(all_bars)} enhanced bars")
    return all_bars


def load_raw_bars(raw_dir: str, pattern: str) -> list:
    """Load all raw bar files from directory (fallback)."""
    all_bars = []
    files = sorted(glob.glob(f"{raw_dir}/{pattern}"))
    
    print(f"Found {len(files)} raw data files")
    
    for filepath in files:
        print(f"  Loading: {Path(filepath).name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    bar = json.loads(line)
                    all_bars.append(bar)
                except:
                    continue
    
    # Sort by time
    all_bars.sort(key=lambda b: b.get('time', b.get('bar_time', '')))
    return all_bars


def load_feature_names(feature_config_path: str) -> list:
    """Load feature names from feature config."""
    try:
        with open(feature_config_path, 'r') as f:
            config = json.load(f)
        return config.get('feature_names', [])
    except:
        return []


def build_sequences(bars: list, seq_len: int, stride: int, feature_names: list = None) -> list:
    """Build sequences from bars."""
    sequences = []
    
    # Group by date
    by_date = defaultdict(list)
    for bar in bars:
        time_str = bar.get('time', bar.get('bar_time', ''))
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_key = dt.strftime('%Y-%m-%d')
            by_date[date_key].append(bar)
        except:
            continue
    
    print(f"Processing {len(by_date)} dates...")
    
    for date, date_bars in sorted(by_date.items()):
        if len(date_bars) < seq_len:
            continue
        
        # Build sequences with stride
        for i in range(0, len(date_bars) - seq_len + 1, stride):
            seq_bars = date_bars[i:i + seq_len]
            
            # Get metadata from last bar
            last_bar = seq_bars[-1]
            time_str = last_bar.get('time', last_bar.get('bar_time', ''))
            
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            except:
                continue
            
            # Determine session
            hour = dt.hour
            if 2 <= hour < 8:
                session = 'ASIA'
            elif 8 <= hour < 14:
                session = 'LDN'
            else:
                session = 'NY'
            
            # Get regime hint if available
            regime_hint = last_bar.get('asm_regime_hint', 0)
            
            # Build aux (future info for labels)
            aux = {
                'future_return_5': last_bar.get('future_return_5', 0),
                'future_dir_5': last_bar.get('future_dir_5', 0),
            }
            
            # Filter bars to only include required features
            if feature_names:
                filtered_seq = []
                for bar in seq_bars:
                    filtered_bar = {}
                    for fname in feature_names:
                        filtered_bar[fname] = bar.get(fname, 0.0)
                    # Keep time for reference
                    filtered_bar['time'] = bar.get('time', bar.get('bar_time', ''))
                    filtered_seq.append(filtered_bar)
                seq_bars = filtered_seq
            
            sequence = {
                'meta': {
                    'symbol': 'GC',
                    'tf': 'M1',
                    'date': date,
                    'session': session,
                    'start_time': seq_bars[0].get('time', seq_bars[0].get('bar_time', '')),
                    'end_time': time_str,
                },
                'seq': seq_bars,
                'aux': aux,
            }
            
            sequences.append(sequence)
    
    return sequences


def save_dataset(sequences: list, output_path: str):
    """Save sequences to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')


def get_stats(sequences: list) -> dict:
    """Get dataset statistics."""
    if not sequences:
        return {}
    
    dates = sorted(set(s['meta']['date'] for s in sequences))
    sessions = defaultdict(int)
    future_dirs = defaultdict(int)
    
    for s in sequences:
        sessions[s['meta']['session']] += 1
        fd = s['aux'].get('future_dir_5', 0)
        if fd > 0:
            future_dirs['up'] += 1
        elif fd < 0:
            future_dirs['down'] += 1
        else:
            future_dirs['flat'] += 1
    
    return {
        'n_samples': len(sequences),
        'date_range': f"{dates[0]} to {dates[-1]}",
        'n_dates': len(dates),
        'sessions': dict(sessions),
        'future_dir_5': dict(future_dirs),
    }


def main():
    print("=" * 80)
    print("Build Encoder Dataset for GC M1 NEW DATA (OOS Phase 3)")
    print("=" * 80)
    
    # Load config
    config_path = "state_enc_v1/configs/encoder_dataset_gc_m1_newdata_v1.2.json"
    config = load_config(config_path)
    paths = config['paths']
    dataset_cfg = config['dataset']
    
    # Try to load bars_enhanced first (preferred)
    bars_enhanced_path = paths.get('bars_enhanced', '')
    bars = load_bars_enhanced(bars_enhanced_path)
    
    # Fallback to raw bars if bars_enhanced not available
    if bars is None:
        print(f"\nFalling back to raw bars from: {paths['raw_data_dir']}")
        bars = load_raw_bars(paths['raw_data_dir'], paths['raw_pattern'])
    
    print(f"Loaded {len(bars)} bars total")
    
    if not bars:
        print("ERROR: No bars loaded!")
        sys.exit(1)
    
    # Load feature names from feature config
    feature_config_path = config.get('feature_config', '')
    feature_names = load_feature_names(feature_config_path)
    if feature_names:
        print(f"Using {len(feature_names)} features from feature config")
    
    # Build sequences
    print(f"\nBuilding sequences (len={dataset_cfg['sequence_length']}, stride={dataset_cfg['stride']})...")
    sequences = build_sequences(
        bars,
        seq_len=dataset_cfg['sequence_length'],
        stride=dataset_cfg['stride'],
        feature_names=feature_names,
    )
    print(f"Built {len(sequences)} sequences")
    
    # Save dataset
    output_path = paths['encoder_dataset']
    save_dataset(sequences, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Print stats
    stats = get_stats(sequences)
    print(f"\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Dates: {stats['n_dates']}")
    print(f"  Sessions: {stats['sessions']}")
    print(f"  Future dir 5: {stats['future_dir_5']}")
    
    print(f"\n" + "=" * 80)
    print("✅ Encoder dataset for NEW DATA built successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
